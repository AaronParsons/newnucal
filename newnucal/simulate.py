"""
Forward visibility simulation.

ForwardModel maps sky spectral coefficients → model visibilities via a 3-D
type-3 NUFFT in (l, m, eta).  The sky spectral basis ``A_sky`` (shape
``(nfreq, nmodes)``) can be any compact basis — DPSS, eigenbasis rows, or
custom modes — set via :meth:`~ForwardModel.set_sky_basis`.  This version also
exposes adjoint/backprojection helpers for dirty-map sky updates.
"""

import numpy as np
import healpy
import jax
import jax.numpy as jnp
from jax_finufft import nufft1, nufft2, nufft3
from healjax import get_interp_weights
import healjax

from .array import HERAArray
from .beam import BeamModel
from .sky import SkyModel
from .hexrect import hex_lattice_matrix, axial_grid_size

DTYPE_R = jnp.float32
DTYPE_C = jnp.complex64
DTYPE_R_NPY = np.float32

try:
    from fftvis.utils import speed_of_light as _C_import
    C = float(_C_import)
except ImportError:
    C = 299_792_458.0  # m/s


class ForwardModel:
    """
    JAX forward visibility simulation using a 3-D type-3 NUFFT.

    The sky and beam are represented in arbitrary spectral bases (currently
    DPSS). Per time step, the sky × beam product is transformed to the delay
    domain via FFT and mapped to visibilities with a single type-3 NUFFT.

    In addition to the forward model, this class provides adjoint helpers for
    dirty-map style sky updates. These helpers intentionally form the adjoint
    of the current forward operator; they are not exact inverses.
    """

    def __init__(
        self,
        array: HERAArray,
        sky_model: SkyModel,
        beam_model: BeamModel,
        freqs,
        eps: float = 1e-6,
        eta_max: float | None = None,
        eta_padding: float = 0.0,
    ):
        self.array = array
        self.sky_model = sky_model
        self.beam_model = beam_model
        self.eps = eps
        self.eta_max = eta_max
        self.eta_padding = float(eta_padding)

        freqs_np = np.asarray(freqs, dtype=np.float64)
        self.freqs = jnp.array(freqs_np, dtype=DTYPE_R)

        self.A_beam = jnp.array(beam_model.A, dtype=DTYPE_R)
        self.beam_coeffs = jnp.array(beam_model.coeffs, dtype=DTYPE_R)

        self.A_sky = jnp.array(sky_model.A, dtype=DTYPE_R)

        self.eq_coords = jnp.array(sky_model.eq_vec)

        nfreq = len(freqs_np)
        dnu = float(freqs_np[1] - freqs_np[0])
        nu_0 = float(freqs_np[0])
        m_arr = np.arange(nfreq)

        # Wrapped FFT delay grid.
        eta_np = np.fft.fftfreq(nfreq, d=dnu)
        phase_np = np.exp(-2j * np.pi * eta_np * nu_0) / nfreq

        self._eta_full = jnp.array(eta_np.astype(DTYPE_R_NPY))
        self._phase_full = jnp.array(phase_np.astype(np.complex64))

        if eta_max is None:
            eta_idx_np = np.arange(nfreq, dtype=np.int32)
        else:
            support = np.abs(eta_np) <= (float(eta_max) + self.eta_padding)
            eta_idx_np = np.where(support)[0].astype(np.int32)
            if eta_idx_np.size == 0:
                raise ValueError('Delay support mask kept zero bins.')

        self._eta_idx = jnp.array(eta_idx_np, dtype=jnp.int32)
        self._eta = self._eta_full[self._eta_idx]
        self._phase = self._phase_full[self._eta_idx]
        self.ndelay_eff = int(eta_idx_np.size)

        bls_np = np.asarray(array.bls, dtype=np.float64)
        nbls = bls_np.shape[0]
        tgt_x = (freqs_np[:, None] * bls_np[None, :, 0] / C).ravel()
        tgt_y = (freqs_np[:, None] * bls_np[None, :, 1] / C).ravel()
        tgt_z = np.repeat(freqs_np, nbls)

        self._tgt_x = jnp.array(tgt_x, dtype=DTYPE_R)
        self._tgt_y = jnp.array(tgt_y, dtype=DTYPE_R)
        self._tgt_z = jnp.array(tgt_z, dtype=DTYPE_R)

        # 2D hex-rect NUFFT path.  A_lat columns are the two physical lattice
        # basis vectors (metres); satisfies A_lat @ bl_grid[i] == bls[i,:2].
        lat_mat = hex_lattice_matrix(array)  # (2,2) float64 metres
        self._lat_mat_2d = jnp.array(lat_mat.astype(DTYPE_R_NPY), dtype=DTYPE_R)
        n_q, n_r = axial_grid_size(array.bl_grid)
        self._2d_n_q = n_q
        self._2d_n_r = n_r
        # FINUFFT shifted-centre convention: mode k is at array index k + N//2.
        bl_grid_np = np.asarray(array.bl_grid[:, :2], dtype=np.int32)
        self._2d_q_idx = jnp.array(bl_grid_np[:, 0] + n_q // 2, dtype=jnp.int32)
        self._2d_r_idx = jnp.array(bl_grid_np[:, 1] + n_r // 2, dtype=jnp.int32)

        self._jit_one = jax.jit(self._simulate_one_precomputed)
        self._jit_one_2d = jax.jit(self._simulate_one_2d_precomputed_sky_spec)

        self._geom_ready = False
        self._topo_all = None
        self._horizon_all = None
        self._beam_spec_horizon_all = None
        self._src_x_all = None
        self._src_y_all = None
        self._src_z_all = None
        self._xi_all = None

    # ------------------------------------------------------------------
    # Pixel masking helpers
    # ------------------------------------------------------------------

    def build_ever_visible_mask(self, rot_matrices):
        """
        Return a boolean array of shape (npix_sky,) that is True for every
        sky pixel above the horizon (topo_z > 0) at any of the given times.

        Call this before :meth:`apply_pixel_mask`.
        """
        rot_ms = np.asarray(rot_matrices, dtype=DTYPE_R_NPY)
        eq = np.asarray(self.eq_coords)   # (3, npix)
        ever_visible = np.zeros(eq.shape[1], dtype=bool)
        for i in range(rot_ms.shape[0]):
            ever_visible |= (rot_ms[i] @ eq)[2] > 0
        return ever_visible

    def apply_pixel_mask(self, mask):
        """
        Permanently restrict the sky model to the pixels selected by *mask*.

        After this call ``npix_sky`` equals ``mask.sum()`` and sky-coefficient
        arrays must have that many rows.  The precomputed geometry is
        invalidated; call :meth:`precompute_time_geometry` again.

        Parameters
        ----------
        mask : array_like of bool, shape (npix_sky_full,)
        """
        mask = np.asarray(mask, dtype=bool)
        self._pixel_mask = mask
        self._pixel_indices = np.where(mask)[0].astype(np.int32)
        eq_full = np.asarray(self.eq_coords)   # (3, npix_full)
        self.eq_coords = jnp.array(eq_full[:, mask], dtype=DTYPE_R)
        # Invalidate geometry
        self._geom_ready = False
        for attr in ('_topo_all', '_horizon_all', '_beam_spec_horizon_all',
                     '_src_x_all', '_src_y_all', '_src_z_all', '_xi_all'):
            setattr(self, attr, None)
        return self

    def precompute_time_geometry(self, rot_matrices):
        """Precompute time-only geometry and interpolation operators.

        Parameters
        ----------
        rot_matrices : array_like, shape (ntime, 3, 3)
        """
        rot_ms = jnp.array(rot_matrices, dtype=DTYPE_R)
        ntime = int(rot_ms.shape[0])
        npix = self.npix_sky
        ndelay = self.ndelay_eff
        two_pi = 2.0 * np.pi

        topo_all = []
        horizon_all = []
        beam_spec_horizon_all = []
        src_x_all = []
        src_y_all = []
        src_z_all = []
        xi_all = []
        interp_px_all = []
        interp_wgt_all = []

        # Extract beam arrays once outside the loop for efficiency
        bc_np = np.asarray(self.beam_coeffs)   # (npix_beam, nmodes_beam)
        ab_np = np.asarray(self.A_beam)         # (nfreq, nmodes_beam)
        eta_np = np.asarray(self._eta, dtype=DTYPE_R_NPY)
        lat_mat_np = np.asarray(self._lat_mat_2d, dtype=DTYPE_R_NPY)  # (2,2) metres
        for i in range(ntime):
            topo = np.array(rot_ms[i] @ self.eq_coords, dtype=DTYPE_R_NPY)
            horizon = (topo[2] > 0).astype(DTYPE_R_NPY)
            topo_th, topo_ph = healjax.vec2ang(
                jnp.array(topo[0]), jnp.array(topo[1]), jnp.array(topo[2])
            )
            px, wgts = get_interp_weights(topo_th, topo_ph, self.beam_model.nside)
            px = np.array(px, dtype=np.int32)
            wgts = np.array(wgts, dtype=DTYPE_R_NPY)

            # Store interpolation stencil for beam solving (update_beam_cache /
            # simulate_variable_beam).  px/wgts shape: (4, npix_sky).
            interp_px_all.append(px)
            interp_wgt_all.append(wgts)

            # Beam spec × horizon for all pixels.  This precomputed array is
            # used by _simulate_one_precomputed and apparent_sky_update_one_time,
            # removing the beam gather and matmul from the JIT hot path.
            bi_np = np.sum(wgts[:, :, None] * bc_np[px], axis=0)   # (npix_sky, nmodes_beam)
            bs_np = (bi_np @ ab_np.T) * horizon[:, None]             # (npix_sky, nfreq)
            beam_spec_horizon_all.append(bs_np)

            src_x = np.repeat(topo[0], ndelay).astype(DTYPE_R_NPY) * two_pi
            src_y = np.repeat(topo[1], ndelay).astype(DTYPE_R_NPY) * two_pi
            src_z = np.tile(eta_np, npix).astype(DTYPE_R_NPY) * two_pi

            # 2D path: axial sky coordinates xi = A_lat.T @ topo[:2]
            xi = (lat_mat_np.T @ topo[:2, :]).astype(DTYPE_R_NPY)   # (2, npix_sky)

            topo_all.append(topo)
            horizon_all.append(horizon)
            src_x_all.append(src_x)
            src_y_all.append(src_y)
            src_z_all.append(src_z)
            xi_all.append(xi)

        self._topo_all = jnp.array(np.stack(topo_all), dtype=DTYPE_R)
        self._horizon_all = jnp.array(np.stack(horizon_all), dtype=DTYPE_R)
        self._beam_spec_horizon_all = jnp.array(np.stack(beam_spec_horizon_all), dtype=DTYPE_R)
        self._src_x_all = jnp.array(np.stack(src_x_all), dtype=DTYPE_R)
        self._src_y_all = jnp.array(np.stack(src_y_all), dtype=DTYPE_R)
        self._src_z_all = jnp.array(np.stack(src_z_all), dtype=DTYPE_R)
        self._xi_all = jnp.array(np.stack(xi_all), dtype=DTYPE_R)   # (ntime, 2, npix_sky)
        # Bilinear stencil for beam solving — list of (4, npix_sky) numpy arrays,
        # one per time step.  Kept as a list so that beam solving can be done with
        # a per-time Python loop without stacking into a single large array.
        self._interp_px_all  = interp_px_all   # list[np.ndarray(4, npix_sky), int32]
        self._interp_wgt_all = interp_wgt_all  # list[np.ndarray(4, npix_sky), float32]

        self._geom_ready = True
        # Invalidate any cached JIT compilation: the precomputed arrays are
        # captured by value in the trace, so a fresh jit is needed after each
        # geometry recompute to avoid stale compiled code.
        self._jit_one = jax.jit(self._simulate_one_precomputed)
        self._jit_one_2d = jax.jit(self._simulate_one_2d_precomputed_sky_spec)
        return self

    def _forward_components_from_cache_sky_spec(self, sky_spec, tind):
        topo = self._topo_all[tind]
        horizon = self._horizon_all[tind]
        beam_spec_h = self._beam_spec_horizon_all[tind]   # (npix_sky, nfreq), already * horizon
        W = sky_spec * beam_spec_h
        W_eta_full = jnp.fft.fft(W, axis=1) * self._phase_full[None, :]
        W_eta = W_eta_full[:, self._eta_idx]
        return topo, horizon, beam_spec_h, W, W_eta

    def _forward_components_from_cache(self, sky_coeffs, tind):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._forward_components_from_cache_sky_spec(sky_spec, tind)

    def forward_components_one_time(self, sky_coeffs, tind):
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        return self._forward_components_from_cache(sky_coeffs, tind)

    def _simulate_one_precomputed_sky_spec(self, sky_spec, tind):
        _, _, _, _, W_eta = self._forward_components_from_cache_sky_spec(sky_spec, tind)
        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C),
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _simulate_one_precomputed(self, sky_coeffs, tind):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_precomputed_sky_spec(sky_spec, tind)

    def simulate(self, sky_coeffs, rot_matrices):
        if (not self._geom_ready) or (self._topo_all.shape[0] != rot_matrices.shape[0]):
            self.precompute_time_geometry(rot_matrices)
        sky_spec = sky_coeffs @ self.A_sky.T
        inds = jnp.arange(rot_matrices.shape[0], dtype=jnp.int32)
        return jax.vmap(lambda tind: self._simulate_one_precomputed_sky_spec(sky_spec, tind))(inds)

    def adjoint_residual_one_time(self, residual_fb, tind):
        """Backproject one-time residuals to a pixel-delay dirty map."""
        c = jnp.conj(residual_fb.reshape(-1).astype(DTYPE_C))
        dirty_flat = nufft3(
            c,
            self._tgt_x, self._tgt_y, self._tgt_z,
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            iflag=1,
            eps=self.eps,
        )
        dirty = jnp.conj(dirty_flat.reshape(self.npix_sky, self.ndelay_eff))
        return dirty / (self.nfreq * self.nbls)

    def dirty_apparent_sky_one_time(self, residual_fb, tind):
        """Backproject one-time residuals to dirty apparent sky in frequency space."""
        dirty_eta = self.adjoint_residual_one_time(residual_fb, tind)
        tmp = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C)
        tmp = tmp.at[:, self._eta_idx].set(dirty_eta / self._phase[None, :])
        dirty_pf = jnp.fft.ifft(tmp, axis=1)
        return dirty_pf

    def apparent_sky_update_one_time(
        self,
        residual_fb,
        tind,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Form a beam-weighted apparent-sky correction for one time."""
        dirty_pf = self.dirty_apparent_sky_one_time(residual_fb, tind)
        beam_spec_h = self._beam_spec_horizon_all[tind]   # (npix_sky, nfreq), already * horizon
        weight = jnp.abs(beam_spec_h) ** 2
        delta_app = step_size * jnp.conj(beam_spec_h.astype(DTYPE_C)) * dirty_pf / (weight.astype(DTYPE_C) + beam_reg)
        return delta_app, weight

    def _accumulate_equatorial_sky_update_impl(self, residual_vis, sky_update_fn, step_size, beam_reg):
        inds = jnp.arange(int(residual_vis.shape[0]), dtype=jnp.int32)

        def one_time(resid_t, tind):
            return sky_update_fn(resid_t, tind, step_size=step_size, beam_reg=beam_reg)

        delta_all, weight_all = jax.vmap(one_time)(residual_vis, inds)
        num = jnp.sum(weight_all.astype(DTYPE_C) * delta_all, axis=0)
        den = jnp.sum(weight_all.real.astype(DTYPE_R), axis=0)
        delta_eq_pf = num / (den.astype(DTYPE_C) + 1e-6)
        delta_eq_pf = delta_eq_pf / residual_vis.shape[0]
        return (delta_eq_pf.real @ self.A_sky).astype(DTYPE_R)

    def accumulate_equatorial_sky_update(
        self,
        residual_vis,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Accumulate a global equatorial sky update from all times."""
        return self._accumulate_equatorial_sky_update_impl(
            residual_vis, self.apparent_sky_update_one_time, step_size, beam_reg
        )

    def _accumulate_beam_update_impl(self, sky_coeffs, residual_vis, dirty_fn, step_size, sky_reg):
        ab_np = np.asarray(self.A_beam, dtype=DTYPE_R_NPY)

        sky_spec = np.asarray((jnp.array(sky_coeffs) @ self.A_sky.T).astype(DTYPE_C))
        sky_weight = (sky_spec.real ** 2 + sky_spec.imag ** 2).astype(DTYPE_R_NPY)
        sky_pow = sky_weight.sum(axis=1)

        num = np.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_NPY)
        den = np.zeros(self.npix_beam, dtype=DTYPE_R_NPY)

        ntime = int(residual_vis.shape[0])
        for tind in range(ntime):
            dirty_pf = np.asarray(dirty_fn(residual_vis[tind], tind))
            horizon = np.asarray(self._horizon_all[tind])

            delta_bsf = step_size * np.conj(sky_spec) * dirty_pf / (sky_weight + sky_reg)
            delta_bi = (np.real(delta_bsf) * horizon[:, None]) @ ab_np

            px  = self._interp_px_all[tind]
            wgt = self._interp_wgt_all[tind]

            px_flat = px.reshape(-1)
            w_flat = (wgt * sky_pow[None, :]).reshape(-1)
            contrib = (
                w_flat[:, None]
                * np.repeat(delta_bi[None, :, :], 4, axis=0).reshape(-1, self.nmodes_beam)
            )

            np.add.at(num, px_flat, contrib)
            np.add.at(den, px_flat, w_flat)

        delta_bc = num / (den[:, None] + 1e-6)
        delta_bc /= ntime
        return jnp.array(delta_bc, dtype=DTYPE_R)

    def accumulate_beam_update(
        self,
        sky_coeffs,
        residual_vis,
        step_size: float = 1.0,
        sky_reg: float = 1e-3,
    ):
        """Accumulate a global beam-coefficient update from all times.

        Dual of :meth:`accumulate_equatorial_sky_update`.  With sky and gains
        held fixed, this backprojects the gain-calibrated residuals through the
        sky model to estimate the beam-coefficient correction that best reduces
        the residual.
        """
        return self._accumulate_beam_update_impl(
            sky_coeffs, residual_vis, self.dirty_apparent_sky_one_time, step_size, sky_reg
        )

    # ------------------------------------------------------------------
    # Beam update helpers
    # ------------------------------------------------------------------

    def beam_spec_horizon_from_coeffs(self, beam_coeffs):
        """Return beam_spec × horizon arrays for explicit *beam_coeffs*.

        Useful for beam-optimization code paths that want to evaluate candidate
        beams without mutating the cached beam state.  Shape is
        ``(ntime, npix_sky, nfreq)``.
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        bc_np = np.asarray(beam_coeffs, dtype=DTYPE_R_NPY)
        ab_np = np.asarray(self.A_beam)
        beam_spec_horizon_all = []
        for i in range(len(self._interp_px_all)):
            px   = self._interp_px_all[i]
            wgt  = self._interp_wgt_all[i]
            horiz = np.asarray(self._horizon_all[i])
            bi = np.sum(wgt[:, :, None] * bc_np[px], axis=0)
            bs = (bi @ ab_np.T) * horiz[:, None]
            beam_spec_horizon_all.append(bs)
        return jnp.array(np.stack(beam_spec_horizon_all), dtype=DTYPE_R)

    def update_beam_cache(self, beam_coeffs):
        """Recompute :attr:`_beam_spec_horizon_all` from new *beam_coeffs*.

        Cheaper than a full :meth:`precompute_time_geometry` call because the
        NUFFT source coordinates and interpolation stencil are already stored.
        After this call :attr:`_jit_one` is recompiled so subsequent
        :meth:`simulate` calls use the updated beam.

        Parameters
        ----------
        beam_coeffs : array_like, shape (npix_beam, nmodes_beam)
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        bc_np = np.asarray(beam_coeffs, dtype=DTYPE_R_NPY)
        ab_np = np.asarray(self.A_beam)
        ntime = len(self._interp_px_all)
        beam_spec_horizon_all = []
        for i in range(ntime):
            px   = self._interp_px_all[i]   # (4, npix_sky)
            wgt  = self._interp_wgt_all[i]  # (4, npix_sky)
            horiz = np.asarray(self._horizon_all[i])  # (npix_sky,)
            bi = np.sum(wgt[:, :, None] * bc_np[px], axis=0)  # (npix_sky, nmodes_beam)
            bs = (bi @ ab_np.T) * horiz[:, None]               # (npix_sky, nfreq)
            beam_spec_horizon_all.append(bs)
        self._beam_spec_horizon_all = jnp.array(
            np.stack(beam_spec_horizon_all), dtype=DTYPE_R
        )
        self.beam_coeffs = jnp.array(bc_np, dtype=DTYPE_R)
        self.beam_model.coeffs = bc_np
        # Invalidate compiled simulations so the new precomputed beam is used.
        self._jit_one = jax.jit(self._simulate_one_precomputed)
        self._jit_one_2d = jax.jit(self._simulate_one_2d_precomputed_sky_spec)

    def _simulate_one_variable_beam_sky_spec(self, sky_spec, beam_coeffs, tind):
        """Forward model for one time step with *beam_coeffs* as a traced input."""
        px  = self._interp_px_all[tind]
        wgt = jnp.array(self._interp_wgt_all[tind])
        bi = jnp.sum(wgt[:, :, None] * beam_coeffs[px], axis=0)
        beam_spec_h = (bi @ self.A_beam.T) * self._horizon_all[tind][:, None]
        W = sky_spec * beam_spec_h
        W_eta_full = jnp.fft.fft(W, axis=1) * self._phase_full[None, :]
        W_eta = W_eta_full[:, self._eta_idx]
        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C),
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _simulate_one_variable_beam(self, sky_coeffs, beam_coeffs, tind):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_variable_beam_sky_spec(sky_spec, beam_coeffs, tind)

    def simulate_variable_beam(self, sky_coeffs, beam_coeffs, rot_matrices):
        """Simulate all time steps with *beam_coeffs* as a differentiable input."""
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        ntime = len(self._interp_px_all)
        sky_spec = sky_coeffs @ self.A_sky.T
        return jnp.stack([
            self._simulate_one_variable_beam_sky_spec(sky_spec, beam_coeffs, t)
            for t in range(ntime)
        ])

    # ------------------------------------------------------------------
    # 2D hex-rect NUFFT path — forward model
    # ------------------------------------------------------------------

    def _nufft2d_W_to_vis(self, W, xi, freqs):
        """Apply per-frequency type-1 2D NUFFTs to a (npix_sky, nfreq) weight array."""
        n_q = self._2d_n_q
        n_r = self._2d_n_r
        q_idx = self._2d_q_idx
        r_idx = self._2d_r_idx

        def one_freq(args):
            W_fi, nu = args
            x_j = (2.0 * jnp.pi * nu / C * xi[0]).astype(DTYPE_R)
            y_j = (2.0 * jnp.pi * nu / C * xi[1]).astype(DTYPE_R)
            grid = nufft1((n_q, n_r), W_fi, x_j, y_j, iflag=1, eps=self.eps)
            return grid[q_idx, r_idx]

        return jax.vmap(one_freq)((W.T, freqs))   # (nfreq, nbls)

    def _simulate_one_2d_precomputed_sky_spec(self, sky_spec, tind):
        """2D hex-rect NUFFT forward for one time step (cached beam).

        Parameters
        ----------
        sky_spec : jnp.ndarray, shape (npix_sky, nfreq)
        tind     : int

        Returns
        -------
        vis : jnp.ndarray, shape (nfreq, nbls), complex64
        """
        beam_spec_h = self._beam_spec_horizon_all[tind]
        W = (sky_spec * beam_spec_h).astype(DTYPE_C)
        xi = self._xi_all[tind]
        return self._nufft2d_W_to_vis(W, xi, self.freqs)

    def _simulate_one_2d_precomputed(self, sky_coeffs, tind):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_2d_precomputed_sky_spec(sky_spec, tind)

    def _simulate_one_2d_variable_beam_sky_spec(self, sky_spec, beam_coeffs, tind):
        """2D forward for one time step with *beam_coeffs* as a traced input."""
        px  = self._interp_px_all[tind]
        wgt = jnp.array(self._interp_wgt_all[tind])
        bi = jnp.sum(wgt[:, :, None] * beam_coeffs[px], axis=0)
        beam_spec_h = (bi @ self.A_beam.T) * self._horizon_all[tind][:, None]
        W = (sky_spec * beam_spec_h).astype(DTYPE_C)
        xi = self._xi_all[tind]
        return self._nufft2d_W_to_vis(W, xi, self.freqs)

    def simulate_2d(self, sky_coeffs, rot_matrices):
        """Simulate all times using the 2D hex-rect NUFFT path.

        Each frequency is handled by a type-1 2D NUFFT that maps sky × beam
        weighted pixels to a compact uniform uv grid; baselines are read from
        that grid at their integer axial positions.

        The channel spacing of *freqs* should satisfy
        :func:`~newnucal.hexrect.critical_channel_spacing` to avoid aliasing.

        Returns
        -------
        vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex64
        """
        if (not self._geom_ready) or (self._xi_all is None) or (self._topo_all.shape[0] != rot_matrices.shape[0]):
            self.precompute_time_geometry(rot_matrices)
        sky_spec = sky_coeffs @ self.A_sky.T
        inds = jnp.arange(rot_matrices.shape[0], dtype=jnp.int32)
        return jax.vmap(
            lambda tind: self._simulate_one_2d_precomputed_sky_spec(sky_spec, tind)
        )(inds)

    def simulate_variable_beam_2d(self, sky_coeffs, beam_coeffs, rot_matrices):
        """Simulate all times with explicit *beam_coeffs* via the 2D path."""
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        ntime = len(self._interp_px_all)
        sky_spec = sky_coeffs @ self.A_sky.T
        return jnp.stack([
            self._simulate_one_2d_variable_beam_sky_spec(sky_spec, beam_coeffs, t)
            for t in range(ntime)
        ])

    # ------------------------------------------------------------------
    # 2D hex-rect NUFFT path — adjoint / dirty-map helpers
    # ------------------------------------------------------------------

    def adjoint_residual_one_time_2d(self, residual_fb, tind):
        """Backproject one-time residuals to a pixel × frequency dirty map.

        Adjoint of the 2D forward: scatters baseline residuals to the uniform
        uv grid then applies a type-2 NUFFT to evaluate at sky pixel positions.
        Returns a complex ``(npix_sky, nfreq)`` array already in frequency
        domain — no IFFT is needed (unlike the 3D path).

        Parameters
        ----------
        residual_fb : jnp.ndarray, shape (nfreq, nbls), complex64
        tind        : int

        Returns
        -------
        dirty_pf : jnp.ndarray, shape (npix_sky, nfreq), complex64
        """
        xi = self._xi_all[tind]    # (2, npix_sky)
        n_q = self._2d_n_q
        n_r = self._2d_n_r
        q_idx = self._2d_q_idx    # (nbls,) shifted
        r_idx = self._2d_r_idx    # (nbls,) shifted
        freqs = self.freqs         # (nfreq,)
        resid = residual_fb.astype(DTYPE_C)   # (nfreq, nbls)

        def one_freq(args):
            resid_f, nu = args                                          # (nbls,), scalar
            # Scatter residuals to (n_q, n_r) grid at shifted bl positions
            grid = jnp.zeros((n_q, n_r), dtype=DTYPE_C).at[q_idx, r_idx].set(resid_f)
            x_j = (2.0 * jnp.pi * nu / C * xi[0]).astype(DTYPE_R)     # (npix_sky,)
            y_j = (2.0 * jnp.pi * nu / C * xi[1]).astype(DTYPE_R)
            # nufft2 with iflag=-1 is the adjoint of nufft1 with iflag=+1
            return nufft2(grid, x_j, y_j, iflag=-1, eps=self.eps)      # (npix_sky,)

        dirty_freq_first = jax.vmap(one_freq)((resid, freqs))  # (nfreq, npix_sky)
        return (dirty_freq_first.T / (self.nfreq * self.nbls))     # (npix_sky, nfreq)

    def dirty_apparent_sky_one_time_2d(self, residual_fb, tind):
        """Backproject to dirty apparent sky in frequency domain (2D path).

        Unlike the 3D path, the 2D adjoint already returns a frequency-domain
        quantity, so no IFFT is needed.  This method is provided for API
        symmetry with :meth:`dirty_apparent_sky_one_time`.
        """
        return self.adjoint_residual_one_time_2d(residual_fb, tind)

    def apparent_sky_update_one_time_2d(
        self,
        residual_fb,
        tind,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Beam-weighted apparent-sky correction for one time (2D path)."""
        dirty_pf = self.dirty_apparent_sky_one_time_2d(residual_fb, tind)
        beam_spec_h = self._beam_spec_horizon_all[tind]   # (npix_sky, nfreq)
        weight = jnp.abs(beam_spec_h) ** 2
        delta_app = (
            step_size
            * jnp.conj(beam_spec_h.astype(DTYPE_C))
            * dirty_pf
            / (weight.astype(DTYPE_C) + beam_reg)
        )
        return delta_app, weight

    def accumulate_equatorial_sky_update_2d(
        self,
        residual_vis,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Accumulate a global equatorial sky update from all times (2D path).

        Mirrors :meth:`accumulate_equatorial_sky_update` but uses the 2D adjoint.
        """
        return self._accumulate_equatorial_sky_update_impl(
            residual_vis, self.apparent_sky_update_one_time_2d, step_size, beam_reg
        )

    def accumulate_beam_update_2d(
        self,
        sky_coeffs,
        residual_vis,
        step_size: float = 1.0,
        sky_reg: float = 1e-3,
    ):
        """Accumulate a global beam-coefficient update from all times (2D path).

        Mirrors :meth:`accumulate_beam_update` but uses the 2D adjoint.
        """
        return self._accumulate_beam_update_impl(
            sky_coeffs, residual_vis, self.dirty_apparent_sky_one_time_2d, step_size, sky_reg
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npix_sky(self) -> int:
        # eq_coords has shape (3, npix_active); reflects any applied pixel mask
        return int(self.eq_coords.shape[1])

    @property
    def npix_beam(self) -> int:
        return int(self.beam_coeffs.shape[0])

    @property
    def nmodes_beam(self) -> int:
        return int(self.beam_coeffs.shape[1])

    @property
    def nfreq(self) -> int:
        return int(self.freqs.shape[0])

    @property
    def nbls(self) -> int:
        return self.array.nbls


# ---------------------------------------------------------------------------
# Coordinate rotation helpers
# ---------------------------------------------------------------------------

def compute_rotation_matrices(times, location):
    from matvis.cpu.coords import CoordinateRotationERFA
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    dummy_sc = SkyCoord(ra=[0.0] * u.deg, dec=[location.lat.deg] * u.deg, frame="icrs")
    crds = CoordinateRotationERFA(
        skycoords=dummy_sc,
        times=times,
        telescope_loc=location,
        flux=np.ones(1),
        update_bcrs_every=1e9,
        precision=2,
    )
    crds.setup()

    rot_ms = np.empty((len(times), 3, 3), dtype=DTYPE_R_NPY)
    for i, t in enumerate(times):
        rot_ms[i] = _erfa_rot_matrix(crds, t)
    return rot_ms


def _erfa_rot_matrix(crds, t):
    obsf = crds._get_obsf(t, crds.telescope_loc)
    astrom = crds._apco(obsf)

    ce = np.cos(astrom["eral"])
    se = np.sin(astrom["eral"])
    c2h = np.array([[ce, se, 0], [-se, ce, 0], [0, 0, 1]])

    sx = np.sin(astrom["xpl"])
    cx = np.cos(astrom["xpl"])
    sy = np.sin(astrom["ypl"])
    cy = np.cos(astrom["ypl"])
    pm = np.array(
        [[cx, 0, sx], [sx * sy, cy, -cx * sy], [-sx * cy, sy, cx * cy]]
    )

    enu = np.array(
        [
            [0, 1, 0],
            [-astrom["sphi"], 0, astrom["cphi"]],
            [astrom["cphi"], 0, astrom["sphi"]],
        ]
    )
    return enu @ pm @ c2h
