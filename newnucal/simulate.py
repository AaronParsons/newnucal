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
from jax_finufft import nufft3
from healjax import get_interp_weights
import healjax

from .array import HERAArray
from .beam import BeamModel
from .sky import SkyModel

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

        self._jit_one = jax.jit(self._simulate_one_precomputed)

        self._geom_ready = False
        self._topo_all = None
        self._horizon_all = None
        self._beam_spec_horizon_all = None
        self._src_x_all = None
        self._src_y_all = None
        self._src_z_all = None

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
                     '_src_x_all', '_src_y_all', '_src_z_all'):
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
        interp_px_all = []
        interp_wgt_all = []

        # Extract beam arrays once outside the loop for efficiency
        bc_np = np.asarray(self.beam_coeffs)   # (npix_beam, nmodes_beam)
        ab_np = np.asarray(self.A_beam)         # (nfreq, nmodes_beam)
        eta_np = np.asarray(self._eta, dtype=DTYPE_R_NPY)
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

            topo_all.append(topo)
            horizon_all.append(horizon)
            src_x_all.append(src_x)
            src_y_all.append(src_y)
            src_z_all.append(src_z)

        self._topo_all = jnp.array(np.stack(topo_all), dtype=DTYPE_R)
        self._horizon_all = jnp.array(np.stack(horizon_all), dtype=DTYPE_R)
        self._beam_spec_horizon_all = jnp.array(np.stack(beam_spec_horizon_all), dtype=DTYPE_R)
        self._src_x_all = jnp.array(np.stack(src_x_all), dtype=DTYPE_R)
        self._src_y_all = jnp.array(np.stack(src_y_all), dtype=DTYPE_R)
        self._src_z_all = jnp.array(np.stack(src_z_all), dtype=DTYPE_R)
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
        return self

    def _forward_components_from_cache(self, sky_coeffs, tind):
        topo = self._topo_all[tind]
        horizon = self._horizon_all[tind]
        beam_spec_h = self._beam_spec_horizon_all[tind]   # (npix_sky, nfreq), already * horizon
        sky_spec = sky_coeffs @ self.A_sky.T               # (npix_sky, nfreq)
        W = sky_spec * beam_spec_h
        W_eta_full = jnp.fft.fft(W, axis=1) * self._phase_full[None, :]
        W_eta = W_eta_full[:, self._eta_idx]
        return topo, horizon, beam_spec_h, W, W_eta

    def forward_components_one_time(self, sky_coeffs, tind):
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        return self._forward_components_from_cache(sky_coeffs, tind)

    def _simulate_one_precomputed(self, sky_coeffs, tind):
        _, _, _, _, W_eta = self._forward_components_from_cache(sky_coeffs, tind)
        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C),
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def simulate(self, sky_coeffs, rot_matrices):
        if (not self._geom_ready) or (self._topo_all.shape[0] != rot_matrices.shape[0]):
            self.precompute_time_geometry(rot_matrices)
        inds = jnp.arange(rot_matrices.shape[0], dtype=jnp.int32)
        return jax.vmap(lambda tind: self._jit_one(sky_coeffs, tind))(inds)

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

    def accumulate_equatorial_sky_update(
        self,
        residual_vis,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Accumulate a global equatorial sky update from all times."""
        num = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C)
        den = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_R)
        for tind in range(int(residual_vis.shape[0])):
            delta_eq, weight_eq = self.apparent_sky_update_one_time(
                residual_vis[tind], tind, step_size=step_size, beam_reg=beam_reg
            )
            num = num + weight_eq.astype(DTYPE_C) * delta_eq
            den = den + weight_eq.real.astype(DTYPE_R)
        delta_eq_pf = num / (den.astype(DTYPE_C) + 1e-6)
        delta_eq_pf = delta_eq_pf / residual_vis.shape[0]  # divide by ntimes
        delta_coeffs = (delta_eq_pf.real @ self.A_sky).astype(DTYPE_R)
        return delta_coeffs

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

        Math sketch (one time step *t*, pixel *p*, frequency *f*):

        * Backproject: ``dirty[p,f] = NUFFT^H(r[t]) → IFFT → pixel×freq``
        * Matched filter in sky: ``δb[p,f] = step_size · sky^* · dirty / (|sky|² + reg)``
        * Project to beam DPSS: ``δbᵢ[p,m] = real(δb[p,:]) · horizon[p] @ A_beam``
        * Adjoint scatter to beam pixels:
          ``δbc[q,m] += wgt[k,p] · |sky|² · δbᵢ[p,m]``  for each neighbour *k*

        Accumulate numerator/denominator over all times, then normalise.

        Parameters
        ----------
        sky_coeffs : array_like, shape (npix_sky, nmodes_sky)
        residual_vis : array_like, shape (ntime, nfreq, nbls)
            Gain-calibrated residual ``data_cal − vis_model`` (no gain factor).
        step_size : float
        sky_reg : float
            Regularisation added to ``|sky_spec|²`` denominator (prevents
            blow-up where the sky is faint).

        Returns
        -------
        delta_bc : jnp.ndarray, shape (npix_beam, nmodes_beam)
        """
        ab_np = np.asarray(self.A_beam, dtype=DTYPE_R_NPY)   # (nfreq, nmodes_beam)

        # Sky spectrum in pixel × frequency space (complex for matched filter)
        sky_spec = np.asarray(
            (jnp.array(sky_coeffs) @ self.A_sky.T).astype(DTYPE_C)
        )                                                    # (npix_sky, nfreq)
        sky_weight = (sky_spec.real ** 2 + sky_spec.imag ** 2).astype(
            DTYPE_R_NPY
        )                                                    # |sky_spec|² (npix_sky, nfreq)
        # Per-pixel sky power (summed over frequency) used to weight the scatter
        sky_pow = sky_weight.sum(axis=1)                     # (npix_sky,)

        num = np.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_NPY)
        den = np.zeros(self.npix_beam, dtype=DTYPE_R_NPY)

        ntime = int(residual_vis.shape[0])
        for tind in range(ntime):
            # Backproject residuals to apparent pixel×frequency map (complex)
            dirty_pf = np.asarray(
                self.dirty_apparent_sky_one_time(residual_vis[tind], tind)
            )                                                # (npix_sky, nfreq)
            horizon = np.asarray(self._horizon_all[tind])   # (npix_sky,)

            # Matched-filter update in apparent-beam space
            delta_bsf = (
                step_size * np.conj(sky_spec) * dirty_pf / (sky_weight + sky_reg)
            )                                                # (npix_sky, nfreq) complex

            # Project real part onto beam DPSS, masked by horizon
            delta_bi = (np.real(delta_bsf) * horizon[:, None]) @ ab_np
            # (npix_sky, nmodes_beam)

            # Adjoint scatter-add: sky pixel → its 4 beam-pixel neighbours
            px  = self._interp_px_all[tind]   # (4, npix_sky) int32
            wgt = self._interp_wgt_all[tind]  # (4, npix_sky) float32
            for k in range(4):
                w_k = wgt[k] * sky_pow         # (npix_sky,) scatter weight
                np.add.at(num, px[k], w_k[:, None] * delta_bi)
                np.add.at(den, px[k], w_k)

        delta_bc = num / (den[:, None] + 1e-6)
        delta_bc /= ntime
        return jnp.array(delta_bc, dtype=DTYPE_R)

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
        # Invalidate compiled simulation so the new precomputed beam is used.
        self._jit_one = jax.jit(self._simulate_one_precomputed)

    def _simulate_one_variable_beam(self, sky_coeffs, beam_coeffs, tind):
        """Forward model for one time step with *beam_coeffs* as a traced input.

        Differentiable w.r.t. both *sky_coeffs* and *beam_coeffs* via JAX AD.
        Uses the bilinear stencil stored during :meth:`precompute_time_geometry`.
        Only supports the non-zenith-cut path.
        """
        px  = self._interp_px_all[tind]   # numpy (4, npix_sky) — static indices
        wgt = jnp.array(self._interp_wgt_all[tind])  # (4, npix_sky)
        # Interpolate beam coeffs to sky-pixel positions.
        # beam_coeffs[px] gathers with static indices — JAX differentiates via scatter-add.
        bi = jnp.sum(wgt[:, :, None] * beam_coeffs[px], axis=0)   # (npix_sky, nmodes_beam)
        beam_spec_h = (bi @ self.A_beam.T) * self._horizon_all[tind][:, None]  # (npix_sky, nfreq)
        sky_spec = sky_coeffs @ self.A_sky.T                        # (npix_sky, nfreq)
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

    def simulate_variable_beam(self, sky_coeffs, beam_coeffs, rot_matrices):
        """Simulate all time steps with *beam_coeffs* as a differentiable input.

        Unlike :meth:`simulate`, this path computes the beam interpolation
        on-the-fly inside the computation graph, making the output differentiable
        w.r.t. *beam_coeffs*.  The Python loop is unrolled by JAX when JIT-compiled
        (e.g. by jaxopt), so no Python overhead appears in gradient computation.

        The zenith-cut optimisation is not supported here; the full pixel set is
        always used.

        Parameters
        ----------
        sky_coeffs : array_like, shape (npix_sky, nmodes_sky)
        beam_coeffs : array_like, shape (npix_beam, nmodes_beam)
        rot_matrices : array_like, shape (ntime, 3, 3)  — passed for API symmetry

        Returns
        -------
        vis : jnp.ndarray, shape (ntime, nfreq, nbls)
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        ntime = len(self._interp_px_all)
        return jnp.stack([
            self._simulate_one_variable_beam(sky_coeffs, beam_coeffs, t)
            for t in range(ntime)
        ])

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
