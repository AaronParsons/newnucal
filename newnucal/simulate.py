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
from jax_finufft.options import Opts as _NufftOpts
from healjax import get_interp_weights
import healjax

from .array import HERAArray
from .beam import BeamModel
from .sky import SkyModel
from .hexrect import hex_lattice_matrix, axial_grid_size
from .utils import DTYPE_R_JAX, DTYPE_C_JAX, DTYPE_R_NPY, C


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
        nufft_upsampfac: float = 1.25,
        freq_batch_size: int = 8,
    ):
        self.array = array
        self.sky_model = sky_model
        self.beam_model = beam_model
        self.eps = eps
        self._nufft_opts = _NufftOpts(upsampfac=nufft_upsampfac)
        self.eta_max = eta_max
        self.eta_padding = float(eta_padding)
        self.freq_batch_size = freq_batch_size

        freqs_np = np.asarray(freqs, dtype=DTYPE_R_NPY)
        self.freqs = jnp.array(freqs_np, dtype=DTYPE_R_JAX)

        self.A_beam = jnp.array(beam_model.A, dtype=DTYPE_R_JAX)
        self._beam_coeffs_full = jnp.array(beam_model.coeffs, dtype=DTYPE_R_JAX)
        self.beam_coeffs = self._beam_coeffs_full

        self.A_sky = jnp.array(sky_model.A, dtype=DTYPE_R_JAX)

        self._eq_coords_full = jnp.array(sky_model.eq_vec)
        self.eq_coords = self._eq_coords_full

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

        bls_np = np.asarray(array.bls, dtype=DTYPE_R_NPY)
        nbls = bls_np.shape[0]
        tgt_x = (freqs_np[:, None] * bls_np[None, :, 0] / C).ravel()
        tgt_y = (freqs_np[:, None] * bls_np[None, :, 1] / C).ravel()
        tgt_z = np.repeat(freqs_np, nbls)

        self._tgt_x = jnp.array(tgt_x, dtype=DTYPE_R_JAX)
        self._tgt_y = jnp.array(tgt_y, dtype=DTYPE_R_JAX)
        self._tgt_z = jnp.array(tgt_z, dtype=DTYPE_R_JAX)

        # 2D hex-rect NUFFT path.  A_lat columns are the two physical lattice
        # basis vectors (metres); satisfies A_lat @ bl_grid[i] == bls[i,:2].
        lat_mat = hex_lattice_matrix(array)  # (2,2) float64 metres
        self._lat_mat_2d = jnp.array(lat_mat.astype(DTYPE_R_NPY), dtype=DTYPE_R_JAX)
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
        self._2d_x_flat = None   # (ntime*nfreq, npix) — cached NUFFT source coords
        self._2d_y_flat = None

    # ------------------------------------------------------------------
    # Pixel masking helpers
    # ------------------------------------------------------------------

    def build_sky_mask_altitude(self, rot_matrices, min_altitude_deg=0.0):
        """
        Return a boolean array for sky pixels with maximum altitude >= min_altitude_deg.

        For each sky pixel, computes the maximum altitude (elevation angle) reached
        across all times, then returns a mask of pixels exceeding the threshold.

        Parameters
        ----------
        rot_matrices : array, shape (ntime, 3, 3)
            Rotation matrices from equatorial to topocentric coordinates.
        min_altitude_deg : float, optional
            Minimum altitude in degrees. Default 0 (horizon). Set to 0 to remove
            permanently below-horizon pixels.

        Returns
        -------
        mask : np.ndarray, shape (npix_sky_full,), dtype bool
            True for pixels with peak altitude >= min_altitude_deg.
        """
        rot_ms = np.asarray(rot_matrices, dtype=DTYPE_R_NPY)
        eq = np.asarray(self._eq_coords_full)   # (3, npix_full)

        min_altitude_rad = np.radians(min_altitude_deg)
        npix_full = eq.shape[1]
        max_alt = np.full(npix_full, -np.pi/2, dtype=DTYPE_R_NPY)  # Initialize to -90°

        for i in range(rot_ms.shape[0]):
            topo = rot_ms[i] @ eq  # (3, npix_full)
            # Altitude is arcsin(z) where z is the zenith component
            alt = np.arcsin(np.clip(topo[2], -1, 1))  # Clip to avoid NaN
            max_alt = np.maximum(max_alt, alt)

        return max_alt >= min_altitude_rad

    def apply_sky_mask(self, mask):
        """
        Apply a new pixel mask, replacing any previously-applied mask.

        After this call ``npix_sky`` equals ``mask.sum()`` and sky-coefficient
        arrays must have that many rows.  The precomputed geometry is
        invalidated; call :meth:`precompute_time_geometry` again.

        Applying a new mask always supersedes any previously-applied mask,
        allowing users to iterate on different masks without recreating
        the ForwardModel.

        Parameters
        ----------
        mask : array_like of bool, shape (npix_sky_full,)
        """
        mask = np.asarray(mask, dtype=bool)
        # Always reset to full-sky coordinates to avoid state corruption
        # when switching from an empty mask to a non-empty one.
        self.eq_coords = jnp.array(self._eq_coords_full, dtype=DTYPE_R_JAX)

        self._pixel_mask = mask
        self._pixel_indices = np.where(mask)[0].astype(np.int32)
        eq_full = np.asarray(self._eq_coords_full)   # (3, npix_full)
        self.eq_coords = jnp.array(eq_full[:, mask], dtype=DTYPE_R_JAX)
        # Invalidate geometry
        self._geom_ready = False
        for attr in ('_topo_all', '_horizon_all', '_beam_spec_horizon_all',
                     '_src_x_all', '_src_y_all', '_src_z_all', '_xi_all'):
            setattr(self, attr, None)
        return self

    def _to_active_sky_mask(self, mask):
        """Convert a full-sky or active-sky boolean mask to active-pixel space.

        Accepts either a full-sky mask of length ``npix_sky_full`` or an
        already-active mask of length ``npix_sky``. Returns a boolean array
        of length ``npix_sky`` (the active pixel count).

        Parameters
        ----------
        mask : array_like, dtype bool
            Shape ``(npix_sky_full,)`` or ``(npix_sky,)``.

        Returns
        -------
        mask_active : np.ndarray, shape (npix_sky,), dtype bool
        """
        mask = np.asarray(mask, dtype=bool)
        has_sky_mask = hasattr(self, '_pixel_indices') and self._pixel_indices is not None
        if has_sky_mask and len(mask) == self.npix_sky_full:
            return mask[self._pixel_indices]
        return mask

    # ------------------------------------------------------------------
    # Beam masking helpers
    # ------------------------------------------------------------------

    def build_beam_mask_altitude(self, rot_matrices, min_altitude_deg=0.0):
        """
        Return a boolean array for beam pixels above minimum altitude.

        The beam is fixed in the topocentric frame. Since the beam doesn't rotate,
        all beam pixels have constant altitude; this method returns pixels with
        altitude >= min_altitude_deg.

        Parameters
        ----------
        rot_matrices : array, shape (ntime, 3, 3)
            Rotation matrices (unused, kept for API compatibility).
        min_altitude_deg : float, optional
            Minimum altitude in degrees. Default 0 (horizon).
            Use 0 to remove permanently below-horizon pixels.

        Returns
        -------
        mask : np.ndarray, shape (npix_beam_full,), dtype bool
            True for beam pixels with altitude >= min_altitude_deg.
        """
        nside = self.beam_model.nside
        npix_beam_full = healpy.nside2npix(nside)

        min_altitude_rad = np.radians(min_altitude_deg)

        # Get HEALPix coordinates for all beam pixels
        theta, phi = healpy.pix2ang(nside, np.arange(npix_beam_full))

        # In topocentric frame: z points to zenith, x to East, y to North
        # HEALPix theta is colatitude (angle from North pole)
        # Altitude = pi/2 - theta
        alt = np.pi / 2 - theta

        return alt >= min_altitude_rad

    def build_sky_mask_from_beam_pixels(self, beam_pixel_mask):
        """
        Return a boolean array for sky pixels that are illuminated by selected beam pixels.

        Given a mask of beam pixels to keep, returns a mask of sky pixels whose
        HEALPix interpolation neighborhood touches at least one of the selected
        beam pixels. This is the inverse of selecting beam pixels that touch
        a given set of sky pixels.

        Parameters
        ----------
        beam_pixel_mask : array_like, shape (npix_beam_full,), dtype bool
            Boolean mask of beam pixels to consider. Sky pixels are selected if
            their interpolation stencil touches any beam pixel where mask is True.

        Returns
        -------
        mask : np.ndarray, shape (npix_sky_full,), dtype bool
            True for sky pixels whose interpolation touches selected beam pixels.

        Raises
        ------
        RuntimeError
            If geometry has not been precomputed (call precompute_time_geometry first).
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')

        beam_pixel_mask = np.asarray(beam_pixel_mask, dtype=bool)
        # Compute mask in active-sky space
        sky_mask_active = np.zeros(self.npix_sky, dtype=bool)

        for tind in range(len(self._interp_px_all)):
            px = np.asarray(self._interp_px_all[tind], dtype=np.int32)  # (4, npix_sky)
            sky_mask_active |= np.any(beam_pixel_mask[px], axis=0)

        # Expand to full-sky space if a mask is active
        if hasattr(self, '_pixel_indices') and self._pixel_indices is not None:
            sky_mask = np.zeros(self.npix_sky_full, dtype=bool)
            sky_mask[self._pixel_indices] = sky_mask_active
            return sky_mask

        # No mask applied; return in active (full-sky) space
        return sky_mask_active

    def build_beam_mask_from_sky_pixels(self, sky_pixel_mask):
        """
        Return a boolean array for beam pixels illuminated by selected sky pixels.

        Given a mask of sky pixels to keep, returns a mask of beam pixels that are
        touched by at least one of the selected sky pixels' HEALPix interpolation
        neighborhoods. This is the inverse of selecting sky pixels illuminated by
        a given set of beam pixels.

        Parameters
        ----------
        sky_pixel_mask : array_like, shape (npix_sky,), dtype bool
            Boolean mask of sky pixels to consider. Can be either full-sky or masked size:
            - If no sky mask has been applied: shape (npix_sky_full,)
            - If a sky mask has been applied: shape (npix_sky_masked,)
            The function automatically detects which case applies.
            Beam pixels are selected if they appear in the interpolation
            neighborhood of any sky pixel where mask is True.

        Returns
        -------
        mask : np.ndarray, shape (npix_beam_full,), dtype bool
            True for beam pixels that touch selected sky pixels' neighborhoods.

        Raises
        ------
        RuntimeError
            If geometry has not been precomputed (call precompute_time_geometry first).
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')

        sky_pixel_mask = np.asarray(sky_pixel_mask, dtype=bool)
        npix_beam_full = healpy.nside2npix(self.beam_model.nside)
        beam_mask = np.zeros(npix_beam_full, dtype=bool)

        mask_active = self._to_active_sky_mask(sky_pixel_mask)

        for tind in range(len(self._interp_px_all)):
            px = np.asarray(self._interp_px_all[tind], dtype=np.int32)  # (4, npix_sky)
            beam_mask[px[:, mask_active].ravel()] = True

        return beam_mask

    def apply_beam_mask(self, mask):
        """
        Permanently restrict the beam model to the pixels selected by *mask*.

        After this call the beam coefficients array has shape (npix_beam, nmodes_beam)
        where npix_beam = mask.sum(). The precomputed geometry is invalidated;
        call :meth:`precompute_time_geometry` again.

        Parameters
        ----------
        mask : array_like of bool, shape (npix_beam_full,)
        """
        mask = np.asarray(mask, dtype=bool)
        self._beam_mask = mask
        self._beam_indices = np.where(mask)[0].astype(np.int32)

        # Reduce beam coefficients (always from original unmasked version)
        beam_coeffs_full = np.asarray(self._beam_coeffs_full)
        self.beam_coeffs = jnp.array(beam_coeffs_full[mask], dtype=DTYPE_R_JAX)

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
        rot_ms = jnp.array(rot_matrices, dtype=DTYPE_R_JAX)
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

        # If a beam mask is active, expand to full size for interpolation
        if hasattr(self, '_beam_mask') and self._beam_mask is not None:
            npix_beam_full = len(self._beam_mask)
            bc_full = np.zeros((npix_beam_full, bc_np.shape[1]), dtype=bc_np.dtype)
            bc_full[self._beam_indices] = bc_np
            bc_np = bc_full

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

        self._topo_all = jnp.array(np.stack(topo_all), dtype=DTYPE_R_JAX)
        self._horizon_all = jnp.array(np.stack(horizon_all), dtype=DTYPE_R_JAX)
        self._beam_spec_horizon_all = jnp.array(np.stack(beam_spec_horizon_all), dtype=DTYPE_R_JAX)
        self._src_x_all = jnp.array(np.stack(src_x_all), dtype=DTYPE_R_JAX)
        self._src_y_all = jnp.array(np.stack(src_y_all), dtype=DTYPE_R_JAX)
        self._src_z_all = jnp.array(np.stack(src_z_all), dtype=DTYPE_R_JAX)
        self._xi_all = jnp.array(np.stack(xi_all), dtype=DTYPE_R_JAX)   # (ntime, 2, npix_sky)

        # 2D path: precompute flattened NUFFT source coords for all (time, freq) pairs.
        # x_flat[t*nfreq + f, j] = 2π * freqs[f] / C * xi[t, 0, j]
        # Shape: (ntime*nfreq, npix_sky).  Cached here so simulate_2d avoids
        # recomputing the freq-scale per call.
        _two_pi_over_c = 2.0 * np.pi / C
        _xi_j = self._xi_all                          # (ntime, 2, npix)
        _freqs = self.freqs                            # (nfreq,)
        self._2d_x_flat = (
            (_two_pi_over_c * _xi_j[:, 0:1, :] * _freqs[None, :, None])
            .reshape(ntime * self.nfreq, self.npix_sky)
            .astype(DTYPE_R_JAX)
        )
        self._2d_y_flat = (
            (_two_pi_over_c * _xi_j[:, 1:2, :] * _freqs[None, :, None])
            .reshape(ntime * self.nfreq, self.npix_sky)
            .astype(DTYPE_R_JAX)
        )

        # Bilinear stencil for beam solving — list of (4, npix_sky) numpy arrays,
        # one per time step.  Kept as a list so that beam solving can be done with
        # a per-time Python loop without stacking into a single large array.
        self._interp_px_all  = interp_px_all   # list[np.ndarray(4, npix_sky), int32]
        self._interp_wgt_all = interp_wgt_all  # list[np.ndarray(4, npix_sky), DTYPE_R]

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
            W_eta.ravel().astype(DTYPE_C_JAX),
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
            opts=self._nufft_opts,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _simulate_one_precomputed(self, sky_coeffs, tind):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_precomputed_sky_spec(sky_spec, tind)

    def _simulate_impl(self, sky_coeffs, rot_matrices):
        """Core simulate implementation (used internally; consider using simulate() instead)."""
        if (not self._geom_ready) or (self._topo_all.shape[0] != rot_matrices.shape[0]):
            self.precompute_time_geometry(rot_matrices)
        sky_spec = sky_coeffs @ self.A_sky.T
        inds = jnp.arange(rot_matrices.shape[0], dtype=jnp.int32)
        return jax.vmap(lambda tind: self._simulate_one_precomputed_sky_spec(sky_spec, tind))(inds)

    def simulate_3d(self, sky_coeffs, rot_matrices, t_chunk_size=12):
        """Simulate visibilities using 3D type-3 NUFFT with automatic time chunking.

        Processes time steps in chunks to keep precomputed geometry arrays small.
        For a single chunk, falls back to direct simulation; for multiple chunks,
        accumulates results from each chunk.

        Parameters
        ----------
        sky_coeffs : jnp.ndarray, shape (npix_sky, nmodes_sky)
        rot_matrices : jnp.ndarray, shape (ntime, 3, 3)
        t_chunk_size : int, optional
            Number of time steps per chunk. Default 12 (~6–12× memory reduction
            for 96-time observations). Set to ntime to disable chunking.

        Returns
        -------
        vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex64
        """
        ntime = rot_matrices.shape[0]
        if ntime <= t_chunk_size:
            # Single chunk: use direct implementation
            return self._simulate_impl(sky_coeffs, rot_matrices)

        # Multi-chunk: accumulate results
        chunks = []
        for t_start in range(0, ntime, t_chunk_size):
            t_end = min(t_start + t_chunk_size, ntime)
            rot_chunk = rot_matrices[t_start:t_end]
            # Invalidate geometry and recompute for this chunk
            self._geom_ready = False
            for attr in ('_topo_all', '_horizon_all', '_beam_spec_horizon_all',
                         '_src_x_all', '_src_y_all', '_src_z_all', '_xi_all',
                         '_interp_px_all', '_interp_wgt_all'):
                setattr(self, attr, None)
            vis_chunk = self._simulate_impl(sky_coeffs, rot_chunk)
            chunks.append(vis_chunk)

        return jnp.concatenate(chunks, axis=0)

    def adjoint_residual_one_time(self, residual_fb, tind):
        """Backproject one-time residuals to a pixel-delay dirty map."""
        c = jnp.conj(residual_fb.reshape(-1).astype(DTYPE_C_JAX))
        dirty_flat = nufft3(
            c,
            self._tgt_x, self._tgt_y, self._tgt_z,
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            iflag=1,
            eps=self.eps,
            opts=self._nufft_opts,
        )
        dirty = jnp.conj(dirty_flat.reshape(self.npix_sky, self.ndelay_eff))
        return dirty / (self.nfreq * self.nbls)

    def dirty_apparent_sky_one_time(self, residual_fb, tind):
        """Backproject one-time residuals to dirty apparent sky in frequency space."""
        dirty_eta = self.adjoint_residual_one_time(residual_fb, tind)
        tmp = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C_JAX)
        tmp = tmp.at[:, self._eta_idx].set(dirty_eta / self._phase[None, :])
        dirty_pf = jnp.fft.ifft(tmp, axis=1)
        return dirty_pf

    def _sky_update_from_dirty(self, dirty_pf, tind, step_size=1.0, beam_reg=1e-3):
        """Form sky correction from pre-computed dirty apparent sky map.

        Parameters
        ----------
        dirty_pf : jnp.ndarray, shape (npix_sky, nfreq), complex
            Frequency-domain dirty apparent sky map
        tind : int
            Time index
        step_size : float
        beam_reg : float
            Regularisation for division by beam weight

        Returns
        -------
        delta_app : jnp.ndarray, shape (npix_sky, nfreq), complex
            Sky correction term
        weight : jnp.ndarray, shape (npix_sky, nfreq), real
            Beam power weight (|beam_spec_h|^2)
        """
        beam_spec_h = self._beam_spec_horizon_all[tind]   # (npix_sky, nfreq), already * horizon
        weight = jnp.abs(beam_spec_h) ** 2
        delta_app = step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)
        return delta_app, weight

    def apparent_sky_update_one_time(
        self,
        residual_fb,
        tind,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Form a beam-weighted apparent-sky correction for one time."""
        dirty_pf = self.dirty_apparent_sky_one_time(residual_fb, tind)
        return self._sky_update_from_dirty(dirty_pf, tind, step_size, beam_reg)

    def _accumulate_equatorial_sky_update_impl(self, residual_vis, sky_update_fn, step_size, beam_reg):
        inds = jnp.arange(int(residual_vis.shape[0]), dtype=jnp.int32)

        def one_time(resid_t, tind):
            return sky_update_fn(resid_t, tind, step_size=step_size, beam_reg=beam_reg)

        delta_all, weight_all = jax.vmap(one_time)(residual_vis, inds)
        num = jnp.sum(weight_all.astype(DTYPE_C_JAX) * delta_all, axis=0)
        den = jnp.sum(weight_all.real.astype(DTYPE_R_JAX), axis=0)
        delta_eq_pf = num / (den.astype(DTYPE_C_JAX) + self.eps)
        delta_eq_pf = delta_eq_pf / residual_vis.shape[0]
        return (delta_eq_pf.real @ self.A_sky).astype(DTYPE_R_JAX)

    def accumulate_equatorial_sky_update_3d(
        self,
        residual_vis,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Accumulate a global equatorial sky update from all times (3D path)."""
        return self._accumulate_equatorial_sky_update_impl(
            residual_vis, self.apparent_sky_update_one_time, step_size, beam_reg
        )

    def _form_beam_update_one_time(self, sky_spec, sky_weight, dirty_pf, tind, step_size, sky_reg):
        """Form per-pixel beam update delta from pre-computed dirty apparent sky.

        Parameters
        ----------
        sky_spec : np.ndarray, shape (npix_sky, nfreq), complex
            Sky spectrum (sky_coeffs @ A_sky.T)
        sky_weight : np.ndarray, shape (npix_sky, nfreq), real
            Sky power (|sky_spec|^2)
        dirty_pf : np.ndarray, shape (npix_sky, nfreq), complex
            Frequency-domain dirty apparent sky
        tind : int
            Time index
        step_size : float
        sky_reg : float
            Regularisation for division by sky weight

        Returns
        -------
        delta_bi : np.ndarray, shape (npix_sky, nmodes_beam), real
            Per-pixel beam basis coefficient update
        """
        ab_np = np.asarray(self.A_beam, dtype=DTYPE_R_NPY)
        horizon = np.asarray(self._horizon_all[tind])
        delta_bsf = step_size * np.conj(sky_spec) * dirty_pf / (sky_weight + sky_reg)
        delta_bi = (np.real(delta_bsf) * horizon[:, None]) @ ab_np
        return delta_bi

    def _accumulate_beam_update_impl(self, sky_coeffs, residual_vis, dirty_fn, step_size, sky_reg):
        ab_np = np.asarray(self.A_beam, dtype=DTYPE_R_NPY)

        sky_spec = np.asarray((jnp.array(sky_coeffs) @ self.A_sky.T).astype(DTYPE_C_JAX))
        sky_weight = (sky_spec.real ** 2 + sky_spec.imag ** 2).astype(DTYPE_R_NPY)
        sky_pow = sky_weight.sum(axis=1)

        npix_beam_accum = self.npix_beam
        has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
        if has_beam_mask:
            npix_beam_accum = int(self._beam_mask.sum())
            npix_beam_full = len(self._beam_mask)

        num = np.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_NPY)
        den = np.zeros(npix_beam_accum, dtype=DTYPE_R_NPY)

        ntime = int(residual_vis.shape[0])
        for tind in range(ntime):
            dirty_pf = np.asarray(dirty_fn(residual_vis[tind], tind))
            delta_bi = self._form_beam_update_one_time(sky_spec, sky_weight, dirty_pf, tind, step_size, sky_reg)

            px  = self._interp_px_all[tind]
            wgt = self._interp_wgt_all[tind]

            px_flat = px.reshape(-1)
            w_flat = (wgt * sky_pow[None, :]).reshape(-1)
            contrib = (
                w_flat[:, None]
                * np.repeat(delta_bi[None, :, :], 4, axis=0).reshape(-1, self.nmodes_beam)
            )

            if has_beam_mask:
                # Only accumulate to masked beam pixels
                mask_flat = self._beam_mask[px_flat]
                px_masked = np.searchsorted(self._beam_indices, px_flat[mask_flat])
                np.add.at(num, px_masked, contrib[mask_flat])
                np.add.at(den, px_masked, w_flat[mask_flat])
            else:
                np.add.at(num, px_flat, contrib)
                np.add.at(den, px_flat, w_flat)

        delta_bc_masked = num / (den[:, None] + self.eps)
        delta_bc_masked /= ntime

        # Expand back to full beam if mask was applied
        if has_beam_mask:
            delta_bc_full = np.zeros((npix_beam_full, self.nmodes_beam), dtype=DTYPE_R_NPY)
            delta_bc_full[self._beam_indices] = np.asarray(delta_bc_masked)
            delta_bc = delta_bc_full
        else:
            delta_bc = delta_bc_masked

        return jnp.array(delta_bc, dtype=DTYPE_R_JAX)

    def accumulate_beam_update_3d(
        self,
        sky_coeffs,
        residual_vis,
        step_size: float = 1.0,
        sky_reg: float = 1e-3,
    ):
        """Accumulate a global beam-coefficient update from all times (3D path).

        Dual of :meth:`accumulate_equatorial_sky_update_3d`.  With sky and gains
        held fixed, this backprojects the gain-calibrated residuals through the
        sky model to estimate the beam-coefficient correction that best reduces
        the residual.
        """
        return self._accumulate_beam_update_impl(
            sky_coeffs, residual_vis, self.dirty_apparent_sky_one_time, step_size, sky_reg
        )

    def accumulate_sky_and_beam_update_3d(
        self,
        sky_coeffs,
        residual_vis,
        sky_step_size: float = 1.0,
        beam_step_size: float = 1.0,
        beam_reg: float = 1e-3,
        sky_reg: float = 1e-3,
    ):
        """Accumulate sky and beam updates from shared adjoint (3D path).

        Computes the dirty apparent sky map once per time step and derives both
        sky and beam corrections from it. This is more efficient than calling
        :meth:`accumulate_equatorial_sky_update_3d` and :meth:`accumulate_beam_update_3d`
        separately. Optimized to avoid host/device churn by keeping accumulation
        on JAX arrays and using JAX scatter operations.

        Parameters
        ----------
        sky_coeffs : jnp.ndarray, shape (npix_sky, nmodes_sky)
        residual_vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex
        sky_step_size : float
        beam_step_size : float
        beam_reg : float
            Beam regularisation parameter
        sky_reg : float
            Sky regularisation parameter

        Returns
        -------
        sky_update : jnp.ndarray, shape (npix_sky, nmodes_sky)
        beam_update : jnp.ndarray, shape (npix_beam, nmodes_beam)
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')

        # Precompute static quantities (keep in JAX to avoid host churn)
        sky_spec_jax = jnp.array(sky_coeffs) @ self.A_sky.T  # (npix_sky, nfreq)
        sky_weight = jnp.abs(sky_spec_jax) ** 2              # (npix_sky, nfreq)
        sky_pow = sky_weight.sum(axis=1)                     # (npix_sky,)

        ntime = int(residual_vis.shape[0])

        # Process one time step
        def one_time_step(carry_sky, carry_beam, tind):
            # Get dirty apparent sky via 3D adjoint (includes IFFT)
            dirty_pf = self.dirty_apparent_sky_one_time(residual_vis[tind], int(tind))

            # Sky update contribution
            beam_spec_h = self._beam_spec_horizon_all[int(tind)]
            weight = jnp.abs(beam_spec_h) ** 2
            delta_app = sky_step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)

            new_carry_sky = (
                carry_sky[0] + weight.astype(DTYPE_C_JAX) * delta_app,
                carry_sky[1] + weight.astype(DTYPE_R_JAX),
            )

            # Beam update contribution
            horizon = self._horizon_all[int(tind)]
            delta_bsf = beam_step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
            delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

            # Interpolate beam contributions to beam pixel grid
            px = self._interp_px_all[int(tind)]       # (4, npix_sky)
            wgt = self._interp_wgt_all[int(tind)]     # (4, npix_sky)
            px_flat = px.reshape(-1)                  # (4*npix_sky,)
            w_flat = (wgt * sky_pow[None, :]).reshape(-1)  # (4*npix_sky,)

            # Build contributions: broadcast delta_bi to all 4 neighbors of each sky pixel
            contrib = (w_flat[:, None] * jnp.repeat(delta_bi[None, :, :], 4, axis=0).reshape(-1, self.nmodes_beam))

            # Handle beam mask if present
            has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
            if has_beam_mask:
                mask_flat = self._beam_mask[px_flat]
                px_masked = jnp.searchsorted(self._beam_indices, px_flat[mask_flat])
                # Scatter into masked accumulation arrays
                beam_num_update = jnp.zeros((int(self._beam_mask.sum()), self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(int(self._beam_mask.sum()), dtype=DTYPE_R_JAX)
                beam_num_update = beam_num_update.at[px_masked].add(contrib[mask_flat])
                beam_den_update = beam_den_update.at[px_masked].add(w_flat[mask_flat])
            else:
                beam_num_update = jnp.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(self.npix_beam, dtype=DTYPE_R_JAX)
                beam_num_update = beam_num_update.at[px_flat].add(contrib)
                beam_den_update = beam_den_update.at[px_flat].add(w_flat)

            new_carry_beam = (
                carry_beam[0] + beam_num_update,
                carry_beam[1] + beam_den_update,
            )

            return new_carry_sky, new_carry_beam

        # Initialize accumulators
        sky_num_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C_JAX)
        sky_den_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_R_JAX)

        has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
        npix_beam_accum = int(self._beam_mask.sum()) if has_beam_mask else self.npix_beam
        beam_num_init = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
        beam_den_init = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

        # Accumulate over time (Python loop necessary due to time-indexed array access)
        sky_num, sky_den = sky_num_init, sky_den_init
        beam_num, beam_den = beam_num_init, beam_den_init

        for tind in range(ntime):
            (sky_num, sky_den), (beam_num, beam_den) = one_time_step(
                (sky_num, sky_den), (beam_num, beam_den), tind
            )

        # Finalize sky update
        delta_eq_pf = sky_num / (sky_den.astype(DTYPE_C_JAX) + self.eps)
        delta_eq_pf = delta_eq_pf / ntime
        sky_update = (jnp.real(delta_eq_pf) @ jnp.array(self.A_sky, dtype=DTYPE_R_JAX)).astype(DTYPE_R_JAX)

        # Finalize beam update
        beam_delta_masked = beam_num / (beam_den[:, None] + self.eps)
        beam_delta_masked = beam_delta_masked / ntime

        if has_beam_mask:
            npix_beam_full = len(self._beam_mask)
            beam_delta_full = jnp.zeros((npix_beam_full, self.nmodes_beam), dtype=DTYPE_R_JAX)
            beam_indices_jax = jnp.array(self._beam_indices)
            beam_delta_full = beam_delta_full.at[beam_indices_jax].set(beam_delta_masked)
            beam_update = beam_delta_full
        else:
            beam_update = beam_delta_masked

        return sky_update, beam_update

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
        return jnp.array(np.stack(beam_spec_horizon_all), dtype=DTYPE_R_JAX)

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
            np.stack(beam_spec_horizon_all), dtype=DTYPE_R_JAX
        )
        # Update both the full version and the masked version (if mask is active)
        self._beam_coeffs_full = jnp.array(bc_np, dtype=DTYPE_R_JAX)
        if hasattr(self, '_beam_mask') and self._beam_mask is not None:
            # Re-apply mask to the updated full coefficients
            self.beam_coeffs = jnp.array(bc_np[self._beam_indices], dtype=DTYPE_R_JAX)
        else:
            self.beam_coeffs = jnp.array(bc_np, dtype=DTYPE_R_JAX)
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
            W_eta.ravel().astype(DTYPE_C_JAX),
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
            opts=self._nufft_opts,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _simulate_one_variable_beam(self, sky_coeffs, beam_coeffs, tind):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_variable_beam_sky_spec(sky_spec, beam_coeffs, tind)

    def simulate_variable_beam_3d(self, sky_coeffs, beam_coeffs, rot_matrices):
        """Simulate all time steps with *beam_coeffs* as a differentiable input (3D path)."""
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
        """Apply per-frequency type-1 2D NUFFTs to a (npix_sky, nfreq) weight array.

        Uses frequency batching to reduce vmap overhead: groups frequencies into batches
        before vmapping, reducing JAX trace count from nfreq to ceil(nfreq/freq_batch_size).
        """
        n_q = self._2d_n_q
        n_r = self._2d_n_r
        q_idx = self._2d_q_idx
        r_idx = self._2d_r_idx
        freq_batch_size = getattr(self, 'freq_batch_size', 8)  # Default batch size

        def one_freq(args):
            W_fi, nu = args
            x_j = (2.0 * jnp.pi * nu / C * xi[0]).astype(DTYPE_R_JAX)
            y_j = (2.0 * jnp.pi * nu / C * xi[1]).astype(DTYPE_R_JAX)
            grid = nufft1((n_q, n_r), W_fi, x_j, y_j, iflag=1, eps=self.eps, opts=self._nufft_opts)
            return grid[q_idx, r_idx]

        nfreq = len(freqs)
        n_batch = (nfreq + freq_batch_size - 1) // freq_batch_size

        # If no batching needed (small nfreq), use simple vmap
        if n_batch == 1:
            return jax.vmap(one_freq)((W.T, freqs))

        # Batch frequencies and vmap over batches
        vis_batches = []
        for b in range(n_batch):
            start = b * freq_batch_size
            end = min(start + freq_batch_size, nfreq)
            W_batch = W.T[start:end]           # (batch_size, npix_sky)
            freqs_batch = freqs[start:end]     # (batch_size,)
            vis_batch = jax.vmap(one_freq)((W_batch, freqs_batch))  # (batch_size, nbls)
            vis_batches.append(vis_batch)

        return jnp.concatenate(vis_batches, axis=0)  # (nfreq, nbls)

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
        W = (sky_spec * beam_spec_h).astype(DTYPE_C_JAX)
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
        W = (sky_spec * beam_spec_h).astype(DTYPE_C_JAX)
        xi = self._xi_all[tind]
        return self._nufft2d_W_to_vis(W, xi, self.freqs)

    def _simulate_2d_impl(self, sky_coeffs, rot_matrices):
        """Core 2D simulate implementation (used internally; consider using simulate_2d() instead)."""
        if (not self._geom_ready) or (self._2d_x_flat is None) or (self._topo_all.shape[0] != rot_matrices.shape[0]):
            self.precompute_time_geometry(rot_matrices)
        ntime = self._topo_all.shape[0]
        sky_spec = (sky_coeffs @ self.A_sky.T).astype(DTYPE_C_JAX)   # (npix, nfreq)
        # sky × beam for all times: (ntime, npix, nfreq) → (ntime*nfreq, npix)
        W_flat = (
            (sky_spec[None, :, :] * self._beam_spec_horizon_all)
            .transpose(0, 2, 1)
            .reshape(ntime * self.nfreq, self.npix_sky)
        )
        n_q, n_r = self._2d_n_q, self._2d_n_r
        q_idx, r_idx = self._2d_q_idx, self._2d_r_idx

        def _one(args):
            return nufft1(
                (n_q, n_r), args[0], args[1], args[2],
                iflag=1, eps=self.eps, opts=self._nufft_opts,
            )[q_idx, r_idx]

        vis_flat = jax.vmap(_one)((W_flat, self._2d_x_flat, self._2d_y_flat))
        return vis_flat.reshape(ntime, self.nfreq, self.nbls)

    def simulate_2d(self, sky_coeffs, rot_matrices, t_chunk_size=12):
        """Simulate visibilities using 2D hex-rect NUFFT with automatic time chunking.

        Each frequency is handled by a type-1 2D NUFFT that maps sky × beam
        weighted pixels to a compact uniform uv grid; baselines are read from
        that grid at their integer axial positions.

        Processes time steps in chunks to keep precomputed geometry arrays small.
        For a single chunk, falls back to direct simulation; for multiple chunks,
        accumulates results from each chunk.

        The channel spacing of *freqs* should satisfy
        :func:`~newnucal.hexrect.critical_channel_spacing` to avoid aliasing.

        Parameters
        ----------
        sky_coeffs : jnp.ndarray, shape (npix_sky, nmodes_sky)
        rot_matrices : jnp.ndarray, shape (ntime, 3, 3)
        t_chunk_size : int, optional
            Number of time steps per chunk. Default 12 (~6–12× memory reduction
            for 96-time observations). Set to ntime to disable chunking.

        Returns
        -------
        vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex64
        """
        ntime = rot_matrices.shape[0]
        if ntime <= t_chunk_size:
            # Single chunk: use direct implementation
            return self._simulate_2d_impl(sky_coeffs, rot_matrices)

        # Multi-chunk: accumulate results
        chunks = []
        for t_start in range(0, ntime, t_chunk_size):
            t_end = min(t_start + t_chunk_size, ntime)
            rot_chunk = rot_matrices[t_start:t_end]
            # Invalidate geometry and recompute for this chunk
            self._geom_ready = False
            for attr in ('_topo_all', '_horizon_all', '_beam_spec_horizon_all',
                         '_src_x_all', '_src_y_all', '_src_z_all', '_xi_all',
                         '_interp_px_all', '_interp_wgt_all', '_2d_x_flat', '_2d_y_flat'):
                setattr(self, attr, None)
            vis_chunk = self._simulate_2d_impl(sky_coeffs, rot_chunk)
            chunks.append(vis_chunk)

        return jnp.concatenate(chunks, axis=0)

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
        resid = residual_fb.astype(DTYPE_C_JAX)   # (nfreq, nbls)

        def one_freq(args):
            resid_f, nu = args                                          # (nbls,), scalar
            # Scatter residuals to (n_q, n_r) grid at shifted bl positions
            grid = jnp.zeros((n_q, n_r), dtype=DTYPE_C_JAX).at[q_idx, r_idx].set(resid_f)
            x_j = (2.0 * jnp.pi * nu / C * xi[0]).astype(DTYPE_R_JAX)     # (npix_sky,)
            y_j = (2.0 * jnp.pi * nu / C * xi[1]).astype(DTYPE_R_JAX)
            # nufft2 with iflag=-1 is the adjoint of nufft1 with iflag=+1
            return nufft2(grid, x_j, y_j, iflag=-1, eps=self.eps, opts=self._nufft_opts)  # (npix_sky,)

        dirty_freq_first = jax.vmap(one_freq)((resid, freqs))  # (nfreq, npix_sky)
        # Divide by nbls (per-frequency PSF amplitude) to match the 3D path scale.
        # The 3D path's extra 1/nfreq from ifft is cancelled by 1/phase ~ nfreq,
        # so the 3D dirty sky has amplitude ~|resid| — divide 2D by nbls to match.
        return (dirty_freq_first.T / self.nbls)     # (npix_sky, nfreq)

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
        return self._sky_update_from_dirty(dirty_pf, tind, step_size, beam_reg)

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

    def accumulate_sky_and_beam_update_2d(
        self,
        sky_coeffs,
        residual_vis,
        sky_step_size: float = 1.0,
        beam_step_size: float = 1.0,
        beam_reg: float = 1e-3,
        sky_reg: float = 1e-3,
    ):
        """Accumulate sky and beam updates from shared adjoint (2D path).

        Computes the dirty apparent sky map once per time step and derives both
        sky and beam corrections from it. Optimized to avoid host/device churn
        by keeping accumulation on JAX arrays and using JAX scatter operations.

        Parameters
        ----------
        sky_coeffs : jnp.ndarray, shape (npix_sky, nmodes_sky)
        residual_vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex
        sky_step_size : float
        beam_step_size : float
        beam_reg : float
            Beam regularisation parameter
        sky_reg : float
            Sky regularisation parameter

        Returns
        -------
        sky_update : jnp.ndarray, shape (npix_sky, nmodes_sky)
        beam_update : jnp.ndarray, shape (npix_beam, nmodes_beam)
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')

        # Precompute static quantities (keep in JAX to avoid host churn)
        sky_spec_jax = jnp.array(sky_coeffs) @ self.A_sky.T  # (npix_sky, nfreq)
        sky_weight = jnp.abs(sky_spec_jax) ** 2              # (npix_sky, nfreq)
        sky_pow = sky_weight.sum(axis=1)                     # (npix_sky,)

        ntime = int(residual_vis.shape[0])

        # Process one time step
        def one_time_step(carry_sky, carry_beam, tind):
            # Get dirty apparent sky via 2D adjoint
            dirty_pf = self.dirty_apparent_sky_one_time_2d(residual_vis[tind], int(tind))

            # Sky update contribution
            beam_spec_h = self._beam_spec_horizon_all[int(tind)]
            weight = jnp.abs(beam_spec_h) ** 2
            delta_app = sky_step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)

            new_carry_sky = (
                carry_sky[0] + weight.astype(DTYPE_C_JAX) * delta_app,
                carry_sky[1] + weight.astype(DTYPE_R_JAX),
            )

            # Beam update contribution: interpolate and scatter
            horizon = self._horizon_all[int(tind)]
            delta_bsf = beam_step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
            delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

            # Interpolate beam contributions to beam pixel grid
            px = self._interp_px_all[int(tind)]       # (4, npix_sky)
            wgt = self._interp_wgt_all[int(tind)]     # (4, npix_sky)
            px_flat = px.reshape(-1)                  # (4*npix_sky,)
            w_flat = (wgt * sky_pow[None, :]).reshape(-1)  # (4*npix_sky,)

            # Build contributions: broadcast delta_bi to all 4 neighbors of each sky pixel
            contrib = (w_flat[:, None] * jnp.repeat(delta_bi[None, :, :], 4, axis=0).reshape(-1, self.nmodes_beam))

            # Handle beam mask if present
            has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
            if has_beam_mask:
                mask_flat = self._beam_mask[px_flat]
                px_masked = jnp.searchsorted(self._beam_indices, px_flat[mask_flat])
                # Scatter into masked accumulation arrays
                beam_num_update = jnp.zeros((int(self._beam_mask.sum()), self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(int(self._beam_mask.sum()), dtype=DTYPE_R_JAX)
                beam_num_update = beam_num_update.at[px_masked].add(contrib[mask_flat])
                beam_den_update = beam_den_update.at[px_masked].add(w_flat[mask_flat])
            else:
                beam_num_update = jnp.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(self.npix_beam, dtype=DTYPE_R_JAX)
                beam_num_update = beam_num_update.at[px_flat].add(contrib)
                beam_den_update = beam_den_update.at[px_flat].add(w_flat)

            new_carry_beam = (
                carry_beam[0] + beam_num_update,
                carry_beam[1] + beam_den_update,
            )

            return new_carry_sky, new_carry_beam

        # Initialize accumulators
        sky_num_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C_JAX)
        sky_den_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_R_JAX)

        has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
        npix_beam_accum = int(self._beam_mask.sum()) if has_beam_mask else self.npix_beam
        beam_num_init = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
        beam_den_init = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

        # Accumulate over time (Python loop necessary due to time-indexed array access)
        sky_num, sky_den = sky_num_init, sky_den_init
        beam_num, beam_den = beam_num_init, beam_den_init

        for tind in range(ntime):
            (sky_num, sky_den), (beam_num, beam_den) = one_time_step(
                (sky_num, sky_den), (beam_num, beam_den), tind
            )

        # Finalize sky update
        delta_eq_pf = sky_num / (sky_den.astype(DTYPE_C_JAX) + self.eps)
        delta_eq_pf = delta_eq_pf / ntime
        sky_update = (jnp.real(delta_eq_pf) @ jnp.array(self.A_sky, dtype=DTYPE_R_JAX)).astype(DTYPE_R_JAX)

        # Finalize beam update
        beam_delta_masked = beam_num / (beam_den[:, None] + self.eps)
        beam_delta_masked = beam_delta_masked / ntime

        if has_beam_mask:
            npix_beam_full = len(self._beam_mask)
            beam_delta_full = jnp.zeros((npix_beam_full, self.nmodes_beam), dtype=DTYPE_R_JAX)
            beam_indices_jax = jnp.array(self._beam_indices)
            beam_delta_full = beam_delta_full.at[beam_indices_jax].set(beam_delta_masked)
            beam_update = beam_delta_full
        else:
            beam_update = beam_delta_masked

        return sky_update, beam_update

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npix_sky(self) -> int:
        # eq_coords has shape (3, npix_active); reflects any applied pixel mask
        return int(self.eq_coords.shape[1])

    @property
    def npix_sky_full(self) -> int:
        # Full-sky pixel count, unaffected by any applied pixel mask
        return int(self._eq_coords_full.shape[1])

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
