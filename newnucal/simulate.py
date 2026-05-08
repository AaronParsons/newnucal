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
from dataclasses import dataclass
from typing import Optional

from .array import HERAArray
from .beam import BeamModel
from .sky import SkyModel
from .hexrect import hex_lattice_matrix, axial_grid_size
from .utils import DTYPE_R_JAX, DTYPE_C_JAX, DTYPE_R_NPY, C


@dataclass
class TimeGeometry:
    """Local time-dependent geometry state for a simulation.

    Holds precomputed coordinate transformations and interpolation weights
    for a specific time range. Using separate objects avoids mutating global
    ForwardModel state during chunked simulation.
    """
    geom_ready: bool = False
    topo_all: Optional[jnp.ndarray] = None
    horizon_all: Optional[jnp.ndarray] = None
    beam_spec_horizon_all: Optional[jnp.ndarray] = None
    src_x_all: Optional[jnp.ndarray] = None
    src_y_all: Optional[jnp.ndarray] = None
    src_z_all: Optional[jnp.ndarray] = None
    xi_all: Optional[jnp.ndarray] = None
    interp_px_all: Optional[jnp.ndarray] = None
    interp_wgt_all: Optional[jnp.ndarray] = None
    _2d_x_flat: Optional[jnp.ndarray] = None
    _2d_y_flat: Optional[jnp.ndarray] = None


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
        method: str = "2d",
    ):
        self.array = array
        self.sky_model = sky_model
        self.beam_model = beam_model
        self.eps = eps
        self._nufft_opts = _NufftOpts(upsampfac=nufft_upsampfac)
        self.eta_max = eta_max
        self.eta_padding = float(eta_padding)
        self.freq_batch_size = freq_batch_size
        if method not in ("2d", "3d"):
            raise ValueError(f"method must be '2d' or '3d', got {method!r}")
        self.method = method

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

    def simulate(self, sky_coeffs, rot_matrices, t_chunk_size=12):
        if self.method == "2d":
            return self.simulate_2d(sky_coeffs, rot_matrices, t_chunk_size)
        else:
            return self.simulate_3d(sky_coeffs, rot_matrices, t_chunk_size)

    def simulate_variable_beam(self, sky_coeffs, beam_coeffs, rot_matrices):
        if self.method == "2d":
            return self.simulate_variable_beam_2d(sky_coeffs, beam_coeffs, rot_matrices)
        else:
            return self.simulate_variable_beam_3d(sky_coeffs, beam_coeffs, rot_matrices)

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

    def _build_time_geometry(self, rot_matrices) -> TimeGeometry:
        """Build a local TimeGeometry object without mutating self state.

        Parameters
        ----------
        rot_matrices : array_like, shape (ntime, 3, 3)

        Returns
        -------
        TimeGeometry
            Local geometry object with all time-dependent arrays.
        """
        # Use JAX arrays throughout for JIT compatibility with traced inputs
        rot_ms = jnp.array(rot_matrices, dtype=DTYPE_R_JAX)
        ntime = int(rot_ms.shape[0])
        npix = self.npix_sky
        ndelay = self.ndelay_eff
        two_pi = 2.0 * np.pi

        # Convert constants to JAX for consistency
        bc_jax = jnp.array(self.beam_coeffs, dtype=DTYPE_R_JAX)
        if hasattr(self, '_beam_mask') and self._beam_mask is not None:
            npix_beam_full = len(self._beam_mask)
            bc_full = jnp.zeros((npix_beam_full, bc_jax.shape[1]), dtype=DTYPE_R_JAX)
            bc_full = bc_full.at[self._beam_indices].set(bc_jax)
            bc_jax = bc_full

        ab_jax = jnp.array(self.A_beam, dtype=DTYPE_R_JAX)
        eta_jax = jnp.array(self._eta, dtype=DTYPE_R_JAX)
        lat_mat_jax = jnp.array(self._lat_mat_2d, dtype=DTYPE_R_JAX)
        eq_coords_jax = jnp.array(self.eq_coords, dtype=DTYPE_R_JAX)

        # Compute all geometry per time using JAX (works in JIT with traced inputs)
        topo_all = []
        horizon_all = []
        beam_spec_horizon_all = []
        src_x_all = []
        src_y_all = []
        src_z_all = []
        xi_all = []
        interp_px_all = []
        interp_wgt_all = []

        for i in range(ntime):
            topo = (rot_ms[i] @ eq_coords_jax).astype(DTYPE_R_JAX)
            horizon = (topo[2] > 0).astype(DTYPE_R_JAX)
            topo_th, topo_ph = healjax.vec2ang(topo[0], topo[1], topo[2])
            px, wgts = get_interp_weights(topo_th, topo_ph, self.beam_model.nside)
            px_jax = jnp.array(px, dtype=jnp.int32)
            wgts_jax = jnp.array(wgts, dtype=DTYPE_R_JAX)

            interp_px_all.append(px_jax)
            interp_wgt_all.append(wgts_jax)

            bi_jax = jnp.sum(wgts_jax[:, :, None] * bc_jax[px_jax], axis=0)
            bs_jax = (bi_jax @ ab_jax.T) * horizon[:, None]
            beam_spec_horizon_all.append(bs_jax)

            src_x = jnp.repeat(topo[0], ndelay) * two_pi
            src_y = jnp.repeat(topo[1], ndelay) * two_pi
            src_z = jnp.tile(eta_jax, npix) * two_pi

            xi = lat_mat_jax.T @ topo[:2, :]

            topo_all.append(topo)
            horizon_all.append(horizon)
            src_x_all.append(src_x)
            src_y_all.append(src_y)
            src_z_all.append(src_z)
            xi_all.append(xi)

        topo_all_jax = jnp.stack(topo_all)
        horizon_all_jax = jnp.stack(horizon_all)
        beam_spec_horizon_all_jax = jnp.stack(beam_spec_horizon_all)
        src_x_all_jax = jnp.stack(src_x_all)
        src_y_all_jax = jnp.stack(src_y_all)
        src_z_all_jax = jnp.stack(src_z_all)
        xi_all_jax = jnp.stack(xi_all)

        # 2D path: precompute flattened NUFFT source coords
        _two_pi_over_c = 2.0 * np.pi / C
        _freqs = self.freqs
        _2d_x_flat = (
            (_two_pi_over_c * xi_all_jax[:, 0:1, :] * _freqs[None, :, None])
            .reshape(ntime * self.nfreq, self.npix_sky)
            .astype(DTYPE_R_JAX)
        )
        _2d_y_flat = (
            (_two_pi_over_c * xi_all_jax[:, 1:2, :] * _freqs[None, :, None])
            .reshape(ntime * self.nfreq, self.npix_sky)
            .astype(DTYPE_R_JAX)
        )

        interp_px_all_jax = jnp.stack(interp_px_all)
        interp_wgt_all_jax = jnp.stack(interp_wgt_all)

        return TimeGeometry(
            geom_ready=True,
            topo_all=topo_all_jax,
            horizon_all=horizon_all_jax,
            beam_spec_horizon_all=beam_spec_horizon_all_jax,
            src_x_all=src_x_all_jax,
            src_y_all=src_y_all_jax,
            src_z_all=src_z_all_jax,
            xi_all=xi_all_jax,
            interp_px_all=interp_px_all_jax,
            interp_wgt_all=interp_wgt_all_jax,
            _2d_x_flat=_2d_x_flat,
            _2d_y_flat=_2d_y_flat,
        )

    def precompute_time_geometry(self, rot_matrices):
        """Precompute time-only geometry and interpolation operators.

        Parameters
        ----------
        rot_matrices : array_like, shape (ntime, 3, 3)
        """
        geom = self._build_time_geometry(rot_matrices)
        self._topo_all = geom.topo_all
        self._horizon_all = geom.horizon_all
        self._beam_spec_horizon_all = geom.beam_spec_horizon_all
        self._src_x_all = geom.src_x_all
        self._src_y_all = geom.src_y_all
        self._src_z_all = geom.src_z_all
        self._xi_all = geom.xi_all
        self._interp_px_all = geom.interp_px_all
        self._interp_wgt_all = geom.interp_wgt_all
        self._2d_x_flat = geom._2d_x_flat
        self._2d_y_flat = geom._2d_y_flat
        self._geom_ready = True
        return self

    def _kernel_simulate_one_3d(self, W_eta, src_x, src_y, src_z):
        """3D NUFFT kernel with geometry as explicit arguments (no JIT recompilation).

        Parameters
        ----------
        W_eta : jnp.ndarray, shape (npix_sky, ndelay_eff)
        src_x, src_y, src_z : jnp.ndarray, shape (npix_sky * ndelay_eff,)
        """
        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C_JAX),
            src_x, src_y, src_z,
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
            opts=self._nufft_opts,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _forward_components_from_cache_sky_spec(self, sky_spec, tind, geom=None):
        """Extract forward model components using cached geometry.

        Parameters
        ----------
        sky_spec : jnp.ndarray, shape (npix_sky, nfreq)
        tind : int
            Time index
        geom : TimeGeometry, optional
            Local geometry object. If None, uses self state.
        """
        if geom is None:
            topo = self._topo_all[tind]
            horizon = self._horizon_all[tind]
            beam_spec_h = self._beam_spec_horizon_all[tind]
        else:
            topo = geom.topo_all[tind]
            horizon = geom.horizon_all[tind]
            beam_spec_h = geom.beam_spec_horizon_all[tind]
        W = sky_spec * beam_spec_h
        W_eta_full = jnp.fft.fft(W, axis=1) * self._phase_full[None, :]
        W_eta = W_eta_full[:, self._eta_idx]
        return topo, horizon, beam_spec_h, W, W_eta

    def _forward_components_from_cache(self, sky_coeffs, tind, geom=None):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._forward_components_from_cache_sky_spec(sky_spec, tind, geom=geom)

    def forward_components_one_time(self, sky_coeffs, tind):
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        return self._forward_components_from_cache(sky_coeffs, tind)

    def _simulate_one_precomputed_sky_spec(self, sky_spec, tind, geom=None):
        _, _, _, _, W_eta = self._forward_components_from_cache_sky_spec(sky_spec, tind, geom=geom)
        if geom is None:
            src_x, src_y, src_z = self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind]
        else:
            src_x, src_y, src_z = geom.src_x_all[tind], geom.src_y_all[tind], geom.src_z_all[tind]
        return self._kernel_simulate_one_3d(W_eta, src_x, src_y, src_z)

    def _simulate_one_precomputed(self, sky_coeffs, tind, geom=None):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_precomputed_sky_spec(sky_spec, tind, geom=geom)

    def _simulate_impl(self, sky_coeffs, rot_matrices, geom=None):
        """Core simulate implementation.

        Parameters
        ----------
        sky_coeffs : jnp.ndarray, shape (npix_sky, nmodes_sky)
        rot_matrices : jnp.ndarray, shape (ntime, 3, 3)
        geom : TimeGeometry, optional
            Local geometry object. If None, uses cached state if available,
            otherwise builds local geometry (works in JIT with traced inputs).
        """
        if geom is None:
            # Try to use cached geometry if available and matches
            if self._geom_ready and self._topo_all.shape[0] == rot_matrices.shape[0]:
                # Use cached geometry (geom stays None to use self state)
                pass
            else:
                # Build local geometry (works in JIT because _build_time_geometry uses JAX)
                geom = self._build_time_geometry(rot_matrices)
        else:
            # Validate provided geometry
            if not geom.geom_ready or geom.topo_all.shape[0] != rot_matrices.shape[0]:
                geom = self._build_time_geometry(rot_matrices)
        sky_spec = sky_coeffs @ self.A_sky.T
        inds = jnp.arange(rot_matrices.shape[0], dtype=jnp.int32)
        return jax.vmap(lambda tind: self._simulate_one_precomputed_sky_spec(sky_spec, tind, geom=geom))(inds)

    def simulate_3d(self, sky_coeffs, rot_matrices, t_chunk_size=12):
        """Simulate visibilities using 3D type-3 NUFFT with automatic time chunking.

        Processes time steps in chunks to keep precomputed geometry arrays small.
        For a single chunk, falls back to direct simulation; for multiple chunks,
        accumulates results from each chunk using local geometry per chunk.
        Global geometry cache is not affected by chunking.

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

        # Multi-chunk: use local geometry per chunk, no self mutation
        chunks = []
        for t_start in range(0, ntime, t_chunk_size):
            t_end = min(t_start + t_chunk_size, ntime)
            rot_chunk = rot_matrices[t_start:t_end]
            local_geom = self._build_time_geometry(rot_chunk)
            vis_chunk = self._simulate_impl(sky_coeffs, rot_chunk, geom=local_geom)
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

    def _accumulate_beam_update_impl(self, sky_coeffs, residual_vis, dirty_fn, step_size, sky_reg):
        """Accumulate beam update over time using JAX scan (not NumPy loop).

        Keeps accumulators on device and avoids host/device transfers.
        """
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')

        # Precompute static quantities (keep in JAX to avoid host churn)
        sky_spec_jax = jnp.array(sky_coeffs) @ self.A_sky.T  # (npix_sky, nfreq)
        sky_weight = jnp.abs(sky_spec_jax) ** 2              # (npix_sky, nfreq)
        sky_pow = sky_weight.sum(axis=1)                     # (npix_sky,)

        ntime = int(residual_vis.shape[0])

        # Determine beam mask status once (before scan)
        has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
        npix_beam_accum = int(self._beam_mask.sum()) if has_beam_mask else self.npix_beam

        # Initialize accumulators
        beam_num_init = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
        beam_den_init = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

        # Precompute constants for beam scatter (outside scan for efficiency)
        if has_beam_mask:
            beam_mask_jax = jnp.array(self._beam_mask)
            beam_indices_jax = jnp.array(self._beam_indices)

        # Process one time step (no Python conditionals inside scan)
        if has_beam_mask:
            def one_time_step_masked(carry, tind):
                beam_carry = carry
                # Get dirty apparent sky
                dirty_pf = dirty_fn(residual_vis[tind], tind)
                horizon = self._horizon_all[tind]
                delta_bsf = step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
                delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

                px = self._interp_px_all[tind]  # (4, npix_sky)
                wgt = self._interp_wgt_all[tind]  # (4, npix_sky)

                # Scatter beam contributions for each of 4 neighbors (avoid large temporary)
                beam_num_update = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

                def scatter_neighbor_masked(k, carry):
                    b_num, b_den = carry
                    px_k = px[k]  # (npix_sky,)
                    w_k = wgt[k] * sky_pow  # (npix_sky,)
                    mask_k = beam_mask_jax[px_k]
                    # Compute indices for all pixels; searchsorted handles out-of-range gracefully
                    px_k_masked = jnp.searchsorted(beam_indices_jax, px_k)
                    # Zero out contributions where mask is False to avoid spurious accumulation
                    safe_w = jnp.where(mask_k, w_k, 0.0)
                    safe_delta = jnp.where(mask_k[:, None], delta_bi, 0.0)
                    b_num = b_num.at[px_k_masked].add(safe_w[:, None] * safe_delta)
                    b_den = b_den.at[px_k_masked].add(safe_w)
                    return (b_num, b_den)

                beam_num_update, beam_den_update = jax.lax.fori_loop(
                    0, 4, scatter_neighbor_masked, (beam_num_update, beam_den_update)
                )

                new_beam_carry = (
                    beam_carry[0] + beam_num_update,
                    beam_carry[1] + beam_den_update,
                )

                return new_beam_carry, None

            body_fn = one_time_step_masked
        else:
            def one_time_step_full(carry, tind):
                beam_carry = carry
                # Get dirty apparent sky
                dirty_pf = dirty_fn(residual_vis[tind], tind)
                horizon = self._horizon_all[tind]
                delta_bsf = step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
                delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

                px = self._interp_px_all[tind]
                wgt = self._interp_wgt_all[tind]
                px_flat = px.reshape(-1)
                w_flat = (wgt * sky_pow[None, :]).reshape(-1)
                contrib = (w_flat[:, None] * jnp.repeat(delta_bi[None, :, :], 4, axis=0).reshape(-1, self.nmodes_beam))

                beam_num_update = jnp.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(self.npix_beam, dtype=DTYPE_R_JAX)
                beam_num_update = beam_num_update.at[px_flat].add(contrib)
                beam_den_update = beam_den_update.at[px_flat].add(w_flat)

                new_beam_carry = (
                    beam_carry[0] + beam_num_update,
                    beam_carry[1] + beam_den_update,
                )

                return new_beam_carry, None

            body_fn = one_time_step_full

        # Accumulate over time using jax.lax.scan
        (beam_num, beam_den), _ = jax.lax.scan(
            body_fn,
            (beam_num_init, beam_den_init),
            jnp.arange(ntime),
        )

        # Finalize beam update
        delta_bc_masked = beam_num / (beam_den[:, None] + self.eps)
        delta_bc_masked = delta_bc_masked / ntime

        if has_beam_mask:
            npix_beam_full = len(self._beam_mask)
            delta_bc_full = jnp.zeros((npix_beam_full, self.nmodes_beam), dtype=DTYPE_R_JAX)
            beam_indices_jax = jnp.array(self._beam_indices)
            delta_bc_full = delta_bc_full.at[beam_indices_jax].set(delta_bc_masked)
            delta_bc = delta_bc_full
        else:
            delta_bc = delta_bc_masked

        return delta_bc

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

        # Determine beam mask status once (before scan)
        has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
        npix_beam_accum = int(self._beam_mask.sum()) if has_beam_mask else self.npix_beam

        # Initialize accumulators
        sky_num_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C_JAX)
        sky_den_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_R_JAX)
        beam_num_init = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
        beam_den_init = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

        # Precompute constants for beam scatter (outside scan for efficiency)
        if has_beam_mask:
            beam_mask_jax = jnp.array(self._beam_mask)
            beam_indices_jax = jnp.array(self._beam_indices)

        # Process one time step (no Python conditionals inside scan)
        if has_beam_mask:
            def one_time_step_masked(carry, tind):
                sky_carry, beam_carry = carry
                # Get dirty apparent sky via 3D adjoint (includes IFFT)
                dirty_pf = self.dirty_apparent_sky_one_time(residual_vis[tind], tind)

                # Sky update contribution
                beam_spec_h = self._beam_spec_horizon_all[tind]
                weight = jnp.abs(beam_spec_h) ** 2
                delta_app = sky_step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)

                new_sky_carry = (
                    sky_carry[0] + weight.astype(DTYPE_C_JAX) * delta_app,
                    sky_carry[1] + weight.astype(DTYPE_R_JAX),
                )

                # Beam update contribution
                horizon = self._horizon_all[tind]
                delta_bsf = beam_step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
                delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

                px = self._interp_px_all[tind]  # (4, npix_sky)
                wgt = self._interp_wgt_all[tind]  # (4, npix_sky)

                # Scatter beam contributions for each of 4 neighbors (avoid large temporary)
                beam_num_update = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

                def scatter_neighbor_masked(k, carry):
                    b_num, b_den = carry
                    px_k = px[k]  # (npix_sky,)
                    w_k = wgt[k] * sky_pow  # (npix_sky,)
                    mask_k = beam_mask_jax[px_k]
                    # Compute indices for all pixels; searchsorted handles out-of-range gracefully
                    px_k_masked = jnp.searchsorted(beam_indices_jax, px_k)
                    # Zero out contributions where mask is False to avoid spurious accumulation
                    safe_w = jnp.where(mask_k, w_k, 0.0)
                    safe_delta = jnp.where(mask_k[:, None], delta_bi, 0.0)
                    b_num = b_num.at[px_k_masked].add(safe_w[:, None] * safe_delta)
                    b_den = b_den.at[px_k_masked].add(safe_w)
                    return (b_num, b_den)

                beam_num_update, beam_den_update = jax.lax.fori_loop(
                    0, 4, scatter_neighbor_masked, (beam_num_update, beam_den_update)
                )

                new_beam_carry = (
                    beam_carry[0] + beam_num_update,
                    beam_carry[1] + beam_den_update,
                )

                return (new_sky_carry, new_beam_carry), None

            body_fn = one_time_step_masked
        else:
            def one_time_step_full(carry, tind):
                sky_carry, beam_carry = carry
                # Get dirty apparent sky via 3D adjoint (includes IFFT)
                dirty_pf = self.dirty_apparent_sky_one_time(residual_vis[tind], tind)

                # Sky update contribution
                beam_spec_h = self._beam_spec_horizon_all[tind]
                weight = jnp.abs(beam_spec_h) ** 2
                delta_app = sky_step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)

                new_sky_carry = (
                    sky_carry[0] + weight.astype(DTYPE_C_JAX) * delta_app,
                    sky_carry[1] + weight.astype(DTYPE_R_JAX),
                )

                # Beam update contribution
                horizon = self._horizon_all[tind]
                delta_bsf = beam_step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
                delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

                px = self._interp_px_all[tind]  # (4, npix_sky)
                wgt = self._interp_wgt_all[tind]  # (4, npix_sky)

                # Scatter beam contributions for each of 4 neighbors (avoid large temporary)
                beam_num_update = jnp.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(self.npix_beam, dtype=DTYPE_R_JAX)

                def scatter_neighbor_full(k, carry):
                    b_num, b_den = carry
                    px_k = px[k]  # (npix_sky,)
                    w_k = wgt[k] * sky_pow  # (npix_sky,)
                    b_num = b_num.at[px_k].add(w_k[:, None] * delta_bi)
                    b_den = b_den.at[px_k].add(w_k)
                    return (b_num, b_den)

                beam_num_update, beam_den_update = jax.lax.fori_loop(
                    0, 4, scatter_neighbor_full, (beam_num_update, beam_den_update)
                )

                new_beam_carry = (
                    beam_carry[0] + beam_num_update,
                    beam_carry[1] + beam_den_update,
                )

                return (new_sky_carry, new_beam_carry), None

            body_fn = one_time_step_full

        # Accumulate over time using jax.lax.scan
        ((sky_num, sky_den), (beam_num, beam_den)), _ = jax.lax.scan(
            body_fn,
            ((sky_num_init, sky_den_init), (beam_num_init, beam_den_init)),
            jnp.arange(ntime),
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
        if (
            (not self._geom_ready)
            or self._topo_all.shape[0] != rot_matrices.shape[0]
        ):
            self.precompute_time_geometry(rot_matrices)
        ntime = rot_matrices.shape[0]
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

    def _kernel_nufft2d_W_to_vis(self, W, xi):
        """2D NUFFT kernel with geometry as explicit argument (no JIT recompilation).

        Parameters
        ----------
        W : jnp.ndarray, shape (npix_sky, nfreq), complex
        xi : jnp.ndarray, shape (2, npix_sky)
        """
        return self._nufft2d_W_to_vis(W, xi, self.freqs)

    def _simulate_one_2d_precomputed_sky_spec(self, sky_spec, tind, geom=None):
        """2D hex-rect NUFFT forward for one time step (cached beam).

        Parameters
        ----------
        sky_spec : jnp.ndarray, shape (npix_sky, nfreq)
        tind : int
        geom : TimeGeometry, optional
            Local geometry object. If None, uses self state.

        Returns
        -------
        vis : jnp.ndarray, shape (nfreq, nbls), complex64
        """
        if geom is None:
            beam_spec_h = self._beam_spec_horizon_all[tind]
            xi = self._xi_all[tind]
        else:
            beam_spec_h = geom.beam_spec_horizon_all[tind]
            xi = geom.xi_all[tind]
        W = (sky_spec * beam_spec_h).astype(DTYPE_C_JAX)
        return self._kernel_nufft2d_W_to_vis(W, xi)

    def _simulate_one_2d_precomputed(self, sky_coeffs, tind, geom=None):
        sky_spec = sky_coeffs @ self.A_sky.T
        return self._simulate_one_2d_precomputed_sky_spec(sky_spec, tind, geom=geom)

    def _simulate_one_2d_variable_beam_sky_spec(self, sky_spec, beam_coeffs, tind, geom=None):
        """2D forward for one time step with *beam_coeffs* as a traced input.

        Parameters
        ----------
        sky_spec : jnp.ndarray, shape (npix_sky, nfreq)
        beam_coeffs : jnp.ndarray, shape (npix_beam, nmodes_beam)
        tind : int
        geom : TimeGeometry, optional
            Local geometry object. If None, uses self state.
        """
        if geom is None:
            px = self._interp_px_all[tind]
            wgt = jnp.array(self._interp_wgt_all[tind])
            horizon = self._horizon_all[tind]
            xi = self._xi_all[tind]
        else:
            px = geom.interp_px_all[tind]
            wgt = jnp.array(geom.interp_wgt_all[tind])
            horizon = geom.horizon_all[tind]
            xi = geom.xi_all[tind]
        bi = jnp.sum(wgt[:, :, None] * beam_coeffs[px], axis=0)
        beam_spec_h = (bi @ self.A_beam.T) * horizon[:, None]
        W = (sky_spec * beam_spec_h).astype(DTYPE_C_JAX)
        return self._nufft2d_W_to_vis(W, xi, self.freqs)

    def _kernel_nufft2d_flat(self, W_fi, x_j, y_j):
        """2D NUFFT kernel for flat arrays (no JIT recompilation).

        Parameters
        ----------
        W_fi : jnp.ndarray, shape (npix_sky,), complex
            Flat weight array for one (time, freq)
        x_j, y_j : jnp.ndarray, shape (npix_sky,)
            Flat NUFFT source coordinates
        """
        n_q, n_r = self._2d_n_q, self._2d_n_r
        q_idx, r_idx = self._2d_q_idx, self._2d_r_idx
        grid = nufft1(
            (n_q, n_r), W_fi, x_j, y_j,
            iflag=1, eps=self.eps, opts=self._nufft_opts,
        )
        return grid[q_idx, r_idx]

    def _simulate_2d_impl(self, sky_coeffs, rot_matrices, geom=None):
        """Core 2D simulate implementation.

        Parameters
        ----------
        sky_coeffs : jnp.ndarray, shape (npix_sky, nmodes_sky)
        rot_matrices : jnp.ndarray, shape (ntime, 3, 3)
        geom : TimeGeometry, optional
            Local geometry object. If None, uses cached state if available,
            otherwise builds local geometry (works in JIT with traced inputs).
        """
        if geom is None:
            # Try to use cached geometry if it matches
            if (self._geom_ready and
                self._2d_x_flat is not None and
                self._topo_all.shape[0] == rot_matrices.shape[0]):
                # Cached geometry is valid; use it
                ntime = self._topo_all.shape[0]
                beam_spec_h_all = self._beam_spec_horizon_all
                _2d_x_flat = self._2d_x_flat
                _2d_y_flat = self._2d_y_flat
            else:
                # Geometry not cached or shape mismatch; build locally
                # (_build_time_geometry uses JAX, works in JIT with traced inputs)
                geom = self._build_time_geometry(rot_matrices)
                ntime = geom.topo_all.shape[0]
                beam_spec_h_all = geom.beam_spec_horizon_all
                _2d_x_flat = geom._2d_x_flat
                _2d_y_flat = geom._2d_y_flat
        else:
            # Validate provided geometry
            if not geom.geom_ready or geom._2d_x_flat is None or geom.topo_all.shape[0] != rot_matrices.shape[0]:
                geom = self._build_time_geometry(rot_matrices)
            ntime = geom.topo_all.shape[0]
            beam_spec_h_all = geom.beam_spec_horizon_all
            _2d_x_flat = geom._2d_x_flat
            _2d_y_flat = geom._2d_y_flat

        sky_spec = (sky_coeffs @ self.A_sky.T).astype(DTYPE_C_JAX)   # (npix, nfreq)
        # sky × beam for all times: (ntime, npix, nfreq) → (ntime*nfreq, npix)
        W_flat = (
            (sky_spec[None, :, :] * beam_spec_h_all)
            .transpose(0, 2, 1)
            .reshape(ntime * self.nfreq, self.npix_sky)
        )

        vis_flat = jax.vmap(self._kernel_nufft2d_flat)(W_flat, _2d_x_flat, _2d_y_flat)
        return vis_flat.reshape(ntime, self.nfreq, self.nbls)

    def simulate_2d(self, sky_coeffs, rot_matrices, t_chunk_size=12):
        """Simulate visibilities using 2D hex-rect NUFFT with automatic time chunking.

        Each frequency is handled by a type-1 2D NUFFT that maps sky × beam
        weighted pixels to a compact uniform uv grid; baselines are read from
        that grid at their integer axial positions.

        Processes time steps in chunks to keep precomputed geometry arrays small.
        For a single chunk, falls back to direct simulation; for multiple chunks,
        accumulates results from each chunk using local geometry per chunk.
        Global geometry cache is not affected by chunking.

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

        # Multi-chunk: use local geometry per chunk, no self mutation
        chunks = []
        for t_start in range(0, ntime, t_chunk_size):
            t_end = min(t_start + t_chunk_size, ntime)
            rot_chunk = rot_matrices[t_start:t_end]
            local_geom = self._build_time_geometry(rot_chunk)
            vis_chunk = self._simulate_2d_impl(sky_coeffs, rot_chunk, geom=local_geom)
            chunks.append(vis_chunk)

        return jnp.concatenate(chunks, axis=0)

    def simulate_variable_beam_2d(self, sky_coeffs, beam_coeffs, rot_matrices):
        """Simulate all times with explicit *beam_coeffs* via the 2D path.

        Uses vmap over (time, freq) to avoid Python loop over time steps.
        """
        if (
            (not self._geom_ready)
            or self._topo_all.shape[0] != rot_matrices.shape[0]
        ):
            self.precompute_time_geometry(rot_matrices)
        ntime = self._topo_all.shape[0]

        # Precompute beam_spec_h_all from beam_coeffs for all times
        interp_px_all = jnp.array(self._interp_px_all)  # (ntime, 4, npix_sky)
        interp_wgt_all = jnp.array(self._interp_wgt_all)  # (ntime, 4, npix_sky)
        horizon_all = self._horizon_all  # (ntime, npix_sky)
        beam_coeffs_jax = jnp.array(beam_coeffs, dtype=DTYPE_R_JAX)
        A_beam_jax = jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

        # Interpolate beam at all times: (ntime, 4, npix_sky, nmodes_beam)
        beam_interp = beam_coeffs_jax[interp_px_all]
        # Sum over 4 neighbors: (ntime, npix_sky, nmodes_beam)
        bi_all = jnp.sum(interp_wgt_all[:, :, :, None] * beam_interp, axis=1)
        # Apply horizon and go to frequency domain: (ntime, npix_sky, nfreq)
        beam_spec_h_all = (bi_all @ A_beam_jax.T) * horizon_all[:, :, None]

        # Compute sky × beam for all times using fixed-beam vmap pattern
        sky_spec = (sky_coeffs @ self.A_sky.T).astype(DTYPE_C_JAX)  # (npix_sky, nfreq)
        W_flat = (
            (sky_spec[None, :, :] * beam_spec_h_all)
            .transpose(0, 2, 1)
            .reshape(ntime * self.nfreq, self.npix_sky)
        )

        vis_flat = jax.vmap(self._kernel_nufft2d_flat)(W_flat, self._2d_x_flat, self._2d_y_flat)
        return vis_flat.reshape(ntime, self.nfreq, self.nbls)

    def beam_spec_horizon_from_coeffs_2d(self, beam_coeffs, rot_matrices):
        """Return beam spectra on sky pixels for explicit coefficients via the 2D path."""
        if (
            (not self._geom_ready)
            or self._topo_all.shape[0] != rot_matrices.shape[0]
        ):
            self.precompute_time_geometry(rot_matrices)
        interp_px_all = jnp.array(self._interp_px_all)
        interp_wgt_all = jnp.array(self._interp_wgt_all)
        beam_coeffs_jax = jnp.array(beam_coeffs, dtype=DTYPE_R_JAX)
        beam_interp = beam_coeffs_jax[interp_px_all]
        bi_all = jnp.sum(interp_wgt_all[:, :, :, None] * beam_interp, axis=1)
        return (bi_all @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX).T) * self._horizon_all[:, :, None]

    def simulate_2d_with_beam_spec_horizon(self, sky_coeffs, beam_spec_h_all):
        """Simulate all times with precomputed beam spectra on sky pixels."""
        ntime = beam_spec_h_all.shape[0]
        sky_spec = (sky_coeffs @ self.A_sky.T).astype(DTYPE_C_JAX)
        W_flat = (
            (sky_spec[None, :, :] * beam_spec_h_all)
            .transpose(0, 2, 1)
            .reshape(ntime * self.nfreq, self.npix_sky)
        )
        vis_flat = jax.vmap(self._kernel_nufft2d_flat)(W_flat, self._2d_x_flat, self._2d_y_flat)
        return vis_flat.reshape(ntime, self.nfreq, self.nbls)

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

        # Determine beam mask status once (before scan)
        has_beam_mask = hasattr(self, '_beam_mask') and self._beam_mask is not None
        npix_beam_accum = int(self._beam_mask.sum()) if has_beam_mask else self.npix_beam

        # Initialize accumulators
        sky_num_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C_JAX)
        sky_den_init = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_R_JAX)
        beam_num_init = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
        beam_den_init = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

        # Precompute constants for beam scatter (outside scan for efficiency)
        if has_beam_mask:
            beam_mask_jax = jnp.array(self._beam_mask)
            beam_indices_jax = jnp.array(self._beam_indices)

        # Process one time step (no Python conditionals inside scan)
        if has_beam_mask:
            def one_time_step_masked(carry, tind):
                sky_carry, beam_carry = carry
                # Get dirty apparent sky via 2D adjoint
                dirty_pf = self.dirty_apparent_sky_one_time_2d(residual_vis[tind], tind)

                # Sky update contribution
                beam_spec_h = self._beam_spec_horizon_all[tind]
                weight = jnp.abs(beam_spec_h) ** 2
                delta_app = sky_step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)

                new_sky_carry = (
                    sky_carry[0] + weight.astype(DTYPE_C_JAX) * delta_app,
                    sky_carry[1] + weight.astype(DTYPE_R_JAX),
                )

                # Beam update contribution: interpolate and scatter
                horizon = self._horizon_all[tind]
                delta_bsf = beam_step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
                delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

                px = self._interp_px_all[tind]  # (4, npix_sky)
                wgt = self._interp_wgt_all[tind]  # (4, npix_sky)

                # Scatter beam contributions for each of 4 neighbors (avoid large temporary)
                beam_num_update = jnp.zeros((npix_beam_accum, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(npix_beam_accum, dtype=DTYPE_R_JAX)

                def scatter_neighbor_masked_2d(k, carry):
                    b_num, b_den = carry
                    px_k = px[k]  # (npix_sky,)
                    w_k = wgt[k] * sky_pow  # (npix_sky,)
                    mask_k = beam_mask_jax[px_k]
                    px_k_safe = jnp.where(mask_k, px_k, 0)
                    px_k_masked = jnp.searchsorted(beam_indices_jax, px_k_safe)
                    contrib = jnp.where(mask_k[:, None], w_k[:, None] * delta_bi, 0.0)
                    den_contrib = jnp.where(mask_k, w_k, 0.0)
                    b_num = b_num.at[px_k_masked].add(contrib)
                    b_den = b_den.at[px_k_masked].add(den_contrib)
                    return (b_num, b_den)

                beam_num_update, beam_den_update = jax.lax.fori_loop(
                    0, 4, scatter_neighbor_masked_2d, (beam_num_update, beam_den_update)
                )

                new_beam_carry = (
                    beam_carry[0] + beam_num_update,
                    beam_carry[1] + beam_den_update,
                )

                return (new_sky_carry, new_beam_carry), None

            body_fn = one_time_step_masked
        else:
            def one_time_step_full(carry, tind):
                sky_carry, beam_carry = carry
                # Get dirty apparent sky via 2D adjoint
                dirty_pf = self.dirty_apparent_sky_one_time_2d(residual_vis[tind], tind)

                # Sky update contribution
                beam_spec_h = self._beam_spec_horizon_all[tind]
                weight = jnp.abs(beam_spec_h) ** 2
                delta_app = sky_step_size * jnp.conj(beam_spec_h.astype(DTYPE_C_JAX)) * dirty_pf / (weight.astype(DTYPE_C_JAX) + beam_reg)

                new_sky_carry = (
                    sky_carry[0] + weight.astype(DTYPE_C_JAX) * delta_app,
                    sky_carry[1] + weight.astype(DTYPE_R_JAX),
                )

                # Beam update contribution: interpolate and scatter
                horizon = self._horizon_all[tind]
                delta_bsf = beam_step_size * jnp.conj(sky_spec_jax) * dirty_pf / (sky_weight + sky_reg)
                delta_bi = (jnp.real(delta_bsf) * horizon[:, None]) @ jnp.array(self.A_beam, dtype=DTYPE_R_JAX)

                px = self._interp_px_all[tind]
                wgt = self._interp_wgt_all[tind]
                px_flat = px.reshape(-1)
                w_flat = (wgt * sky_pow[None, :]).reshape(-1)
                contrib = (w_flat[:, None] * jnp.repeat(delta_bi[None, :, :], 4, axis=0).reshape(-1, self.nmodes_beam))

                beam_num_update = jnp.zeros((self.npix_beam, self.nmodes_beam), dtype=DTYPE_R_JAX)
                beam_den_update = jnp.zeros(self.npix_beam, dtype=DTYPE_R_JAX)
                beam_num_update = beam_num_update.at[px_flat].add(contrib)
                beam_den_update = beam_den_update.at[px_flat].add(w_flat)

                new_beam_carry = (
                    beam_carry[0] + beam_num_update,
                    beam_carry[1] + beam_den_update,
                )

                return (new_sky_carry, new_beam_carry), None

            body_fn = one_time_step_full

        # Accumulate over time using jax.lax.scan
        ((sky_num, sky_den), (beam_num, beam_den)), _ = jax.lax.scan(
            body_fn,
            ((sky_num_init, sky_den_init), (beam_num_init, beam_den_init)),
            jnp.arange(ntime),
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
