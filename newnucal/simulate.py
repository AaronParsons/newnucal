"""
Forward visibility simulation.

ForwardModel maps sky DPSS coefficients → model visibilities via a 3-D type-3
NUFFT in (l, m, eta). This version also exposes adjoint/backprojection helpers
for dirty-map sky updates.
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
        sky_nside: int,
        beam_model: BeamModel,
        freqs,
        eps: float = 1e-6,
        eta_max: float | None = None,
        eta_padding: float = 0.0,
    ):
        self.array = array
        self.sky_nside = sky_nside
        self.beam_nside = beam_model.nside
        self.eps = eps
        self.eta_max = eta_max
        self.eta_padding = float(eta_padding)

        freqs_np = np.asarray(freqs, dtype=np.float64)
        self.freqs = jnp.array(freqs_np, dtype=DTYPE_R)

        self.A_beam = jnp.array(beam_model.A_beam, dtype=DTYPE_R)
        self.beam_coeffs = jnp.array(beam_model.coeffs, dtype=DTYPE_R)

        npix_sky = healpy.nside2npix(sky_nside)
        eq_xyz = np.array(
            healpy.pix2vec(sky_nside, np.arange(npix_sky)), dtype=DTYPE_R_NPY
        )
        self.eq_coords = jnp.array(eq_xyz)

        nfreq = len(freqs_np)
        dnu = float(freqs_np[1] - freqs_np[0])
        nu_0 = float(freqs_np[0])
        m_arr = np.arange(nfreq)

        # Wrapped FFT delay grid.
        eta_np = np.fft.fftfreq(nfreq, d=dnu)
        phase_np = np.exp(-2j * np.pi * eta_np * nu_0) / nfreq

        self._eta_full = jnp.array(eta_np.astype(np.float32))
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

        # Beam diameter for FWHM-based zenith cut (metres; None if unavailable)
        self._beam_diameter = getattr(getattr(beam_model, 'beam', None), 'diameter', None)

        self.A_sky = None
        self._jit_one = None

        self._geom_ready = False
        self._topo_all = None
        self._horizon_all = None
        self._beam_px_all = None
        self._beam_wgts_all = None
        self._src_x_all = None
        self._src_y_all = None
        self._src_z_all = None

        # Per-time zenith-cut state (populated by precompute_time_geometry)
        self._zenith_cut_active = False
        self._zenith_px_idx_list    = None   # list[np.ndarray(npix_t,)]
        self._zenith_beam_spec_list = None   # list[jnp.ndarray(npix_t, nfreq)] precomputed beam×horizon
        self._zenith_src_x_list     = None   # list[jnp.ndarray(npix_t*ndelay,)]
        self._zenith_src_y_list     = None
        self._zenith_src_z_list     = None

    # ------------------------------------------------------------------
    # Pixel masking helpers
    # ------------------------------------------------------------------

    def build_ever_visible_mask(self, rot_matrices):
        """
        Return a boolean array of shape (npix_sky,) that is True for every
        sky pixel above the horizon (topo_z > 0) at any of the given times.

        Call this before :meth:`apply_pixel_mask`.
        """
        rot_ms = np.asarray(rot_matrices, dtype=np.float32)
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
        for attr in ('_topo_all', '_horizon_all', '_beam_px_all', '_beam_wgts_all',
                     '_src_x_all', '_src_y_all', '_src_z_all'):
            setattr(self, attr, None)
        self._zenith_cut_active = False
        return self

    def set_sky_dpss(self, A_sky):
        self.A_sky = jnp.array(A_sky, dtype=DTYPE_R)
        self._jit_one = jax.jit(self._simulate_one_precomputed)

    def precompute_time_geometry(self, rot_matrices, zenith_cut_scale=None):
        """Precompute time-only geometry and interpolation operators.

        Parameters
        ----------
        rot_matrices : array_like, shape (ntime, 3, 3)
        zenith_cut_scale : float or None
            If given, enable a per-time dynamic zenith cut.  Only pixels
            within ``zenith_cut_scale × FWHM`` of zenith are included in
            each time step's NUFFT.  FWHM is the Airy-beam first-null
            width (1.22 λ/D) evaluated at the *lowest* frequency (widest
            beam).  Requires the beam to have a ``diameter`` attribute.
            ``scale=1`` keeps everything inside one FWHM; ``scale=0.5``
            keeps only the inner half-beam.  When active, ``simulate``
            uses a Python loop rather than vmap so that each time step
            passes only its active source points to the NUFFT.
        """
        # Reset zenith-cut state
        self._zenith_cut_active = False
        self._zenith_px_idx_list = None
        self._zenith_src_x_list = None
        self._zenith_src_y_list = None
        self._zenith_src_z_list = None

        rot_ms = jnp.array(rot_matrices, dtype=DTYPE_R)
        ntime = int(rot_ms.shape[0])
        npix = self.npix_sky
        ndelay = self.ndelay_eff
        two_pi = 2.0 * np.pi

        # Zenith-cut threshold: cos(theta_cut) = cos(scale * FWHM)
        cos_theta_cut = None
        if zenith_cut_scale is not None:
            if self._beam_diameter is None:
                raise ValueError(
                    'zenith_cut_scale requires the beam model to have a '
                    '"diameter" attribute (e.g. AiryBeam).'
                )
            nu_min = float(self.freqs.min())
            fwhm_rad = 1.22 * C / (nu_min * self._beam_diameter)
            theta_cut = zenith_cut_scale * fwhm_rad
            cos_theta_cut = float(np.cos(theta_cut))

        topo_all = []
        horizon_all = []
        beam_px_all = []
        beam_wgts_all = []
        src_x_all = []
        src_y_all = []
        src_z_all = []

        zenith_px_idx_list    = [] if cos_theta_cut is not None else None
        zenith_beam_spec_list = [] if cos_theta_cut is not None else None
        zenith_src_x_list     = [] if cos_theta_cut is not None else None
        zenith_src_y_list     = [] if cos_theta_cut is not None else None
        zenith_src_z_list     = [] if cos_theta_cut is not None else None

        eta_np = np.asarray(self._eta, dtype=np.float32)
        for i in range(ntime):
            topo = np.array(rot_ms[i] @ self.eq_coords, dtype=np.float32)
            horizon = (topo[2] > 0).astype(np.float32)
            topo_th, topo_ph = healjax.vec2ang(
                jnp.array(topo[0]), jnp.array(topo[1]), jnp.array(topo[2])
            )
            px, wgts = get_interp_weights(topo_th, topo_ph, self.beam_nside)
            px = np.array(px, dtype=np.int32)
            wgts = np.array(wgts, dtype=np.float32)

            src_x = np.repeat(topo[0], ndelay).astype(np.float32) * two_pi
            src_y = np.repeat(topo[1], ndelay).astype(np.float32) * two_pi
            src_z = np.tile(eta_np, npix).astype(np.float32) * two_pi

            topo_all.append(topo)
            horizon_all.append(horizon)
            beam_px_all.append(px)
            beam_wgts_all.append(wgts)
            src_x_all.append(src_x)
            src_y_all.append(src_y)
            src_z_all.append(src_z)

            if cos_theta_cut is not None:
                # Pixels within zenith cut radius (horizon mask already folded in via
                # beam_spec × horizon below, so we keep all zenith-cut pixels here).
                active = (topo[2] > cos_theta_cut)
                px_idx = np.where(active)[0].astype(np.int32)
                zenith_px_idx_list.append(px_idx)

                # Precompute beam_spec × horizon for active pixels so that
                # _simulate_one_zenith never touches beam_coeffs or px arrays
                # inside JIT (avoiding slow XLA constant-folding of large gathers).
                # px/wgts have shape (4, npix_sky); select active columns.
                px_z    = px[:, px_idx]     # (4, npix_active)
                wgts_z  = wgts[:, px_idx]   # (4, npix_active)
                horiz_z = horizon[px_idx]   # (npix_active,)
                bc_np   = np.asarray(self.beam_coeffs)  # (npix_beam, nmodes_beam)
                bi_np   = np.sum(wgts_z[:, :, None] * bc_np[px_z], axis=0)  # (npix_active, nmodes_beam)
                bs_np   = (bi_np @ np.asarray(self.A_beam).T) * horiz_z[:, None]  # (npix_active, nfreq)
                zenith_beam_spec_list.append(jnp.array(bs_np, dtype=DTYPE_R))

                # Variable-length NUFFT source coords for active pixels
                src_xm = src_x.reshape(npix, ndelay)[px_idx].ravel()
                src_ym = src_y.reshape(npix, ndelay)[px_idx].ravel()
                src_zm = src_z.reshape(npix, ndelay)[px_idx].ravel()
                zenith_src_x_list.append(jnp.array(src_xm, dtype=DTYPE_R))
                zenith_src_y_list.append(jnp.array(src_ym, dtype=DTYPE_R))
                zenith_src_z_list.append(jnp.array(src_zm, dtype=DTYPE_R))

        self._topo_all = jnp.array(np.stack(topo_all), dtype=DTYPE_R)
        self._horizon_all = jnp.array(np.stack(horizon_all), dtype=DTYPE_R)
        self._beam_px_all = jnp.array(np.stack(beam_px_all), dtype=jnp.int32)
        self._beam_wgts_all = jnp.array(np.stack(beam_wgts_all), dtype=DTYPE_R)
        self._src_x_all = jnp.array(np.stack(src_x_all), dtype=DTYPE_R)
        self._src_y_all = jnp.array(np.stack(src_y_all), dtype=DTYPE_R)
        self._src_z_all = jnp.array(np.stack(src_z_all), dtype=DTYPE_R)

        if cos_theta_cut is not None:
            self._zenith_cut_active = True
            self._zenith_px_idx_list    = zenith_px_idx_list
            self._zenith_beam_spec_list = zenith_beam_spec_list
            self._zenith_src_x_list     = zenith_src_x_list
            self._zenith_src_y_list     = zenith_src_y_list
            self._zenith_src_z_list     = zenith_src_z_list

        self._geom_ready = True
        # Invalidate any cached JIT compilation: the precomputed arrays are
        # captured by value in the trace, so a fresh jit is needed after each
        # geometry recompute to avoid stale compiled code.
        if self.A_sky is not None:
            self._jit_one = jax.jit(self._simulate_one_precomputed)
        return self

    def _simulate_one_zenith(self, sky_coeffs, tind):
        """Forward model for one time step restricted to the per-time zenith-cut pixels.

        Uses variable-length source arrays; must be called from a Python loop
        (not vmap) because array sizes differ across time steps.

        ``beam_spec × horizon`` is precomputed during ``precompute_time_geometry``
        so this function only performs a gather + matmul on ``sky_coeffs``, avoiding
        large constant-gather operations that trigger slow XLA constant folding.
        """
        px_idx    = self._zenith_px_idx_list[tind]       # numpy (npix_active,)
        beam_spec = self._zenith_beam_spec_list[tind]    # (npix_active, nfreq) precomputed const

        # sky_coeffs[px_idx] is a gather with constant indices but dynamic data →
        # XLA does NOT constant-fold this; it stays dynamic throughout.
        sky_spec = sky_coeffs[px_idx] @ self.A_sky.T    # (npix_active, nfreq)
        W = sky_spec * beam_spec                         # (npix_active, nfreq) dynamic

        W_eta_full = jnp.fft.fft(W, axis=1) * self._phase_full[None, :]
        W_eta = W_eta_full[:, self._eta_idx]             # (npix_active, ndelay)

        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C),
            self._zenith_src_x_list[tind],
            self._zenith_src_y_list[tind],
            self._zenith_src_z_list[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _simulate_loop(self, sky_coeffs):
        """Simulate all time steps using per-time zenith-cut pixel lists.

        The Python loop is unrolled by JAX's tracing when this is jit-compiled,
        so each time step's source coordinates (fixed numpy arrays) become
        compile-time constants, enabling gradient computation through
        sky_coeffs while still passing only the active pixels to each NUFFT.
        """
        return jnp.stack([
            self._simulate_one_zenith(sky_coeffs, t)
            for t in range(len(self._zenith_px_idx_list))
        ])

    def _forward_components_from_cache(self, sky_coeffs, tind):
        topo = self._topo_all[tind]
        horizon = self._horizon_all[tind]
        px = self._beam_px_all[tind]
        wgts = self._beam_wgts_all[tind]
        beam_interp = jnp.sum(wgts[:, :, None] * self.beam_coeffs[px], axis=0)
        sky_spec = sky_coeffs @ self.A_sky.T
        beam_spec = beam_interp @ self.A_beam.T
        W = sky_spec * beam_spec * horizon[:, None]
        W_eta_full = jnp.fft.fft(W, axis=1) * self._phase_full[None, :]
        W_eta = W_eta_full[:, self._eta_idx]
        return topo, horizon, beam_interp, beam_spec, W, W_eta

    def forward_components_one_time(self, sky_coeffs, tind):
        if not self._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')
        return self._forward_components_from_cache(sky_coeffs, tind)

    def _simulate_one_precomputed(self, sky_coeffs, tind):
        _, _, _, _, _, W_eta = self._forward_components_from_cache(sky_coeffs, tind)
        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C),
            self._src_x_all[tind], self._src_y_all[tind], self._src_z_all[tind],
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
        )
        return vis_flat.reshape(self.nfreq, self.nbls)

    def _simulate_one(self, sky_coeffs, rot_m):
        if not self._geom_ready:
            self.precompute_time_geometry(rot_m[None, ...])
            return self._simulate_one_precomputed(sky_coeffs, 0)
        raise RuntimeError('_simulate_one() requires precomputed geometry; use simulate().')

    def simulate(self, sky_coeffs, rot_matrices):
        if self._jit_one is None:
            raise RuntimeError('Call set_sky_dpss() before simulate().')
        if (not self._geom_ready) or (self._topo_all.shape[0] != rot_matrices.shape[0]):
            self.precompute_time_geometry(rot_matrices)
        if self._zenith_cut_active:
            return self._simulate_loop(sky_coeffs)
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
        sky_coeffs,
        tind,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Form a beam-weighted apparent-sky correction for one time."""
        dirty_pf = self.dirty_apparent_sky_one_time(residual_fb, tind)
        _, horizon, beam_interp, beam_spec, _, _ = self._forward_components_from_cache(sky_coeffs, tind)
        beam_spec = beam_spec * horizon[:, None]
        weight = jnp.abs(beam_spec) ** 2
        delta_app = step_size * jnp.conj(beam_spec.astype(DTYPE_C)) * dirty_pf / (weight.astype(DTYPE_C) + beam_reg)
        return delta_app, weight

    def apparent_to_equatorial_one_time(self, app_pf, tind):
        """Map a one-time apparent-sky update to the equatorial sky grid.

        In the current implementation the apparent sky update is already indexed
        by equatorial sky pixels, because the forward model rotates equatorial
        sky pixels into topocentric coordinates rather than regridding onto a
        topocentric HEALPix mesh. This method is kept as a hook for future
        resampling schemes.
        """
        return app_pf

    def accumulate_equatorial_sky_update(
        self,
        residual_vis,
        sky_coeffs,
        step_size: float = 1.0,
        beam_reg: float = 1e-3,
    ):
        """Accumulate a global equatorial sky update from all times."""
        num = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_C)
        den = jnp.zeros((self.npix_sky, self.nfreq), dtype=DTYPE_R)
        for tind in range(int(residual_vis.shape[0])):
            delta_app, weight_app = self.apparent_sky_update_one_time(
                residual_vis[tind], sky_coeffs, tind, step_size=step_size, beam_reg=beam_reg
            )
            delta_eq = self.apparent_to_equatorial_one_time(delta_app, tind)
            weight_eq = self.apparent_to_equatorial_one_time(weight_app, tind).real.astype(DTYPE_R)
            num = num + weight_eq.astype(DTYPE_C) * delta_eq
            den = den + weight_eq
        delta_eq_pf = num / (den.astype(DTYPE_C) + 1e-6)
        delta_eq_pf = delta_eq_pf / residual_vis.shape[0]  # divide by ntimes
        delta_coeffs = (delta_eq_pf.real @ self.A_sky).astype(DTYPE_R)
        return delta_coeffs

    @property
    def npix_sky(self) -> int:
        # eq_coords has shape (3, npix_active); reflects any applied pixel mask
        return int(self.eq_coords.shape[1])

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

    rot_ms = np.empty((len(times), 3, 3), dtype=np.float32)
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
