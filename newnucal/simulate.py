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

    def set_sky_dpss(self, A_sky):
        self.A_sky = jnp.array(A_sky, dtype=DTYPE_R)
        self._jit_one = jax.jit(self._simulate_one_precomputed)

    def precompute_time_geometry(self, rot_matrices):
        """Precompute time-only geometry and interpolation operators."""
        rot_ms = jnp.array(rot_matrices, dtype=DTYPE_R)
        ntime = int(rot_ms.shape[0])
        npix = self.npix_sky
        ndelay = self.ndelay_eff
        two_pi = 2.0 * np.pi

        topo_all = []
        horizon_all = []
        beam_px_all = []
        beam_wgts_all = []
        src_x_all = []
        src_y_all = []
        src_z_all = []

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

        self._topo_all = jnp.array(np.stack(topo_all), dtype=DTYPE_R)
        self._horizon_all = jnp.array(np.stack(horizon_all), dtype=DTYPE_R)
        self._beam_px_all = jnp.array(np.stack(beam_px_all), dtype=jnp.int32)
        self._beam_wgts_all = jnp.array(np.stack(beam_wgts_all), dtype=DTYPE_R)
        self._src_x_all = jnp.array(np.stack(src_x_all), dtype=DTYPE_R)
        self._src_y_all = jnp.array(np.stack(src_y_all), dtype=DTYPE_R)
        self._src_z_all = jnp.array(np.stack(src_z_all), dtype=DTYPE_R)
        self._geom_ready = True
        return self

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
        return healpy.nside2npix(self.sky_nside)

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
