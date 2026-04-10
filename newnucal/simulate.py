"""
Forward visibility simulation.

ForwardModel maps sky DPSS coefficients → model visibilities via:

  1. Rotate equatorial sky-pixel unit vectors to topocentric frame  (R(t) @ eq)
  2. Apply horizon mask  (topo_z > 0)
  3. Interpolate beam DPSS coefficients at rotated sky-pixel positions
     using bilinear HEALPix interpolation (healjax)
  4. Reconstruct per-pixel sky and beam spectra, multiply, apply horizon mask:
       W[p, f] = sky(ν_f, p) * beam(ν_f, p, t) * horizon[p]
  5. DFT W over the frequency axis → delay domain:
       W_η[p, m] = FFT_f{W[p, f]}[m] * exp(-2πi η_m ν_0) / nfreq
     so that  W[p, f] = Σ_m W_η[p, m] exp(2πi η_m ν_f)  exactly.
  6. jax_finufft.nufft3 (type-3, non-uniform → non-uniform):

       sources : (2π * topo_x_p, 2π * topo_y_p, 2π * η_m)  per (pixel, delay)
       targets : (b_x * ν / c,   b_y * ν / c,   ν)          per (freq, baseline)

     computes  V[f,b] = Σ_{p,m} W_η[p,m] exp(2πi ν_f (η_m + b·topo_p / c))
                      = Σ_p W[p,f] exp(2πi ν_f b·topo_p / c)   (exact)

Time loop uses jax.lax.scan over pre-computed rotation matrices.
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
    DPSS).  Per time step, the sky × beam spectral product is transformed to
    the delay domain via FFT, producing source weights W_η[p, m].  A single
    type-3 NUFFT then maps all (pixel, delay) sources to all (freq, baseline)
    targets simultaneously.

    Parameters
    ----------
    array : HERAArray
    sky_nside : int
        HEALPix resolution for the sky model.
    beam_model : BeamModel
    freqs : array_like, shape (nfreq,)
        Frequency array in Hz.  Must match beam_model.freqs.
    eps : float
        NUFFT accuracy parameter.
    """

    def __init__(
        self,
        array: HERAArray,
        sky_nside: int,
        beam_model: BeamModel,
        freqs,
        eps: float = 1e-6,
    ):
        self.array = array
        self.sky_nside = sky_nside
        self.beam_nside = beam_model.nside
        self.eps = eps

        freqs_np = np.asarray(freqs, dtype=np.float64)
        self.freqs = jnp.array(freqs_np, dtype=DTYPE_R)

        # Beam model (fixed)
        self.A_beam = jnp.array(beam_model.A_beam, dtype=DTYPE_R)       # (nfreq, nmodes_beam)
        self.beam_coeffs = jnp.array(beam_model.coeffs, dtype=DTYPE_R)  # (npix_beam, nmodes_beam)

        # Equatorial unit vectors for all sky pixels
        npix_sky = healpy.nside2npix(sky_nside)
        eq_xyz = np.array(
            healpy.pix2vec(sky_nside, np.arange(npix_sky)), dtype=DTYPE_R_NPY
        )
        self.eq_coords = jnp.array(eq_xyz)  # (3, npix_sky)

        # ------------------------------------------------------------------
        # Delay grid and per-mode phase correction — depend only on freqs
        # ------------------------------------------------------------------
        nfreq = len(freqs_np)
        dnu   = float(freqs_np[1] - freqs_np[0])
        nu_0  = float(freqs_np[0])
        m_arr = np.arange(nfreq)

        # η_m = m / (nfreq * Δν)  seconds (m > nfreq/2 are wrapped negative delays)
        eta_np = m_arr / (nfreq * dnu)

        # After FFT over f, multiply by phase[m] so that
        #   W[p, f] = Σ_m (FFT(W)[p,m] * phase[m]) * exp(2πi η_m ν_f)
        phase_np = np.exp(-2j * np.pi * m_arr * nu_0 / (nfreq * dnu)) / nfreq

        self._eta   = jnp.array(eta_np.astype(np.float32))      # (nfreq,)
        self._phase = jnp.array(phase_np.astype(np.complex64))  # (nfreq,)

        # ------------------------------------------------------------------
        # NUFFT target coordinates — fixed across all time steps
        # ------------------------------------------------------------------
        #   tgt[f * nbls + b] = (ν_f * b_x/c,  ν_f * b_y/c,  ν_f)
        bls_np = np.asarray(array.bls, dtype=np.float64)  # (nbls, 3)
        nbls   = bls_np.shape[0]
        tgt_x  = (freqs_np[:, None] * bls_np[None, :, 0] / C).ravel()
        tgt_y  = (freqs_np[:, None] * bls_np[None, :, 1] / C).ravel()
        tgt_z  = np.repeat(freqs_np, nbls)   # ν_f repeated for each baseline

        self._tgt_x = jnp.array(tgt_x, dtype=DTYPE_R)
        self._tgt_y = jnp.array(tgt_y, dtype=DTYPE_R)
        self._tgt_z = jnp.array(tgt_z, dtype=DTYPE_R)

        # Sky DPSS matrix — set by set_sky_dpss()
        self.A_sky    = None
        self._jit_one = None  # compiled after set_sky_dpss

    # ------------------------------------------------------------------
    # Sky DPSS matrix setter
    # ------------------------------------------------------------------

    def set_sky_dpss(self, A_sky):
        """
        Set the sky DPSS matrix and recompile the JIT-compiled step.

        Parameters
        ----------
        A_sky : array_like, shape (nfreq, nmodes_sky)
        """
        self.A_sky    = jnp.array(A_sky, dtype=DTYPE_R)
        self._jit_one = jax.jit(self._simulate_one)

    # ------------------------------------------------------------------
    # Core simulation (single time step) — pure JAX, differentiable
    # ------------------------------------------------------------------

    def _simulate_one(self, sky_coeffs, rot_m):
        """
        Simulate visibilities for a single integration.

        Parameters
        ----------
        sky_coeffs : jnp.array, shape (npix_sky, nmodes_sky)
            DPSS sky coefficients in equatorial coordinates.
        rot_m : jnp.array, shape (3, 3)
            Rotation matrix: equatorial → topocentric.

        Returns
        -------
        vis : jnp.array, shape (nfreq, nbls), complex64
        """
        # 1. Rotate sky pixels to topocentric frame
        topo = jnp.dot(rot_m, self.eq_coords)  # (3, npix_sky)

        # 2. Horizon mask — multiplicative, keeps shapes static
        horizon = (topo[2] > 0).astype(DTYPE_R)  # (npix_sky,)

        # 3. Interpolate beam DPSS coefficients at topocentric sky positions
        topo_th, topo_ph = healjax.vec2ang(topo[0], topo[1], topo[2])
        px, wgts = get_interp_weights(topo_th, topo_ph, self.beam_nside)
        beam_interp = jnp.sum(
            wgts[:, :, None] * self.beam_coeffs[px], axis=0
        )  # (npix_sky, nmodes_beam)

        # 4. Reconstruct per-pixel spectra and form the product
        sky_spec  = sky_coeffs  @ self.A_sky.T    # (npix_sky, nfreq)
        beam_spec = beam_interp @ self.A_beam.T   # (npix_sky, nfreq)
        W = sky_spec * beam_spec * horizon[:, None]  # (npix_sky, nfreq)

        # 5. DFT W over the frequency axis → delay domain
        #
        #   W_η[p, m] = FFT_f{W[p, f]}[m] * phase[m]
        #
        #   chosen so that  W[p, f] = Σ_m W_η[p, m] exp(2πi η_m ν_f)  exactly,
        #   where  η_m = m / (nfreq Δν)  and  phase[m] = exp(-2πi η_m ν_0) / nfreq.
        W_eta = jnp.fft.fft(W, axis=1) * self._phase[None, :]   # (npix_sky, nfreq), complex

        # 6. Source coordinates for NUFFT type-3
        #    Ordering: pixel p is outer, delay mode m is inner
        #    → flat index p * nfreq + m
        two_pi = 2.0 * jnp.pi
        src_x = jnp.repeat(topo[0], self.nfreq) * two_pi   # (npix * nfreq,)
        src_y = jnp.repeat(topo[1], self.nfreq) * two_pi
        src_z = jnp.tile(self._eta, self.npix_sky) * two_pi  # η cycles per pixel

        # 7. Type-3 NUFFT:  non-uniform sources → non-uniform targets
        #
        #   f_j = Σ_k c_k exp(i s_k · t_j)            [iflag = +1]
        #
        #   = Σ_{p,m} W_η[p,m] exp(i (2π topo_x · b_x ν/c
        #                             + 2π topo_y · b_y ν/c
        #                             + 2π η_m · ν))
        #   = Σ_{p,m} W_η[p,m] exp(2πi ν_f (b·topo_p/c + η_m))
        #   = Σ_p W[p,f] exp(2πi ν_f b·topo_p/c)   (exact, by construction)
        vis_flat = nufft3(
            W_eta.ravel().astype(DTYPE_C),
            src_x, src_y, src_z,
            self._tgt_x, self._tgt_y, self._tgt_z,
            iflag=1,
            eps=self.eps,
        )  # (nfreq * nbls,)

        return vis_flat.reshape(self.nfreq, self.nbls)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(self, sky_coeffs, rot_matrices):
        """
        Simulate visibilities over multiple integrations.

        Parameters
        ----------
        sky_coeffs : jnp.array, shape (npix_sky, nmodes_sky)
        rot_matrices : jnp.array, shape (ntime, 3, 3)
            Pre-computed equatorial → topocentric rotation matrices.

        Returns
        -------
        vis : jnp.array, shape (ntime, nfreq, nbls), complex64
        """
        if self._jit_one is None:
            raise RuntimeError("Call set_sky_dpss() before simulate().")

        def step(_, rot_m):
            return None, self._jit_one(sky_coeffs, rot_m)

        _, vis = jax.lax.scan(step, None, rot_matrices)
        return vis  # (ntime, nfreq, nbls)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    """
    Compute equatorial → topocentric rotation matrices for each time step.

    Uses the same ERFA-based approach as matvis.

    Parameters
    ----------
    times : astropy.time.Time, shape (ntime,)
    location : astropy.coordinates.EarthLocation

    Returns
    -------
    rot_matrices : ndarray, shape (ntime, 3, 3), float32
    """
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
    """Extract the 3×3 equatorial→topocentric rotation matrix at time t."""
    obsf   = crds._get_obsf(t, crds.telescope_loc)
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
