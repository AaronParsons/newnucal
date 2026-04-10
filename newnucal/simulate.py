"""
Forward visibility simulation.

ForwardModel maps sky DPSS coefficients → model visibilities via:

  1. Rotate equatorial sky-pixel unit vectors to topocentric frame (R(t) @ eq)
  2. Apply horizon mask  (topo_z > 0)
  3. Interpolate beam DPSS coefficients at rotated sky-pixel positions
     using bilinear HEALPix interpolation (healjax)
  4. Reconstruct per-pixel sky and beam spectra via DPSS @ A.T
  5. jax_finufft.nufft1 batched over frequencies:
       x[freq, px] = 2π * (basis_matrix @ topo)[0, px] * freq
     produces vis_grid[freq, n_modes, n_modes]
  6. Extract baselines at their integer grid positions bl_grid[:, 0/1]

Time loop uses jax.lax.scan over pre-computed rotation matrices.
"""

import functools

import numpy as np
import healpy
import jax
import jax.numpy as jnp
from jax_finufft import nufft1
from healjax import get_interp_weights
import healjax

from .array import HERAArray
from .beam import BeamModel


class ForwardModel:
    """
    JAX forward visibility simulation.

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

        freqs = np.asarray(freqs, dtype=np.float32)
        self.freqs = jnp.array(freqs)

        # Pre-computed static quantities (captured by JIT)
        self.A_sky = jnp.array(beam_model.A_beam)   # sky DPSS set separately below
        # NOTE: A_sky will be replaced by the sky's own DPSS matrix when
        # set_sky_dpss() is called.  We initialise to beam's A here as a
        # placeholder so the object is valid immediately.

        self.A_beam = jnp.array(beam_model.A_beam)
        self.beam_coeffs = jnp.array(beam_model.coeffs)  # (npix_beam, nmodes_beam)

        # Equatorial unit vectors for all sky pixels
        npix_sky = healpy.nside2npix(sky_nside)
        eq_xyz = np.array(healpy.pix2vec(sky_nside, np.arange(npix_sky)), dtype=np.float32)
        self.eq_coords = jnp.array(eq_xyz)  # (3, npix_sky)

        self.basis_matrix = jnp.array(array.basis_matrix, dtype=jnp.float32)  # (3, 3)
        self.bl_grid = jnp.array(array.bl_grid, dtype=int)  # (nbls, 2)
        self.n_modes = array.n_modes

        # JIT-compile the single-time simulator; nside and n_modes are
        # captured as Python literals in the closure so they are static.
        self._jit_one = jax.jit(self._simulate_one)

    # ------------------------------------------------------------------
    # Sky DPSS matrix setter  (called by Calibrator when sky model is set)
    # ------------------------------------------------------------------

    def set_sky_dpss(self, A_sky):
        """Set the sky DPSS matrix.  A_sky shape: (nfreq, nmodes_sky)."""
        self.A_sky = jnp.array(A_sky, dtype=jnp.float32)
        # Recompile JIT with the new matrix captured.
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
        topo = jnp.dot(rot_m, self.eq_coords)   # (3, npix_sky)

        # 2. Transform to antenna-grid coordinates (basis_matrix already /c)
        topo_grid = jnp.dot(self.basis_matrix.T, topo)  # (3, npix_sky)

        # 3. Horizon mask — multiply flux, not filter (keeps shapes static)
        horizon = (topo_grid[2] > 0).astype(jnp.float32)  # (npix_sky,)

        # 4. Interpolate beam DPSS coefficients at topocentric sky positions
        topo_th, topo_ph = healjax.vec2ang(topo[0], topo[1], topo[2])
        px, wgts = get_interp_weights(topo_th, topo_ph, self.beam_nside)
        # px: (4, npix_sky), wgts: (4, npix_sky)
        interp_beam_coeffs = jnp.sum(
            wgts[:, :, None] * self.beam_coeffs[px], axis=0
        )  # (npix_sky, nmodes_beam)

        # 5. Reconstruct spectra  →  (npix_sky, nfreq)
        sky_spec = sky_coeffs @ self.A_sky.T        # (npix_sky, nfreq)
        beam_spec = interp_beam_coeffs @ self.A_beam.T  # (npix_sky, nfreq)

        # 6. Weighted flux with horizon mask  →  (nfreq, npix_sky), complex64
        flux = ((sky_spec * beam_spec) * horizon[:, None]).T.astype(jnp.complex64)

        # 7. Frequency-dependent NUFFT non-uniform point coordinates
        #    x[freq, px] = 2π * topo_grid[0, px] * freq
        x_nu = (2 * jnp.pi * topo_grid[0][None, :] * self.freqs[:, None]).astype(jnp.float32)
        y_nu = (2 * jnp.pi * topo_grid[1][None, :] * self.freqs[:, None]).astype(jnp.float32)

        # 8. Type-1 NUFFT: (nfreq, npix) → (nfreq, n_modes, n_modes)
        vis_grid = nufft1(
            (self.n_modes, self.n_modes),
            flux,
            x_nu,
            y_nu,
            eps=self.eps,
        )

        # 9. Extract at integer baseline grid positions (Python negative
        #    indexing gives the correct NUFFT mode for negative baselines)
        vis = vis_grid[:, self.bl_grid[:, 0], self.bl_grid[:, 1]]  # (nfreq, nbls)
        return vis

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

    # CoordinateRotationERFA needs at least one source to initialise;
    # we feed a dummy zenith source (as a length-1 array) and extract
    # only the rotation matrix.
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
