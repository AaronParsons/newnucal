"""
Shared pytest fixtures for newnucal tests.

All fixtures use small sizes so tests run quickly:
  hexnum=2  →  7 antennas, ~20 unique baselines
  nfreq=16, nside=8 (768 pixels), ntimes=2
"""

import os
# Force FINUFFT to use a single OpenMP thread so that nufft3 calls are
# perfectly reproducible across separate invocations.  Without this,
# parallel float reductions produce different rounding depending on thread
# scheduling, causing test_resuming_preserves_anderson_state to fail
# non-deterministically.  Must be set before the first FINUFFT call.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from astropy.time import Time
from astropy.coordinates import EarthLocation

from newnucal.array import HERAArray
from newnucal.basis import dpss_matrix


HERA_LOC = EarthLocation.from_geodetic(
    lat=-30.72152612068926,
    lon=21.42830382686301,
    height=1051.69,
)

NFREQ = 16
FREQS = np.linspace(100e6, 200e6, NFREQ, endpoint=False)
NSIDE_SKY = 8
NSIDE_BEAM = 8
NTIMES = 2
ETA_MAX_SKY = 40e-9   # seconds
ETA_MAX_BEAM = 20e-9  # seconds


@pytest.fixture(scope="session")
def freqs():
    return FREQS.copy()


@pytest.fixture(scope="session")
def array():
    return HERAArray.from_hex(hexnum=2)


@pytest.fixture(scope="session")
def A_sky(freqs):
    return dpss_matrix(freqs, ETA_MAX_SKY)


@pytest.fixture(scope="session")
def A_beam(freqs):
    return dpss_matrix(freqs, ETA_MAX_BEAM)


@pytest.fixture(scope="session")
def sky_basis(freqs):
    from newnucal.basis import SkyBasis
    return SkyBasis.from_dpss(freqs, ETA_MAX_SKY)


@pytest.fixture(scope="session")
def beam_basis(freqs):
    from newnucal.basis import BeamBasis
    return BeamBasis.from_dpss(freqs, ETA_MAX_BEAM)


@pytest.fixture(scope="session")
def sky_model(freqs, sky_basis):
    from newnucal.sky import SkyModel
    return SkyModel(nside=NSIDE_SKY, freqs=freqs, basis=sky_basis)


@pytest.fixture(scope="function")
def beam_model(freqs, beam_basis):
    from newnucal.beam import BeamModel
    return BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=beam_basis)


@pytest.fixture(scope="session")
def rot_matrices():
    """Two rotation matrices: one near transit, one ~1 hr later."""
    from newnucal.simulate import compute_rotation_matrices
    t0 = Time("2018-08-31T04:02:30", format="isot", scale="utc")
    times = Time(
        np.linspace(t0.jd, t0.jd + 1 / 24, NTIMES),
        format="jd",
    )
    return compute_rotation_matrices(times, HERA_LOC)


@pytest.fixture(scope="function")
def forward_model(array, sky_model, beam_model, freqs, rot_matrices):
    from newnucal.simulate import ForwardModel
    fwd = ForwardModel(array, sky_model, beam_model, freqs)
    fwd.precompute_time_geometry(rot_matrices)
    return fwd
