"""Visibility convention: newnucal must match matvis (and so HERA data)
without conjugation.

For an antenna pair (i, j), bls = x_j − x_i and
V_ij = Σ I B exp(+2πi ν bls·ŝ / c).  The reference is matvis, the simulator
HERA data products are checked against; it is not a formula typed into this
test.  Before the 2026-10-08 change newnucal returned the conjugate, and this
test failed with a phase error of ~2 (see memos/memo-001-coupling-identifiability).
"""

import numpy as np
import pytest
import jax.numpy as jnp
from astropy.time import Time

from newnucal import HERAArray, BeamModel, SkyModel, ForwardModel
from newnucal.realdata import hera_array_from_file_pairs, snap_to_hex_lattice
from newnucal.simulate import compute_rotation_matrices

from .conftest import HERA_LOC

JD = 2458362.67
FREQS = np.array([150e6, 151e6])


@pytest.fixture(scope="module")
def setup():
    arr = HERAArray.from_hex(2)
    rot = compute_rotation_matrices(Time([JD], format="jd"), HERA_LOC)
    sky = SkyModel(16, FREQS, np.eye(FREQS.size))
    beam = BeamModel(16, FREQS, np.eye(FREQS.size))
    beam.coeffs = np.ones_like(beam.coeffs)             # flat beam: no interpolation error
    eq = np.asarray(sky.eq_vec)
    topo = rot[0] @ eq
    # an off-zenith pixel (so every baseline has a non-trivial phase)
    p = int(np.argmax(np.where(topo[2] > 0.7, topo[0] + 0.3 * topo[1], -9.0)))
    return arr, rot, sky, beam, eq, p


@pytest.fixture(scope="module")
def matvis_vis(setup):
    from matvis import simulate_vis
    from pyuvdata.analytic_beam import UniformBeam
    arr, rot, sky, beam, eq, p = setup
    ra = np.arctan2(eq[1, p], eq[0, p]) % (2 * np.pi)
    dec = np.arcsin(eq[2, p])
    mv = simulate_vis(ants={k: np.asarray(v) for k, v in arr.ants.items()},
                      fluxes=np.ones((1, FREQS.size)), ra=np.array([ra]),
                      dec=np.array([dec]), freqs=FREQS, times=Time([JD], format="jd"),
                      beams=[UniformBeam()], telescope_loc=HERA_LOC, precision=2,
                      antpairs=np.asarray(arr.antpairs, dtype=int))
    return np.asarray(mv)[:, 0, :]                         # (nfreq, nbls)


@pytest.mark.parametrize("method", ["2d", "3d"])
def test_matches_matvis_without_conjugation(setup, matvis_vis, method):
    arr, rot, sky, beam, eq, p = setup
    fm = ForwardModel(arr, sky, beam, FREQS, method=method)
    T = np.zeros((eq.shape[1], FREQS.size))
    T[p] = 1.0
    V = np.asarray(fm.simulate(jnp.asarray(T), jnp.asarray(rot)))[0]   # (nfreq, nbls)
    ph_nn = V / np.abs(V)
    ph_mv = matvis_vis / np.abs(matvis_vis)
    # Residual phase difference (~0.04) is a sub-arcminute difference in how
    # the two compute the source direction; conjugation would give ~2.
    assert np.max(np.abs(ph_nn - ph_mv)) < 0.1
    assert np.max(np.abs(ph_nn - np.conj(ph_mv))) > 1.0


def test_bls_orientation_is_xj_minus_xi(setup):
    arr = setup[0]
    for k, (i, j) in enumerate(arr.antpairs):
        np.testing.assert_allclose(arr.bls[k], arr.ants[j] - arr.ants[i])


def test_file_pairs_orientation_is_xj_minus_xi():
    ants = HERAArray.from_hex(2).ants
    snapped, _ = snap_to_hex_lattice(ants)
    pairs = np.array([[0, 1], [2, 0], [3, 4]])     # from_hex(2) has antennas 0–4
    arr = hera_array_from_file_pairs(snapped, pairs)
    for k, (i, j) in enumerate(pairs):
        np.testing.assert_allclose(arr.bls[k], snapped[j] - snapped[i])
