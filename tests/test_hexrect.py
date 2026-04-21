"""
Tests for newnucal.hexrect utilities.
"""

import numpy as np
import pytest

from newnucal.array import HERAArray
from newnucal.hexrect import (
    C,
    axial_grid_size,
    critical_channel_spacing,
    hex_lattice_matrix,
)


@pytest.fixture(scope="module")
def array():
    return HERAArray.from_hex(hexnum=2)


# ---------------------------------------------------------------------------
# critical_channel_spacing
# ---------------------------------------------------------------------------

def test_critical_channel_spacing_formula():
    bmax = 50.0
    dnu = critical_channel_spacing(bmax, field_radius=1.0, safety=1.0)
    assert abs(dnu - C / (2.0 * bmax)) < 1.0


def test_critical_channel_spacing_field_radius():
    bmax = 50.0
    dnu1 = critical_channel_spacing(bmax, field_radius=1.0)
    dnu_half = critical_channel_spacing(bmax, field_radius=0.5)
    assert abs(dnu_half / dnu1 - 2.0) < 1e-6


def test_critical_channel_spacing_safety():
    bmax = 50.0
    dnu1 = critical_channel_spacing(bmax, safety=1.0)
    dnu2 = critical_channel_spacing(bmax, safety=2.0)
    assert abs(dnu1 / dnu2 - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# hex_lattice_matrix
# ---------------------------------------------------------------------------

def test_hex_lattice_matrix_shape(array):
    A_lat = hex_lattice_matrix(array)
    assert A_lat.shape == (2, 2)
    assert A_lat.dtype == np.float32


def test_hex_lattice_matrix_recovers_baselines(array):
    """A_lat @ bl_grid[i] == bls[i, :2] for all baselines."""
    A_lat = hex_lattice_matrix(array)
    for i in range(array.nbls):
        pred = A_lat @ array.bl_grid[i].astype(np.float64)
        actual = array.bls[i, :2].astype(np.float64)
        assert np.allclose(pred, actual, atol=0.05), (
            f"baseline {i}: A_lat @ {array.bl_grid[i]} = {pred}, expected {actual}"
        )


def test_hex_lattice_matrix_zero_baseline(array):
    """The zero baseline (auto-correlation) maps to the origin."""
    A_lat = hex_lattice_matrix(array)
    zero_idx = np.where((array.bl_grid == 0).all(axis=1))[0]
    if len(zero_idx) > 0:
        pred = A_lat @ array.bl_grid[zero_idx[0]].astype(np.float64)
        assert np.allclose(pred, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# axial_grid_size
# ---------------------------------------------------------------------------

def test_axial_grid_size_symmetric(array):
    n_q, n_r = axial_grid_size(array.bl_grid)
    q_max = int(np.max(np.abs(array.bl_grid[:, 0])))
    r_max = int(np.max(np.abs(array.bl_grid[:, 1])))
    assert n_q == 2 * q_max + 1
    assert n_r == 2 * r_max + 1


def test_axial_grid_size_odd(array):
    """Grid dimensions must be odd for the shifted-centre FINUFFT convention."""
    n_q, n_r = axial_grid_size(array.bl_grid)
    assert n_q % 2 == 1
    assert n_r % 2 == 1


def test_axial_grid_size_explicit():
    bl = np.array([[-4, 0], [0, -2], [3, 1]])
    n_q, n_r = axial_grid_size(bl)
    assert n_q == 9   # 2*4+1
    assert n_r == 5   # 2*2+1


# ---------------------------------------------------------------------------
# Consistency: axial sky coordinate ↔ visibility phase
# ---------------------------------------------------------------------------

def test_axial_sky_coord_dot_product(array):
    """xi = A_lat.T @ s satisfies bl_grid[i] · xi == bls[i,:2] · s for all i."""
    A_lat = hex_lattice_matrix(array)
    rng = np.random.default_rng(0)
    s = rng.standard_normal(2)        # random sky direction cosines
    xi = A_lat.T @ s                  # (2,) axial sky coordinate

    for i in range(array.nbls):
        via_xi  = float(np.dot(xi, array.bl_grid[i]))
        via_bls = float(np.dot(s, array.bls[i, :2]))
        assert abs(via_xi - via_bls) < 1e-2, (
            f"baseline {i}: bl_grid·xi = {via_xi:.6f}, bls·s = {via_bls:.6f}"
        )
