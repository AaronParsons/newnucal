import numpy as np
import pytest
from newnucal.array import HERAArray


def test_from_hex_builds(array):
    assert array.nants == 5   # hexnum=2 hex has 5 antennas
    assert array.nbls > 0


def test_bl_grid_shape(array):
    assert array.bl_grid.shape == (array.nbls, 2)
    assert array.bl_grid.dtype in (np.int32, np.int64, int)


def test_n_modes_odd(array):
    # n_modes = 2*max + 1 is always odd
    assert array.n_modes % 2 == 1


def test_n_modes_covers_all_baselines(array):
    max_abs = np.max(np.abs(array.bl_grid))
    assert array.n_modes >= 2 * max_abs + 1


def test_basis_matrix_shape(array):
    assert array.basis_matrix.shape == (3, 3)


def test_bls_shape(array):
    assert array.bls.shape == (array.nbls, 3)


def test_basis_matrix_zenith_gives_zero_nufft_arg():
    """
    A source at zenith has topocentric unit vector (0, 0, 1).
    basis_matrix.T @ [0, 0, 1] gives the NUFFT argument direction.
    The x and y components should be ~0 (no fringe for a zenith source).
    """
    arr = HERAArray.from_hex(hexnum=2, sep=14.6)
    zenith = np.array([0.0, 0.0, 1.0])
    grid_coords = arr.basis_matrix.T @ zenith  # (3,)
    # x and y components of zenith in grid coords should be near zero
    assert abs(grid_coords[0]) < 1e-10, f"zenith x grid coord not zero: {grid_coords[0]}"
    assert abs(grid_coords[1]) < 1e-10, f"zenith y grid coord not zero: {grid_coords[1]}"
