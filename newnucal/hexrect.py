"""
Hex-rect coordinate utilities for the 2D NUFFT simulation path.

The 2D path exploits the fact that HERA baselines lie on a hexagonal integer
lattice.  By mapping HEALPix sky pixels to dual axial coordinates, each
frequency's sky-weighted integral becomes a 2D type-1 NUFFT onto a compact
uniform grid that is then sampled at the integer lattice positions.

Key relationships
-----------------
Let A_lat be the 2×2 physical lattice matrix whose columns are the two hex
lattice basis vectors in metres.  It satisfies:

    A_lat @ bl_grid[i]  ==  bls[i, :2]   for every baseline i

The dual axial sky coordinate of a topocentric direction s = (ell, m) is:

    xi = A_lat.T @ s    (shape (2,), units metres)

The type-1 NUFFT source position at frequency nu is then:

    x_j = 2π * nu / C * xi1_j
    y_j = 2π * nu / C * xi2_j

and the output at grid index (q + n_q//2, r + n_r//2) equals the visibility
at baseline bl_grid = (q, r) for that frequency.

The grid uses FINUFFT's shifted-centre convention: mode k is at array index
k + N//2, so integer axial coords q, r map to indices q + n_q//2, r + n_r//2.
"""

import numpy as np
from .utils import C


def critical_channel_spacing(bmax_m, field_radius=1.0, safety=1.0):
    """Minimum channel spacing (Hz) for the 2D hex-rect path to be alias-free.

    The type-1 NUFFT grid must be fine enough in frequency that the visibility
    fringes for the longest baseline at the field edge complete at most one
    cycle per channel.

    Parameters
    ----------
    bmax_m : float
        Longest baseline length in metres.
    field_radius : float
        Maximum sky radius in direction cosines (1.0 = full horizon).
    safety : float
        Additional safety factor (> 1 increases the required channel density).

    Returns
    -------
    float  Delta-nu in Hz:  C / (2 * safety * bmax_m * field_radius)
    """
    return C / (2.0 * safety * bmax_m * field_radius)


def hex_lattice_matrix(array):
    """Return the 2×2 physical lattice matrix A_lat (metres) for a HERAArray.

    Columns of A_lat are the two hex lattice basis vectors.  Satisfies:

        A_lat @ bl_grid[i]  ==  array.bls[i, :2]   for all baselines i

    Derived from the stored basis_matrix (which equals basis_matrix_raw / C):

        A_lat = -C * array.basis_matrix[:2, :2].T

    Parameters
    ----------
    array : HERAArray

    Returns
    -------
    A_lat : np.ndarray, shape (2, 2)
    """
    return -C * array.basis_matrix[:2, :2].T


def axial_grid_size(bl_grid):
    """Return the (n_q, n_r) uniform grid dimensions for the 2D NUFFT path.

    The grid is odd-sized and symmetric around zero so that the FINUFFT
    shifted-centre convention maps mode k to array index k + N//2, allowing
    integer axial baseline coordinates to index directly into the grid after
    adding the half-size offset.

    Parameters
    ----------
    bl_grid : array_like, shape (nbls, 2)
        Integer axial baseline coordinates (can be negative).

    Returns
    -------
    (n_q, n_r) : tuple[int, int]
        Both values are odd.
    """
    bl_grid = np.asarray(bl_grid)
    q_max = int(np.max(np.abs(bl_grid[:, 0])))
    r_max = int(np.max(np.abs(bl_grid[:, 1])))
    return 2 * q_max + 1, 2 * r_max + 1
