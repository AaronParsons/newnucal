"""
DPSS (Discrete Prolate Spheroidal Sequence) utilities.

The DPSS basis provides a compact, spectrally-limited representation of
frequency-dependent quantities (sky flux, beam response, gains).  Each
physical quantity is stored as a vector of DPSS coefficients per spatial
location; the DPSS matrix A[nfreq, nmodes] reconstructs the full frequency
axis via  data[..., nfreq] = coeffs[..., nmodes] @ A.T .
"""

import numpy as np
from hera_filters.dspec import dpss_operator

DTYPE_R = np.float32

def dpss_matrix(freqs, eta_max, eigenval_cutoff=1e-9):
    """
    Build the DPSS basis matrix A of shape (nfreq, nmodes).

    Each column of A is a DPSS eigenvector (spectral window of half-width
    eta_max in delay space).  For unflagged data the columns are approximately
    orthonormal, so the fit matrix is just A itself (i.e. projection =
    data @ A  and  reconstruction = coeffs @ A.T).

    Parameters
    ----------
    freqs : array_like, shape (nfreq,)
        Frequency array in Hz.
    eta_max : float
        Maximum delay in seconds (half-width of the DPSS filter).
    eigenval_cutoff : float
        Drop eigenmodes whose eigenvalue is below this threshold.

    Returns
    -------
    A : ndarray, shape (nfreq, nmodes), dtype
    """
    A, _ = dpss_operator(
        np.asarray(freqs, dtype=DTYPE_R),
        filter_centers=[0.0],
        filter_half_widths=[eta_max],
        eigenval_cutoff=[eigenval_cutoff],
    )
    return A.real.astype(DTYPE_R)


def dpss_project(data, A):
    """
    Project data onto the DPSS basis.

    Parameters
    ----------
    data : array_like, shape (..., nfreq)
        Data to project.  Frequency is the *last* axis.
    A : array_like, shape (nfreq, nmodes)
        DPSS basis matrix from :func:`dpss_matrix`.

    Returns
    -------
    coeffs : ndarray, shape (..., nmodes)
        DPSS coefficients.  Reconstruct with :func:`dpss_reconstruct`.
    """
    data = np.asarray(data)
    A = np.asarray(A)
    return data @ A  # (..., nfreq) @ (nfreq, nmodes) → (..., nmodes)


def dpss_reconstruct(coeffs, A):
    """
    Reconstruct frequency-resolved data from DPSS coefficients.

    Parameters
    ----------
    coeffs : array_like, shape (..., nmodes)
    A : array_like, shape (nfreq, nmodes)

    Returns
    -------
    data : ndarray, shape (..., nfreq)
    """
    coeffs = np.asarray(coeffs)
    A = np.asarray(A)
    return coeffs @ A.T  # (..., nmodes) @ (nmodes, nfreq) → (..., nfreq)
