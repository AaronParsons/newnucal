"""
SkyModel — HEALPix sky with a pluggable spectral basis.

Parallels :class:`~newnucal.beam.BeamModel`: stores a HEALPix resolution and
a :class:`~newnucal.basis.SkyBasis` whose ``A`` matrix is the sky spectral
basis used by :class:`~newnucal.simulate.ForwardModel` and
:class:`~newnucal.calibrator.Calibrator`.
"""

import numpy as np
import healpy

from .basis import SkyBasis

DTYPE_R = np.float32

class SkyModel:
    """Sky model on an equatorial HEALPix grid with a spectral basis.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    freqs : array_like, (nfreq,) Hz
    basis : SkyBasis or array_like (nfreq, nmodes)
        Spectral basis.  Accepted forms:
        - :class:`~newnucal.basis.SkyBasis` — ``basis.A`` is used as
          ``A``.
        - ``ndarray`` of shape ``(nfreq, nmodes)`` — used directly.
    """

    def __init__(self, nside: int, freqs, basis):
        self.nside  = int(nside)
        self.freqs  = np.asarray(freqs, dtype=DTYPE_R)
        # Resolve spectral basis: BeamBasis has an .A attribute; ndarray does not.
        if hasattr(basis, 'A'):
            self.A = np.asarray(basis.A, dtype=DTYPE_R)   # (nfreq, nmodes)
        else:
            self.A = np.asarray(basis, dtype=DTYPE_R)     # (nfreq, nmodes)

    def project(self, data):
        """Project data (npix, nfreq) onto internal basis. Return coefficients."""
        return data @ self.A

    def deproject(self, coeffs):
        """De-project coeffs (npix, coeffs) back to frequency basis."""
        return coeffs @ self.A.T

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nmodes(self) -> int:
        return self.A.shape[1]

    @property
    def npix(self) -> int:
        return int(healpy.nside2npix(self.nside))

    @property
    def nfreq(self) -> int:
        return int(self.freqs.shape[0])

    @property
    def eq_vec(self) -> np.ndarray:
        vec = healpy.pix2vec(self.nside, np.arange(self.npix))
        return np.array(vec, dtype=DTYPE_R)
