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
from .utils import DTYPE_R_NPY as DTYPE_R

class SkyModel:
    """Sky model on an equatorial HEALPix grid with a spectral basis.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    freqs : array_like, (nfreq,) Hz
    basis : SkyBasis or array_like (nfreq, nmodes)
        Spectral basis.  Accepted forms:
        - :class:`~newnucal.basis.SkyBasis` — stored directly.
        - ``ndarray`` of shape ``(nfreq, nmodes)`` — wrapped in a
          :class:`~newnucal.basis.SkyBasis`.
    """

    def __init__(self, nside: int, freqs, basis):
        self.nside = int(nside)
        self.freqs = np.asarray(freqs, dtype=DTYPE_R)
        if isinstance(basis, SkyBasis):
            self.basis = basis
        else:
            self.basis = SkyBasis(A=np.asarray(basis, dtype=DTYPE_R))

    @property
    def A(self):
        """Spectral basis matrix (nfreq, nmodes)."""
        return self.basis.A

    def project(self, data):
        """Project data (npix, nfreq) onto internal basis. Return coefficients."""
        return self.basis.project(data)

    def deproject(self, coeffs):
        """De-project coeffs (npix, nmodes) back to frequency basis."""
        return self.basis.deproject(coeffs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nmodes(self) -> int:
        return self.basis.nmodes

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
