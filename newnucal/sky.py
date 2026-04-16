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


class SkyModel:
    """Sky model on an equatorial HEALPix grid with a spectral basis.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    freqs : array_like, (nfreq,) Hz
    basis : SkyBasis
        Spectral basis.  ``A_sky = basis.A`` has shape ``(nfreq, nmodes)``.
    """

    def __init__(self, nside: int, freqs, basis: SkyBasis):
        self.nside  = int(nside)
        self.freqs  = np.asarray(freqs, dtype=np.float64)
        self.basis  = basis

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def A_sky(self) -> np.ndarray:
        """Spectral basis matrix ``(nfreq, nmodes)`` float32."""
        return self.basis.A

    @property
    def nmodes(self) -> int:
        return int(self.basis.nmodes)

    @property
    def npix(self) -> int:
        return int(healpy.nside2npix(self.nside))

    @property
    def nfreq(self) -> int:
        return int(self.freqs.shape[0])
