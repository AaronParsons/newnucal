"""
Beam model on a topocentric HEALPix grid.

The beam is represented by spectral coefficients stored at each HEALPix
pixel.  Pixels are in *topocentric* coordinates (the beam is fixed on
the sky as the Earth rotates).

At a given topocentric direction (th, ph) and frequency ν:

    B(th, ph, ν) = interp(beam_coeffs, th, ph) @ A[ν, :]

where interp uses bilinear HEALPix interpolation (4 neighbours).

The spectral basis ``A`` (shape ``(nfreq, nmodes)``) is supplied as a
:class:`~newnucal.basis.BeamBasis` or a plain ``(nfreq, nmodes)`` ndarray.

Initialisation
--------------
The default starting point is a circularly-symmetric analytic beam
(e.g. AiryBeam) evaluated on the HEALPix grid.  For each pixel the
frequency spectrum is projected onto the chosen basis.
"""

import numpy as np
import healpy
from pyuvdata.analytic_beam import AiryBeam

from .basis import BeamBasis

DTYPE_R = np.float32


class BeamModel:
    """
    HEALPix beam model in topocentric coordinates with a pluggable spectral basis.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    freqs : array_like, shape (nfreq,)
        Frequency array in Hz.
    basis : BeamBasis or array_like (nfreq, nmodes)
        Spectral basis for the beam.  Accepted forms:

        - :class:`~newnucal.basis.BeamBasis` — stored directly.
        - ``ndarray`` of shape ``(nfreq, nmodes)`` — wrapped in a
          :class:`~newnucal.basis.BeamBasis`.
    beam : pyuvdata AnalyticBeam, optional
        Analytic beam used to initialise the pixel coefficients.
        Defaults to an AiryBeam with diameter=14.6 m (HERA element).
    ch_chunk : int
        Number of frequency channels to evaluate at once when sampling
        the analytic beam (avoids large temporary arrays).
    """

    def __init__(
        self,
        nside: int,
        freqs,
        basis,
        beam=None,
        ch_chunk: int = 32,
    ):
        self.nside = nside
        self.freqs = np.asarray(freqs, dtype=np.float64)

        if beam is None:
            beam = AiryBeam(diameter=14.6, include_cross_pols=False)
        self.beam = beam

        # Normalise to a BeamBasis so the canonical array lives in one place.
        if isinstance(basis, BeamBasis):
            self.basis = basis
        else:
            self.basis = BeamBasis(A=np.asarray(basis, dtype=DTYPE_R))

        # Pixel centres in topocentric coordinates
        npix = healpy.nside2npix(nside)
        th, ph = healpy.pix2ang(nside, np.arange(npix))
        az = ph.astype(np.float64)
        za = th.astype(np.float64)

        # Evaluate beam power at each pixel for all frequencies
        beam_power = self._eval_beam(beam, az, za, ch_chunk)  # (npix, nfreq)

        # Project onto spectral basis
        self.coeffs = self.project(beam_power).astype(DTYPE_R)
        # (npix, nmodes)

    # ------------------------------------------------------------------

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

    def _eval_beam(self, beam, az, za, ch_chunk):
        """Evaluate beam power on the pixel grid across all frequencies."""
        nfreq = len(self.freqs)
        npix = len(az)
        beam_power = np.empty((npix, nfreq), dtype=DTYPE_R)

        for ch in range(0, nfreq, ch_chunk):
            ch_sl = slice(ch, ch + ch_chunk)
            freqs_chunk = self.freqs[ch_sl]
            resp = beam.power_eval(
                az_array=az,
                za_array=za,
                freq_array=freqs_chunk,
            )[0, 0]  # → (nfreq_chunk, npix)
            beam_power[:, ch_sl] = resp.T.astype(DTYPE_R)

        return beam_power  # (npix, nfreq)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npix(self) -> int:
        return healpy.nside2npix(self.nside)

    @property
    def nfreq(self) -> int:
        return len(self.freqs)

    @property
    def nmodes(self) -> int:
        return self.A.shape[1]
