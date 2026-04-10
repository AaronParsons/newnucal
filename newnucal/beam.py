"""
Beam model on a topocentric HEALPix grid.

The beam is represented by DPSS spectral coefficients stored at each
HEALPix pixel.  Pixels are in *topocentric* coordinates (the beam is
fixed on the sky as the Earth rotates).

At a given topocentric direction (th, ph) and frequency ν:

    B(th, ph, ν) = interp(beam_coeffs, th, ph) @ A_beam[ν, :]

where interp uses bilinear HEALPix interpolation (4 neighbours).

Initialisation
--------------
The default starting point is a circularly-symmetric analytic beam
(e.g. AiryBeam) evaluated on the HEALPix grid.  For each pixel the
frequency spectrum is projected onto DPSS modes.
"""

import numpy as np
import healpy
from pyuvdata.analytic_beam import AiryBeam

from .dpss import dpss_matrix, dpss_project


class BeamModel:
    """
    HEALPix DPSS beam model in topocentric coordinates.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    freqs : array_like, shape (nfreq,)
        Frequency array in Hz.
    eta_max : float
        DPSS delay half-width in seconds.
    beam : pyuvdata AnalyticBeam, optional
        Analytic beam used to initialise the DPSS coefficients.
        Defaults to an AiryBeam with diameter=14.6 m (HERA element).
    eigenval_cutoff : float
        Eigenvalue cutoff for DPSS mode selection.
    ch_chunk : int
        Number of frequency channels to evaluate at once when sampling
        the analytic beam (avoids large temporary arrays).
    """

    def __init__(
        self,
        nside: int,
        freqs,
        eta_max: float,
        beam=None,
        eigenval_cutoff: float = 1e-9,
        ch_chunk: int = 32,
    ):
        self.nside = nside
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.eta_max = eta_max

        if beam is None:
            beam = AiryBeam(diameter=14.6, include_cross_pols=False)
        self.beam = beam

        # DPSS basis
        self.A_beam = dpss_matrix(self.freqs, eta_max, eigenval_cutoff)
        # (nfreq, nmodes)

        # Pixel centres in topocentric coordinates
        npix = healpy.nside2npix(nside)
        th, ph = healpy.pix2ang(nside, np.arange(npix))
        # AiryBeam uses az = ph (azimuth) and za = th (zenith angle)
        az = ph.astype(np.float64)
        za = th.astype(np.float64)

        # Evaluate beam power at each pixel for all frequencies
        beam_power = self._eval_beam(beam, az, za, ch_chunk)  # (npix, nfreq)

        # Project onto DPSS basis: coeffs[px, mode] = sum_ν beam_power[px, ν] * A[ν, mode]
        self.coeffs = dpss_project(beam_power, self.A_beam).astype(np.float32)
        # (npix, nmodes)

    # ------------------------------------------------------------------

    def _eval_beam(self, beam, az, za, ch_chunk):
        """Evaluate beam power on the pixel grid across all frequencies."""
        nfreq = len(self.freqs)
        npix = len(az)
        beam_power = np.empty((npix, nfreq), dtype=np.float32)

        for ch in range(0, nfreq, ch_chunk):
            ch_sl = slice(ch, ch + ch_chunk)
            freqs_chunk = self.freqs[ch_sl]
            # power_eval returns an array of shape (1, Naxes_vec, nfreq_chunk, npix)
            resp = beam.power_eval(
                az_array=az,
                za_array=za,
                freq_array=freqs_chunk,
            )[0, 0]  # → (nfreq_chunk, npix)
            beam_power[:, ch_sl] = resp.T.astype(np.float32)

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
        return self.A_beam.shape[1]
