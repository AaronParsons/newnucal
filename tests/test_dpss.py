import numpy as np
import pytest
from newnucal.dpss import dpss_matrix, dpss_project, dpss_reconstruct


def test_dpss_matrix_shape(freqs):
    A = dpss_matrix(freqs, eta_max=40e-9)
    assert A.ndim == 2
    assert A.shape[0] == len(freqs)
    assert A.shape[1] >= 1
    assert A.dtype == np.float32


def test_dpss_matrix_nmodes_increases_with_bandwidth(freqs):
    # NW = nfreq * channel_spacing * eta_max must be < nfreq/2.
    # With 16 channels over 100 MHz: channel_spacing = 6.25 MHz.
    # So eta_max < 8 / (16 * 6.25e6) ≈ 80 ns.
    A_narrow = dpss_matrix(freqs, eta_max=10e-9)
    A_wide = dpss_matrix(freqs, eta_max=60e-9)
    assert A_wide.shape[1] >= A_narrow.shape[1]


def test_roundtrip_smooth_function(freqs):
    """A smooth (band-limited) function should reconstruct accurately."""
    A = dpss_matrix(freqs, eta_max=40e-9)
    # Power-law spectrum — well within the DPSS band
    f0 = freqs.mean()
    data = (freqs / f0) ** -2.5
    data = data.reshape(1, -1)  # (1, nfreq)

    coeffs = dpss_project(data, A)       # (1, nmodes)
    rec = dpss_reconstruct(coeffs, A)    # (1, nfreq)

    rms_err = np.sqrt(np.mean((rec - data) ** 2)) / np.std(data)
    assert rms_err < 0.01, f"relative RMS reconstruction error {rms_err:.3e} > 1%"


def test_roundtrip_multi_pixel(freqs):
    """Batch dimension (pixels) passes through correctly."""
    A = dpss_matrix(freqs, eta_max=40e-9)
    npix = 50
    rng = np.random.default_rng(0)
    # Smooth random spectra (sum of a few sinusoids within band)
    nmodes = A.shape[1]
    raw_coeffs = rng.normal(size=(npix, nmodes)).astype(np.float32)
    data = dpss_reconstruct(raw_coeffs, A)   # (npix, nfreq) — in-band by construction

    coeffs = dpss_project(data, A)
    rec = dpss_reconstruct(coeffs, A)

    rms_err = np.sqrt(np.mean((rec - data) ** 2)) / (np.std(data) + 1e-30)
    assert rms_err < 0.01
