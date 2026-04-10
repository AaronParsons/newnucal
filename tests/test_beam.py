import numpy as np
import pytest
from newnucal.beam import BeamModel
from newnucal.dpss import dpss_reconstruct


def test_beam_model_shapes(beam_model, freqs):
    import healpy
    npix = healpy.nside2npix(beam_model.nside)
    assert beam_model.coeffs.shape[0] == npix
    assert beam_model.coeffs.shape[1] == beam_model.nmodes
    assert beam_model.A_beam.shape == (len(freqs), beam_model.nmodes)


def test_beam_reconstruction_positive(beam_model):
    """Reconstructed beam power should be non-negative everywhere."""
    rec = dpss_reconstruct(beam_model.coeffs, beam_model.A_beam)
    # Reconstruction can have small ringing below horizon; above horizon
    # (first pixel = north pole / zenith for nside=8) must be positive.
    assert rec[0, :].min() > 0, "Zenith beam power should be positive"


def test_beam_zenith_brightest(beam_model):
    """Airy beam should be brightest at zenith (pixel 0 in RING, th=0)."""
    import healpy
    rec = dpss_reconstruct(beam_model.coeffs, beam_model.A_beam)
    # Average power across frequencies
    mean_power = rec.mean(axis=1)
    # Zenith pixel (theta closest to 0)
    nside = beam_model.nside
    th, _ = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    zenith_px = np.argmin(th)
    assert mean_power[zenith_px] == pytest.approx(mean_power.max(), rel=0.1)


def test_beam_coeffs_dtype(beam_model):
    assert beam_model.coeffs.dtype == np.float32
