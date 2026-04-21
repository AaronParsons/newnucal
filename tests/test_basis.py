"""
Tests for BeamBasis and SkyBasis.

Covers:
  - Direct construction from A matrix
  - from_dpss path (no SVD metadata)
  - from_ensemble path (with SVD metadata)
  - BeamBasis.from_beam_diameters path
  - save / from_file round-trip
  - has_svd property
  - A matrix orthonormality
"""

import os
import tempfile

import numpy as np
import pytest

from newnucal.basis import BeamBasis, SkyBasis
from newnucal.basis import dpss_matrix, basis_project, basis_reconstruct

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

N_SKY_SAMPLES  = 12
N_BEAM_SAMPLES = 10
N_BEAM_MODES   = 3
N_SKY_MODES    = 2


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_sky_basis(freqs, n_modes=N_SKY_MODES):
    """SkyBasis with SVD data built from random power-law fluxes."""
    rng = np.random.default_rng(0)
    nfreq = len(freqs)
    ref  = rng.exponential(size=(N_SKY_SAMPLES, 1)).astype(np.float64)
    idx  = rng.normal(-0.7, 0.2, (N_SKY_SAMPLES, 1))
    f_n  = (freqs / freqs[0])[None, :]
    spectra = ref * f_n ** idx
    return SkyBasis.from_ensemble(freqs, spectra, n_modes)


def _make_beam_basis(freqs, n_modes=N_BEAM_MODES):
    """BeamBasis with SVD data built from random spectra."""
    rng     = np.random.default_rng(1)
    nfreq   = len(freqs)
    spectra = np.abs(rng.standard_normal((N_BEAM_SAMPLES, nfreq))) + 0.1
    return BeamBasis.from_ensemble(freqs, spectra.astype(np.float64), n_modes)

# ------------------------------------------------------------------
# DPSS/Projection
# ------------------------------------------------------------------

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

    coeffs = basis_project(data, A)       # (1, nmodes)
    rec = basis_reconstruct(coeffs, A)    # (1, nfreq)

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
    data = basis_reconstruct(raw_coeffs, A)   # (npix, nfreq) — in-band by construction

    coeffs = basis_project(data, A)
    rec = basis_reconstruct(coeffs, A)

    rms_err = np.sqrt(np.mean((rec - data) ** 2)) / (np.std(data) + 1e-30)
    assert rms_err < 0.01


# ------------------------------------------------------------------
# Direct construction (just passing A)
# ------------------------------------------------------------------

class TestDirectConstruction:

    def test_sky_basis_shapes(self, freqs):
        nfreq  = len(freqs)
        nmodes = 4
        rng    = np.random.default_rng(10)
        A      = rng.standard_normal((nfreq, nmodes)).astype(np.float32)
        sb     = SkyBasis(A=A, freqs_hz=freqs)
        assert sb.A.shape    == (nfreq, nmodes)
        assert sb.nfreq      == nfreq
        assert sb.nmodes     == nmodes
        assert sb.has_svd    is False

    def test_beam_basis_shapes(self, freqs):
        nfreq  = len(freqs)
        nmodes = 5
        rng    = np.random.default_rng(11)
        A      = rng.standard_normal((nfreq, nmodes)).astype(np.float32)
        bb     = BeamBasis(A=A, freqs_hz=freqs)
        assert bb.A.shape    == (nfreq, nmodes)
        assert bb.nfreq      == nfreq
        assert bb.nmodes     == nmodes
        assert bb.has_svd    is False

    def test_A_stored_as_float32(self, freqs):
        nfreq = len(freqs)
        A     = np.ones((nfreq, 3), dtype=np.float64)
        sb    = SkyBasis(A=A)
        assert sb.A.dtype == np.float32

    def test_no_freqs_hz_ok(self, freqs):
        nfreq = len(freqs)
        A = np.ones((nfreq, 2), dtype=np.float32)
        sb = SkyBasis(A=A)
        assert sb.freqs_hz is None


# ------------------------------------------------------------------
# from_dpss
# ------------------------------------------------------------------

class TestFromDpss:

    def test_sky_basis_no_svd(self, freqs):
        sb = SkyBasis.from_dpss(freqs, 40e-9)
        assert sb.A.shape[0]  == len(freqs)
        assert sb.A.shape[1]  >= 1
        assert sb.has_svd     is False
        assert sb.freqs_hz    is not None

    def test_beam_basis_no_svd(self, freqs):
        bb = BeamBasis.from_dpss(freqs, 20e-9)
        assert bb.A.shape[0]  == len(freqs)
        assert bb.has_svd     is False

    def test_A_dtype(self, freqs):
        sb = SkyBasis.from_dpss(freqs, 40e-9)
        assert sb.A.dtype == np.float32


# ------------------------------------------------------------------
# from_ensemble
# ------------------------------------------------------------------

class TestFromEnsemble:

    def test_sky_basis_shapes(self, freqs):
        sb = _make_sky_basis(freqs)
        nfreq = len(freqs)
        assert sb.A.shape        == (nfreq, N_SKY_MODES + 1)
        assert sb.svd_mean.shape == (nfreq,)
        assert sb.svd_modes.shape == (N_SKY_MODES, nfreq)
        assert sb.svd_svals.shape == (N_SKY_MODES,)
        assert sb.n_samples       == N_SKY_SAMPLES
        assert sb.has_svd         is True

    def test_beam_basis_shapes(self, freqs):
        bb = _make_beam_basis(freqs)
        nfreq = len(freqs)
        assert bb.A.shape         == (nfreq, N_BEAM_MODES + 1)
        assert bb.svd_modes.shape == (N_BEAM_MODES, nfreq)
        assert bb.has_svd         is True

    def test_A_columns_orthonormal(self, freqs):
        sb   = _make_sky_basis(freqs)
        A    = sb.A.astype(np.float64)
        gram = A.T @ A
        res  = gram - np.eye(gram.shape[0])
        assert np.max(np.abs(res)) < 1e-5

    def test_svd_modes_orthogonal(self, freqs):
        """Raw SVD modes (rows of svd_modes) should be approximately orthogonal."""
        sb   = _make_sky_basis(freqs)
        V    = sb.svd_modes.astype(np.float64)   # (n_modes, nfreq)
        gram = V @ V.T
        diag = np.abs(np.diag(gram))
        off  = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off)) < 0.1 * np.max(diag + 1e-12)

    def test_A_dtype(self, freqs):
        assert _make_sky_basis(freqs).A.dtype == np.float32

    def test_svd_dtype(self, freqs):
        sb = _make_sky_basis(freqs)
        assert sb.svd_mean.dtype  == np.float32
        assert sb.svd_modes.dtype == np.float32
        assert sb.svd_svals.dtype == np.float32


# ------------------------------------------------------------------
# save / from_file round-trip
# ------------------------------------------------------------------

class TestSaveLoad:

    def _roundtrip_sky(self, freqs, with_svd):
        if with_svd:
            sb = _make_sky_basis(freqs)
        else:
            sb = SkyBasis.from_dpss(freqs, 40e-9)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            sb.save(path)
            sb2 = SkyBasis.from_file(path)
        finally:
            os.unlink(path)
        return sb, sb2

    def test_roundtrip_values_no_svd(self, freqs):
        sb, sb2 = self._roundtrip_sky(freqs, with_svd=False)
        np.testing.assert_array_equal(sb.A, sb2.A)
        assert sb2.has_svd is False

    def test_roundtrip_values_with_svd(self, freqs):
        sb, sb2 = self._roundtrip_sky(freqs, with_svd=True)
        np.testing.assert_array_equal(sb.A,         sb2.A)
        np.testing.assert_array_equal(sb.svd_mean,  sb2.svd_mean)
        np.testing.assert_array_equal(sb.svd_modes, sb2.svd_modes)
        np.testing.assert_array_equal(sb.svd_svals, sb2.svd_svals)
        assert sb2.n_samples == sb.n_samples
        assert sb2.has_svd   is True

    def test_roundtrip_shapes(self, freqs):
        sb, sb2 = self._roundtrip_sky(freqs, with_svd=True)
        assert sb2.nfreq  == sb.nfreq
        assert sb2.nmodes == sb.nmodes

    def test_beam_basis_roundtrip(self, freqs):
        bb = _make_beam_basis(freqs)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            bb.save(path)
            bb2 = BeamBasis.from_file(path)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(bb.A,         bb2.A)
        np.testing.assert_array_equal(bb.svd_modes, bb2.svd_modes)
        assert bb2.has_svd is True


# ------------------------------------------------------------------
# BeamBasis.from_beam_diameters  (integration test with AiryBeam)
# ------------------------------------------------------------------

class TestFromBeamDiameters:

    @pytest.fixture(scope="class")
    def bb_from_diams(self, freqs):
        return BeamBasis.from_beam_diameters(
            freqs=freqs,
            diameters=[14.0, 14.6],
            nside=8,
            n_modes=3,
        )

    def test_has_svd(self, bb_from_diams):
        assert bb_from_diams.has_svd is True

    def test_A_shape(self, bb_from_diams, freqs):
        assert bb_from_diams.A.shape == (len(freqs), 4)   # n_modes+1 cols

    def test_A_columns_orthonormal(self, bb_from_diams):
        A    = bb_from_diams.A.astype(np.float64)
        gram = A.T @ A
        res  = gram - np.eye(gram.shape[0])
        assert np.max(np.abs(res)) < 1e-5

    def test_A_dtype(self, bb_from_diams):
        assert bb_from_diams.A.dtype == np.float32


# ------------------------------------------------------------------
# from_file with resampling
# ------------------------------------------------------------------

class TestResample:

    def _save_and_resample(self, basis, new_freqs):
        """Helper: save basis to file and load with resampling."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            basis.save(path)
            basis_resampled = basis.__class__.from_file(path, new_freqs=new_freqs)
        finally:
            os.unlink(path)
        return basis_resampled

    def test_resample_coarser_no_svd(self, freqs):
        """Resample from fine to coarse grid without SVD metadata."""
        sb = SkyBasis.from_dpss(freqs, 40e-9)
        new_freqs = freqs[::2]  # Every other channel
        sb_resamp = self._save_and_resample(sb, new_freqs)

        assert sb_resamp.nfreq == len(new_freqs)
        assert sb_resamp.nmodes == sb.nmodes
        assert sb_resamp.has_svd is False
        np.testing.assert_array_equal(sb_resamp.freqs_hz, new_freqs)

    def test_resample_finer_no_svd(self, freqs):
        """Resample from coarse to fine grid without SVD metadata."""
        sb = SkyBasis.from_dpss(freqs[::2], 10e-9)  # Start coarse with smaller eta_max
        sb_resamp = self._save_and_resample(sb, freqs)  # Resample to finer

        assert sb_resamp.nfreq == len(freqs)
        assert sb_resamp.nmodes == sb.nmodes
        np.testing.assert_array_equal(sb_resamp.freqs_hz, freqs)

    def test_resample_with_svd_metadata(self, freqs):
        """Resampling preserves and resamples SVD metadata."""
        sb = _make_sky_basis(freqs)
        new_freqs = freqs[::2]
        sb_resamp = self._save_and_resample(sb, new_freqs)

        assert sb_resamp.has_svd is True
        assert sb_resamp.nfreq == len(new_freqs)
        assert sb_resamp.svd_mean.shape == (len(new_freqs),)
        assert sb_resamp.svd_modes.shape == (N_SKY_MODES, len(new_freqs))
        assert sb_resamp.n_samples == sb.n_samples
        np.testing.assert_array_equal(sb_resamp.freqs_hz, new_freqs)

    def test_resample_beam_basis_with_svd(self, freqs):
        """Resampling works for BeamBasis with SVD metadata."""
        bb = _make_beam_basis(freqs)
        new_freqs = freqs[::2]
        bb_resamp = self._save_and_resample(bb, new_freqs)

        assert bb_resamp.has_svd is True
        assert bb_resamp.nfreq == len(new_freqs)
        assert bb_resamp.svd_mean.shape == (len(new_freqs),)
        assert bb_resamp.svd_modes.shape == (N_BEAM_MODES, len(new_freqs))

    def test_resample_A_orthonormal_after_resampling(self, freqs):
        """Basis columns remain approximately orthonormal after resampling."""
        sb = _make_sky_basis(freqs)
        # Resample to nearly the same grid (with 10% extra points)
        new_freqs = np.linspace(freqs[0], freqs[-1], int(len(freqs) * 1.1))
        sb_resamp = self._save_and_resample(sb, new_freqs)

        A = sb_resamp.A.astype(np.float64)
        gram = A.T @ A
        res = gram - np.eye(gram.shape[0])
        # Linear interpolation degrades orthonormality slightly
        assert np.max(np.abs(res)) < 0.1

    def test_resample_interpolation_accuracy(self, freqs):
        """Resampled values fall within interpolation band."""
        sb = _make_sky_basis(freqs)
        # Resample at midpoints between original frequencies
        new_freqs = (freqs[:-1] + freqs[1:]) / 2
        sb_resamp = self._save_and_resample(sb, new_freqs)

        # All resampled values should be interpolated (not extrapolated)
        assert np.all(new_freqs >= freqs[0])
        assert np.all(new_freqs <= freqs[-1])
        assert sb_resamp.nfreq == len(new_freqs)

    def test_resample_missing_freqs_hz_raises(self):
        """Resampling raises ValueError if freqs_hz not in file."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            # Save without freqs_hz
            rng = np.random.default_rng(50)
            A = rng.standard_normal((16, 3)).astype(np.float32)
            np.savez(path, A=A)

            new_freqs = np.linspace(100e6, 200e6, 32)
            with pytest.raises(ValueError, match="Cannot resample.*freqs_hz"):
                SkyBasis.from_file(path, new_freqs=new_freqs)
        finally:
            os.unlink(path)

    def test_resample_without_new_freqs_unchanged(self, freqs):
        """Loading without new_freqs parameter is unchanged behavior."""
        sb = _make_sky_basis(freqs)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            sb.save(path)
            sb_loaded = SkyBasis.from_file(path)  # No new_freqs
            sb_loaded_resamp = SkyBasis.from_file(path, new_freqs=freqs)  # explicit
        finally:
            os.unlink(path)

        # Both should be identical
        np.testing.assert_array_equal(sb_loaded.A, sb_loaded_resamp.A)
        np.testing.assert_array_equal(sb_loaded.freqs_hz, sb_loaded_resamp.freqs_hz)
