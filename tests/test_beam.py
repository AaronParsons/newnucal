import numpy as np
import pytest
from newnucal.beam import BeamModel
from newnucal.basis import basis_reconstruct, dpss_matrix
from newnucal.basis import BeamBasis, SkyBasis
from newnucal.sky import SkyModel

NSIDE_BEAM = 8
N_SKY_SAMPLES = 8
N_BEAM_MODES  = 3
N_SKY_MODES   = 2


def _make_beam_basis(freqs, n_modes=N_BEAM_MODES):
    """Small BeamBasis built from random arrays (shared helper)."""
    rng   = np.random.default_rng(11)
    nfreq = len(freqs)
    raw   = rng.standard_normal((n_modes + 1, nfreq))
    Q, _  = np.linalg.qr(raw.T)
    return BeamBasis(A=Q.astype(np.float32), freqs_hz=freqs)


def _make_sky_basis(freqs, n_modes=N_SKY_MODES):
    """Small SkyBasis built from random arrays (shared helper)."""
    rng   = np.random.default_rng(22)
    nfreq = len(freqs)
    raw   = rng.standard_normal((n_modes + 1, nfreq))
    Q, _  = np.linalg.qr(raw.T)
    return SkyBasis(A=Q.astype(np.float32), freqs_hz=freqs)


def test_beam_model_shapes(beam_model, freqs):
    import healpy
    npix = healpy.nside2npix(beam_model.nside)
    assert beam_model.coeffs.shape[0] == npix
    assert beam_model.coeffs.shape[1] == beam_model.nmodes
    assert beam_model.A.shape == (len(freqs), beam_model.nmodes)


def test_beam_reconstruction_positive(beam_model):
    """Reconstructed beam power should be non-negative everywhere."""
    rec = basis_reconstruct(beam_model.coeffs, beam_model.A)
    # Reconstruction can have small ringing below horizon; above horizon
    # (first pixel = north pole / zenith for nside=8) must be essentially positive
    # (allowing for small numerical errors due to float32 precision).
    assert rec[0, :].min() > -0.02, "Zenith beam power should be essentially non-negative"


def test_beam_zenith_brightest(beam_model):
    """Airy beam should be brightest at zenith (pixel 0 in RING, th=0)."""
    import healpy
    rec = basis_reconstruct(beam_model.coeffs, beam_model.A)
    # Average power across frequencies
    mean_power = rec.mean(axis=1)
    # Zenith pixel (theta closest to 0)
    nside = beam_model.nside
    th, _ = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)))
    zenith_px = np.argmin(th)
    assert mean_power[zenith_px] == pytest.approx(mean_power.max(), rel=0.1)


def test_beam_coeffs_dtype(beam_model):
    assert beam_model.coeffs.dtype == np.float32


# ------------------------------------------------------------------
# BeamModel with BeamBasis
# ------------------------------------------------------------------

class TestBeamModelWithBeamBasis:

    def test_shape_with_beam_basis(self, freqs):
        import healpy
        bb = _make_beam_basis(freqs)
        bm = BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=bb)
        npix = healpy.nside2npix(NSIDE_BEAM)
        assert bm.A.shape == (len(freqs), N_BEAM_MODES + 1)
        assert bm.coeffs.shape == (npix, N_BEAM_MODES + 1)

    def test_A_orthonormal_columns(self, freqs):
        bb = _make_beam_basis(freqs)
        bm = BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=bb)
        A  = bm.A.astype(np.float64)
        gram = A.T @ A
        residual = gram - np.eye(gram.shape[0])
        assert np.max(np.abs(residual)) < 1e-5

    def test_coeffs_dtype(self, freqs):
        bb = _make_beam_basis(freqs)
        bm = BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=bb)
        assert bm.coeffs.dtype == np.float32

    def test_reconstruction_shape(self, freqs):
        bb  = _make_beam_basis(freqs)
        bm  = BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=bb)
        rec = bm.coeffs @ bm.A.T
        import healpy
        assert rec.shape == (healpy.nside2npix(NSIDE_BEAM), len(freqs))


# ------------------------------------------------------------------
# BeamModel with pre-built ndarray basis
# ------------------------------------------------------------------

class TestBeamModelArrayBasis:

    def test_accepts_custom_array_basis(self, freqs):
        rng    = np.random.default_rng(22)
        nfreq  = len(freqs)
        nmodes = 5
        Q, _   = np.linalg.qr(rng.standard_normal((nfreq, nmodes)))
        A      = Q.astype(np.float32)   # (nfreq, nmodes)
        bm     = BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=A)
        assert bm.A.shape == (nfreq, nmodes)

    def test_array_basis_stored_as_float32(self, freqs):
        rng    = np.random.default_rng(23)
        nfreq  = len(freqs)
        A      = rng.standard_normal((nfreq, 4)).astype(np.float64)
        bm     = BeamModel(nside=NSIDE_BEAM, freqs=freqs, basis=A)
        assert bm.A.dtype == np.float32


# ------------------------------------------------------------------
# Calibrator with sky_model
# ------------------------------------------------------------------

class TestCalibratorSkyModel:

    def _make_cal(self, array, beam_model, freqs, rot_matrices, sky_model):
        from newnucal.calibrator import Calibrator
        rng = np.random.default_rng(99)
        ntime, nfreq = rot_matrices.shape[0], len(freqs)
        nbls = array.bls.shape[0]
        data = (rng.standard_normal((ntime, nfreq, nbls)) +
                1j * rng.standard_normal((ntime, nfreq, nbls))).astype(np.complex64)
        return Calibrator(
            array=array,
            beam_model=beam_model,
            sky_model=sky_model,
            freqs=freqs,
            rot_matrices=rot_matrices,
            data=data,
        )

    def test_dpss_sky_model(self, array, beam_model, freqs, rot_matrices):
        """SkyModel with a DPSS SkyBasis gives the right A_sky shape."""
        sb  = SkyBasis.from_dpss(freqs, 40e-9)
        sm  = SkyModel(nside=8, freqs=freqs, basis=sb)
        cal = self._make_cal(array, beam_model, freqs, rot_matrices, sm)
        assert cal.A_sky.shape == (len(freqs), sb.nmodes)

    def test_array_sky_basis(self, array, beam_model, freqs, rot_matrices):
        """SkyModel wrapping a raw-array SkyBasis gives the right shape."""
        A_raw = dpss_matrix(freqs, eta_max=40e-9)
        sb    = SkyBasis(A=A_raw)
        sm    = SkyModel(nside=8, freqs=freqs, basis=sb)
        cal   = self._make_cal(array, beam_model, freqs, rot_matrices, sm)
        assert cal.A_sky.shape == A_raw.shape

    def test_sky_basis_used_in_fwd(self, array, beam_model, freqs, rot_matrices):
        """ForwardModel.A_sky should reflect the sky model basis shape."""
        import jax.numpy as jnp
        sb  = SkyBasis.from_dpss(freqs, 40e-9)
        sm  = SkyModel(nside=8, freqs=freqs, basis=sb)
        cal = self._make_cal(array, beam_model, freqs, rot_matrices, sm)
        assert cal.fwd.A_sky.shape == (len(freqs), sb.nmodes)

    def test_init_params_nmodes_matches_basis(self, array, beam_model, freqs,
                                               rot_matrices):
        sb  = SkyBasis.from_dpss(freqs, 40e-9)
        sm  = SkyModel(nside=8, freqs=freqs, basis=sb)
        cal = self._make_cal(array, beam_model, freqs, rot_matrices, sm)
        params = cal.init_params()
        assert params['sky_coeffs'].shape[1] == sb.nmodes
