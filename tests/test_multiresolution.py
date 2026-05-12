import numpy as np
import jax.numpy as jnp

from newnucal import (
    BeamModel,
    SkyModel,
    Calibrator,
    init_gain_params,
    resample_sky_coeffs,
    resample_beam_coeffs,
    resample_params,
    init_resampled_joint_state,
)


def test_resample_sky_coeffs_preserves_constant_when_upsampling(freqs, sky_basis):
    low = SkyModel(nside=2, freqs=freqs, basis=sky_basis)
    high = SkyModel(nside=4, freqs=freqs, basis=sky_basis)
    spectra = np.ones((low.npix, low.nfreq), dtype=np.float32) * 3.0
    coeffs_low = low.project(spectra)

    coeffs_high = resample_sky_coeffs(coeffs_low, low, high)
    spectra_high = high.deproject(np.asarray(coeffs_high))

    assert coeffs_high.shape == (high.npix, high.nmodes)
    np.testing.assert_allclose(spectra_high, 3.0, rtol=5e-5, atol=1e-4)


def test_resample_sky_coeffs_preserves_constant_when_downsampling(freqs, sky_basis):
    high = SkyModel(nside=4, freqs=freqs, basis=sky_basis)
    low = SkyModel(nside=2, freqs=freqs, basis=sky_basis)
    spectra = np.ones((high.npix, high.nfreq), dtype=np.float32) * 2.5
    coeffs_high = high.project(spectra)

    coeffs_low = resample_sky_coeffs(coeffs_high, high, low)
    spectra_low = low.deproject(np.asarray(coeffs_low))

    assert coeffs_low.shape == (low.npix, low.nmodes)
    np.testing.assert_allclose(spectra_low, 2.5, rtol=5e-5, atol=1e-4)


def test_resample_sky_coeffs_accepts_healpix_power(freqs, sky_basis):
    low = SkyModel(nside=2, freqs=freqs, basis=sky_basis)
    high = SkyModel(nside=4, freqs=freqs, basis=sky_basis)
    spectra = np.ones((low.npix, low.nfreq), dtype=np.float32)
    coeffs_low = low.project(spectra)

    coeffs_default = resample_sky_coeffs(coeffs_low, low, high)
    coeffs_flux = resample_sky_coeffs(coeffs_low, low, high, healpix_power=-2)

    assert coeffs_default.shape == coeffs_flux.shape
    assert not np.allclose(np.asarray(coeffs_default), np.asarray(coeffs_flux))


def test_resample_beam_coeffs_preserves_constant(freqs, beam_basis):
    low = BeamModel(nside=2, freqs=freqs, basis=beam_basis)
    high = BeamModel(nside=4, freqs=freqs, basis=beam_basis)
    spectra = np.ones((low.npix, low.nfreq), dtype=np.float32) * 0.7
    coeffs_low = low.project(spectra)

    coeffs_high = resample_beam_coeffs(coeffs_low, low, high)
    spectra_high = high.deproject(np.asarray(coeffs_high))

    assert coeffs_high.shape == (high.npix, high.nmodes)
    np.testing.assert_allclose(spectra_high, 0.7, rtol=5e-5, atol=3e-5)


def test_resample_beam_coeffs_can_normalize_max(freqs, beam_basis):
    low = BeamModel(nside=2, freqs=freqs, basis=beam_basis)
    high = BeamModel(nside=4, freqs=freqs, basis=beam_basis)
    spectra = np.ones((low.npix, low.nfreq), dtype=np.float32) * 0.7
    coeffs_low = low.project(spectra)

    coeffs_high = resample_beam_coeffs(
        coeffs_low,
        low,
        high,
        healpix_power=-2,
        normalize="max",
    )
    spectra_high = high.deproject(np.asarray(coeffs_high))

    np.testing.assert_allclose(np.max(np.abs(spectra_high), axis=0), 0.7, rtol=5e-5)


def test_resample_params_can_transfer_sky_or_beam(
    array, freqs, rot_matrices, sky_basis, beam_basis
):
    low_sky = SkyModel(nside=2, freqs=freqs, basis=sky_basis)
    high_sky = SkyModel(nside=4, freqs=freqs, basis=sky_basis)
    low_beam = BeamModel(nside=2, freqs=freqs, basis=beam_basis)
    high_beam = BeamModel(nside=4, freqs=freqs, basis=beam_basis)

    data = np.zeros((len(rot_matrices), len(freqs), array.nbls), dtype=np.complex64)
    cal_low = Calibrator(array, low_beam, low_sky, freqs, rot_matrices, data)
    cal_high = Calibrator(array, high_beam, high_sky, freqs, rot_matrices, data)

    params_low = {
        "sky_coeffs": jnp.zeros((low_sky.npix, low_sky.nmodes), dtype=jnp.float32),
        "beam_coeffs": jnp.array(low_beam.coeffs, dtype=jnp.float32),
        "log_ch_weights": np.zeros((cal_low.ntime, cal_low.nfreq), dtype=np.float32),
        **init_gain_params(cal_low.ntime, cal_low.nfreq),
    }

    sky_only = resample_params(params_low, cal_low, cal_high, sky=True, beam=False)
    both = resample_params(
        params_low,
        cal_low,
        cal_high,
        sky=True,
        beam=True,
        sky_healpix_power=0,
        beam_healpix_power=0,
        beam_normalize="none",
    )

    assert sky_only["sky_coeffs"].shape[0] == high_sky.npix
    assert both["sky_coeffs"].shape[0] == high_sky.npix
    assert both["beam_coeffs"].shape[0] == high_beam.npix
    np.testing.assert_allclose(sky_only["log_amp"], params_low["log_amp"])
    np.testing.assert_allclose(sky_only["log_ch_weights"], params_low["log_ch_weights"])


def test_resample_params_accepts_full_sky_params_from_masked_calibrator(
    array, freqs, rot_matrices, sky_basis, beam_basis
):
    low_sky = SkyModel(nside=2, freqs=freqs, basis=sky_basis)
    high_sky = SkyModel(nside=4, freqs=freqs, basis=sky_basis)
    low_beam = BeamModel(nside=2, freqs=freqs, basis=beam_basis)
    high_beam = BeamModel(nside=4, freqs=freqs, basis=beam_basis)

    data = np.zeros((len(rot_matrices), len(freqs), array.nbls), dtype=np.complex64)
    cal_low = Calibrator(array, low_beam, low_sky, freqs, rot_matrices, data)
    cal_high = Calibrator(array, high_beam, high_sky, freqs, rot_matrices, data)
    mask = np.zeros(high_sky.npix, dtype=bool)
    mask[::3] = True
    cal_high.apply_sky_mask(mask)

    params_high = {
        "sky_coeffs": jnp.zeros((high_sky.npix, high_sky.nmodes), dtype=jnp.float32),
        "beam_coeffs": jnp.array(high_beam.coeffs, dtype=jnp.float32),
        "log_ch_weights": None,
        **init_gain_params(cal_high.ntime, cal_high.nfreq),
    }
    params_low = resample_params(params_high, cal_high, cal_low, sky=True, beam=True)

    assert params_low["sky_coeffs"].shape == (low_sky.npix, low_sky.nmodes)
    assert params_low["beam_coeffs"].shape == (low_beam.npix, low_beam.nmodes)
    assert "log_ch_weights" not in params_low


def test_init_resampled_joint_state_runs_one_step(
    array, freqs, rot_matrices, sky_basis, beam_basis
):
    low_sky = SkyModel(nside=2, freqs=freqs, basis=sky_basis)
    high_sky = SkyModel(nside=4, freqs=freqs, basis=sky_basis)
    low_beam = BeamModel(nside=2, freqs=freqs, basis=beam_basis)
    high_beam = BeamModel(nside=4, freqs=freqs, basis=beam_basis)

    data = np.zeros((len(rot_matrices), len(freqs), array.nbls), dtype=np.complex64)
    cal_low = Calibrator(array, low_beam, low_sky, freqs, rot_matrices, data)
    cal_high = Calibrator(array, high_beam, high_sky, freqs, rot_matrices, data)
    params_low = {
        "sky_coeffs": jnp.zeros((low_sky.npix, low_sky.nmodes), dtype=jnp.float32),
        "beam_coeffs": jnp.array(low_beam.coeffs, dtype=jnp.float32),
        **init_gain_params(cal_low.ntime, cal_low.nfreq),
    }
    state_low = cal_low.init_joint_sky_beam_dirty_state(
        params_low, joint_anderson_history=2, solve_every={"gains": 0, "rfi": 0}
    )

    state_high = init_resampled_joint_state(
        cal_low,
        state_low,
        cal_high,
        joint_anderson_history=2,
        solve_every={"gains": 0, "rfi": 0},
    )
    state_high = cal_high.run_joint_sky_beam_dirty_state(state_high, n_iter=1)

    assert state_high.params["sky_coeffs"].shape[0] == high_sky.npix
    assert state_high.params["beam_coeffs"].shape[0] == high_beam.npix
    assert state_high.step == 1
