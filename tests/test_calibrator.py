"""
Tests for Calibrator methods.

TestFitGainsLinear: closed-loop test that with the true sky fixed,
fit_gains_linear recovers gains to near-machine precision.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from newnucal.basis import basis_project
from newnucal.gains import apply_gains, init_gain_params

# Mirror the conftest sizes
NSIDE_SKY = 8
ETA_SKY = 40e-9


@pytest.fixture
def sky_coeffs_true(forward_model, A_sky, freqs):
    """Random chromatic sky (power-law per pixel) projected onto DPSS basis."""
    rng = np.random.default_rng(42)
    npix = forward_model.npix_sky
    nfreq = len(freqs)
    ref_freq = 150e6
    spec_idx = rng.normal(-0.7, 0.1, npix).astype(np.float32)
    ref_flux  = rng.exponential(scale=1.0, size=npix).astype(np.float32)
    flux_map  = ref_flux[:, None] * (freqs[None, :] / ref_freq) ** spec_idx[:, None]
    return jnp.array(basis_project(flux_map, A_sky), dtype=jnp.float32)


@pytest.fixture
def calibrator_gains_setup(array, beam_model, freqs, rot_matrices,
                            forward_model, sky_coeffs_true, sky_model):
    """Returns (Calibrator, true_gain_params) where data = simulate(sky_true) * gains."""
    from newnucal.calibrator import Calibrator

    ntime = rot_matrices.shape[0]
    nfreq = len(freqs)

    vis_true = forward_model.simulate(sky_coeffs_true, jnp.array(rot_matrices))

    # Smooth gains: exact parametric form that fit_gains_linear inverts analytically
    f_norm = np.linspace(0, 1, nfreq)[None, :]   # (1, nfreq)
    t_norm = np.linspace(0, 1, ntime)[:, None]   # (ntime, 1)

    true_log_amp = (
        0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)
    ).astype(np.float32)
    true_phase = (
        0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)
    ).astype(np.float32)
    true_phi = np.zeros((ntime, 2, nfreq), dtype=np.float32)
    true_phi[:, 0, :] = (1e-4 * np.cos(2 * np.pi * f_norm)
                         + 3e-5 * np.sin(2 * np.pi * t_norm))
    true_phi[:, 1, :] = (5e-5 * np.sin(2 * np.pi * f_norm)
                         + 2e-5 * np.cos(2 * np.pi * t_norm))

    bls_j = jnp.array(array.bls, dtype=jnp.float32)
    vis_data = apply_gains(
        vis_true,
        jnp.array(true_log_amp), jnp.array(true_phase), jnp.array(true_phi),
        bls_j,
    )

    cal = Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=np.array(vis_data),
    )
    true_gains = {
        'log_amp': jnp.array(true_log_amp),
        'phase':   jnp.array(true_phase),
        'phi':     jnp.array(true_phi),
    }
    return cal, true_gains


class TestFitGainsLinear:

    def test_loss_near_zero_with_true_sky(self, calibrator_gains_setup, sky_coeffs_true):
        """With the true sky fixed, the linear gain solve should drive loss near zero."""
        cal, _ = calibrator_gains_setup
        params0 = {'sky_coeffs': sky_coeffs_true, **init_gain_params(cal.ntime, cal.nfreq)}
        loss_init = cal.calc_loss(params0)
        _, loss_fit = cal.fit_gains_linear(sky_coeffs_true)
        assert loss_fit < 1e-4 * loss_init, (
            f"Expected >10 000x loss reduction; got {loss_init:.3e} → {loss_fit:.3e} "
            f"(ratio {loss_fit / loss_init:.3e})"
        )

    def test_calibrated_vis_matches_data(self, calibrator_gains_setup, sky_coeffs_true):
        """Calibrated model visibilities should reproduce data to near-float32 precision."""
        cal, _ = calibrator_gains_setup
        gain_params, _ = cal.fit_gains_linear(sky_coeffs_true)
        params = {'sky_coeffs': sky_coeffs_true, **gain_params}
        vis_fit  = cal.simulate(params)
        rms_res  = float(jnp.sqrt(jnp.mean(jnp.abs(vis_fit - cal.data) ** 2)))
        rms_data = float(jnp.sqrt(jnp.mean(jnp.abs(cal.data) ** 2)))
        assert rms_res / rms_data < 1e-3, (
            f"Relative residual {rms_res / rms_data:.3e} > 1e-3"
        )

    def test_returned_loss_consistent_with_calc_loss(self, calibrator_gains_setup,
                                                      sky_coeffs_true):
        """Loss returned by fit_gains_linear must equal calc_loss on the same params."""
        cal, _ = calibrator_gains_setup
        gain_params, loss_returned = cal.fit_gains_linear(sky_coeffs_true)
        params = {'sky_coeffs': sky_coeffs_true, **gain_params}
        loss_recalc = cal.calc_loss(params)
        # Both calls use the same JIT function; allow small float32 re-evaluation noise.
        assert abs(loss_returned - loss_recalc) / (loss_recalc + 1e-30) < 0.1, (
            f"Returned loss {loss_returned:.3e} vs calc_loss {loss_recalc:.3e}"
        )

    def test_output_shapes(self, calibrator_gains_setup, sky_coeffs_true):
        """Gain arrays must have the correct shapes."""
        cal, _ = calibrator_gains_setup
        gain_params, _ = cal.fit_gains_linear(sky_coeffs_true)
        assert gain_params['log_amp'].shape == (cal.ntime, cal.nfreq)
        assert gain_params['phase'].shape   == (cal.ntime, cal.nfreq)
        assert gain_params['phi'].shape     == (cal.ntime, 2, cal.nfreq)

    def test_output_dtype(self, calibrator_gains_setup, sky_coeffs_true):
        """Gain arrays must be float32."""
        cal, _ = calibrator_gains_setup
        gain_params, _ = cal.fit_gains_linear(sky_coeffs_true)
        for key, val in gain_params.items():
            assert val.dtype == jnp.float32, f"{key}: expected float32, got {val.dtype}"



@pytest.fixture
def calibrator_rfi_setup(array, beam_model, freqs, rot_matrices, forward_model,
                          sky_coeffs_true, sky_model):
    """Return a Calibrator whose data matches the true sky and unity gains."""
    from newnucal.calibrator import Calibrator
    vis_true = forward_model.simulate(sky_coeffs_true, jnp.array(rot_matrices))
    cal = Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=np.array(vis_true),
    )
    params = {
        'sky_coeffs': sky_coeffs_true,
        'beam_coeffs': jnp.array(cal.fwd.beam_coeffs),
        **init_gain_params(cal.ntime, cal.nfreq),
    }
    return cal, params


class TestRFIWeightedCalibrator:

    def test_calc_loss_decreases_when_bad_channel_is_downweighted(self, calibrator_rfi_setup):
        cal, params = calibrator_rfi_setup
        data = np.array(cal.data)
        data[:, 2, :] += 50.0 + 0.0j
        cal.data = jnp.array(data, dtype=jnp.complex64)

        cal.set_channel_weights(np.ones((cal.ntime, cal.nfreq), dtype=np.float32))
        loss_full = cal.calc_loss(params)

        weights = np.ones((cal.ntime, cal.nfreq), dtype=np.float32)
        weights[:, 2] = 0.01
        cal.set_channel_weights(weights)
        loss_downweighted = cal.calc_loss(params)

        assert loss_downweighted < 0.05 * loss_full

    def test_calc_reduced_chi2_near_unity_for_known_noise(self, array, beam_model, freqs, rot_matrices, forward_model, sky_coeffs_true, sky_model):
        from newnucal.calibrator import Calibrator
        rng = np.random.default_rng(123)
        vis_true = np.array(forward_model.simulate(sky_coeffs_true, jnp.array(rot_matrices)))
        sigma = 0.05
        noise = (rng.normal(size=vis_true.shape) + 1j * rng.normal(size=vis_true.shape)) * (sigma / np.sqrt(2.0))
        data = vis_true + noise.astype(np.complex64)

        cal = Calibrator(
            array=array,
            beam_model=beam_model,
            sky_model=sky_model,
            freqs=freqs,
            rot_matrices=rot_matrices,
            data=data,
            inv_noise_var=np.full((len(rot_matrices), len(freqs)), 1.0 / sigma**2, dtype=np.float32),
        )
        params = {
            'sky_coeffs': sky_coeffs_true,
            'beam_coeffs': jnp.array(cal.fwd.beam_coeffs),
            **init_gain_params(cal.ntime, cal.nfreq),
        }
        red_chi2 = cal.calc_reduced_chi2(params, explicit_beam=True)
        assert 0.7 < red_chi2 < 1.3, f"Expected reduced chi2 near 1, got {red_chi2:.3f}"

    def test_fit_with_rfi_reweighting_updates_weights_and_tracks_history(self, calibrator_rfi_setup, monkeypatch):
        cal, params = calibrator_rfi_setup
        data = np.array(cal.data)
        # Inject narrowband contamination in one channel across all baselines/times.
        data[:, 1, :] += 20.0 + 10.0j
        cal.data = jnp.array(data, dtype=jnp.complex64)

        # Dummy fitter: keep parameters fixed and return the current weighted loss.
        def _dummy_fit(p, **kwargs):
            return p, cal.calc_loss(p, explicit_beam=True)

        monkeypatch.setattr(cal, 'dummy_fit', _dummy_fit, raising=False)

        from newnucal.rfi import RFIConfig
        cfg = RFIConfig(
            flagged_weight=0.05,
            min_weight=0.01,
            max_weight=1.0,
            smooth_window=5,
            score_center=1.5,
            score_slope=2.0,
            blend=0.0,
        )
        init_w = np.ones((cal.ntime, cal.nfreq), dtype=np.float32)

        params_out, state = cal.fit_with_rfi_reweighting(
            params,
            initial_channel_weights=init_w,
            inv_noise_var=np.ones((cal.ntime, cal.nfreq), dtype=np.float32),
            n_rounds=2,
            fit_method='dummy_fit',
            fit_kwargs={},
            rfi_config=cfg,
            verbose=False,
        )

        assert params_out.keys() == params.keys()
        assert 'channel_weights' in state
        assert 'history' in state
        assert len(state['history']) == 2
        assert state['channel_weights'].shape == (cal.ntime, cal.nfreq)
        # The contaminated channel should be strongly downweighted.
        assert float(np.mean(state['channel_weights'][:, 1])) < 0.2
        # Calibrator should retain the final weights internally too.
        np.testing.assert_allclose(np.asarray(cal.channel_weights), state['channel_weights'])


@pytest.fixture
def calibrator_noisy_setup(array, beam_model, freqs, rot_matrices, forward_model,
                            sky_coeffs_true, sky_model):
    """Return (Calibrator, true params, sigma) for noisy data with the true sky/beam."""
    from newnucal.calibrator import Calibrator
    rng = np.random.default_rng(2024)
    vis_true = np.array(forward_model.simulate(sky_coeffs_true, jnp.array(rot_matrices)), dtype=np.complex64)
    sigma = 0.05
    noise = (rng.normal(size=vis_true.shape) + 1j * rng.normal(size=vis_true.shape)) * (sigma / np.sqrt(2.0))
    data = vis_true + noise.astype(np.complex64)
    cal = Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=data,
        noise_sigma=np.full((len(rot_matrices), len(freqs)), sigma, dtype=np.float32),
    )
    params = {
        'sky_coeffs': sky_coeffs_true,
        'beam_coeffs': jnp.array(cal.fwd.beam_coeffs),
        **init_gain_params(cal.ntime, cal.nfreq),
    }
    return cal, params, sigma


class TestResumableAndChi2Tol:

    def test_resuming_preserves_anderson_state(self, calibrator_rfi_setup, monkeypatch):
        """Running in chunks should match a single continuous resumable run."""
        import newnucal.calibrator as calmod

        def make_fake_clock(dt=1.0):
            t = {'v': 0.0}
            def _fake_perf_counter():
                t['v'] += dt
                return t['v']
            return _fake_perf_counter

        fit_kwargs = dict(
            sky_step_size=0.2,
            sky_anderson_history=2,
            sky_aa_start=1,
            sky_aa_damping=0.5,
            solve_every={'gains': 0, 'beam': 0},
            target_reduced_chi2=None,
        )

        cal1, params0 = calibrator_rfi_setup
        monkeypatch.setattr(calmod._time, 'perf_counter', make_fake_clock())
        state = cal1.init_alternating_dirty_state(params0, **fit_kwargs)
        state = cal1.run_alternating_dirty_state(state, n_iter=2, verbose=False)
        state = cal1.run_alternating_dirty_state(state, n_iter=3, verbose=False)
        resumed_params = state.params
        resumed_loss = state.loss
        resumed_step = state.step
        resumed_hist_len = len(state.sky_acc._hist_g)

        cal2, params0_b = calibrator_rfi_setup
        monkeypatch.setattr(calmod._time, 'perf_counter', make_fake_clock())
        one_shot_params, one_shot_loss = cal2.fit_alternating_dirty(
            params0_b,
            n_iter=5,
            verbose=False,
            **fit_kwargs,
        )

        assert resumed_step == 5
        assert resumed_hist_len > 0
        np.testing.assert_allclose(np.asarray(resumed_params['sky_coeffs']), np.asarray(one_shot_params['sky_coeffs']), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(resumed_params['beam_coeffs']), np.asarray(one_shot_params['beam_coeffs']), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(resumed_params['log_amp']), np.asarray(one_shot_params['log_amp']), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(resumed_params['phase']), np.asarray(one_shot_params['phase']), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(resumed_params['phi']), np.asarray(one_shot_params['phi']), rtol=1e-6, atol=1e-6)
        assert resumed_loss == pytest.approx(one_shot_loss, rel=1e-6, abs=1e-6)

    def test_target_reduced_chi2_stops_resumable_fit_early(self, calibrator_noisy_setup, monkeypatch):
        """Resumable dirty fitting should stop once the target reduced chi^2 is reached."""
        import newnucal.calibrator as calmod
        cal, params0, _sigma = calibrator_noisy_setup

        # Deterministic timing for the adaptive scheduler
        t = {'v': 0.0}
        def _fake_perf_counter():
            t['v'] += 1.0
            return t['v']
        monkeypatch.setattr(calmod._time, 'perf_counter', _fake_perf_counter)

        state = cal.init_alternating_dirty_state(
            params0,
            sky_step_size=0.05,
            beam_step_size=0.05,
            sky_anderson_history=0,
            beam_anderson_history=0,
            solve_every={'gains': 0, 'beam': 0},
            target_reduced_chi2=1.5,
            reduced_chi2_check_every=1,
        )
        state = cal.run_alternating_dirty_state(state, n_iter=20, verbose=False)

        assert state.stop_reason == 'target_reduced_chi2'
        assert state.step < 20
        assert state.reduced_chi2 is not None
        assert state.reduced_chi2 <= 1.5


class TestPixelMasking:
    def test_apply_pixel_mask_custom_mask(self, calibrator_gains_setup):
        """apply_pixel_mask should accept and apply custom masks."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.npix_full

        # Create a mask for first half of pixels
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True

        cal.apply_pixel_mask(mask)

        assert cal.pixel_mask is not None
        assert np.array_equal(cal.pixel_mask, mask)
        assert cal.fwd.npix_sky == int(mask.sum())

    def test_apply_pixel_mask_invalid_size(self, calibrator_gains_setup):
        """apply_pixel_mask should reject masks of wrong size."""
        cal, _ = calibrator_gains_setup
        bad_mask = np.ones(cal.npix_full + 10, dtype=bool)

        with pytest.raises(ValueError, match="mask size"):
            cal.apply_pixel_mask(bad_mask)

    def test_fit_gains_linear_with_full_sky_input(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_gains_linear should accept full-sky parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.npix_full

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_pixel_mask(mask)

        # Pass full-sky sky_coeffs
        sky_full = sky_coeffs_true
        assert sky_full.shape[0] == npix_full

        gains, loss = cal.fit_gains_linear(sky_full)

        assert gains['log_amp'].shape == (cal.ntime, cal.nfreq)
        assert loss >= 0.0

    def test_fit_gains_linear_with_masked_input(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_gains_linear should accept masked parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.npix_full

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_pixel_mask(mask)

        # Pass masked sky_coeffs
        sky_masked = sky_coeffs_true[mask]
        assert sky_masked.shape[0] == int(mask.sum())

        gains, loss = cal.fit_gains_linear(sky_masked)

        assert gains['log_amp'].shape == (cal.ntime, cal.nfreq)
        assert loss >= 0.0

    def test_fit_sky_dirty_returns_full_sky(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_sky_dirty should always return full-sky parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.npix_full

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_pixel_mask(mask)

        # Fit
        sky_input = sky_coeffs_true
        gain_params, _ = cal.fit_gains_linear(sky_input)
        sky_out, _ = cal.fit_sky_dirty(sky_input, gain_params, n_iter=1)

        # Output should be full-sky
        assert sky_out.shape[0] == npix_full

    def test_fit_sky_dirty_unsolved_pixels_unchanged(self, calibrator_gains_setup, sky_coeffs_true):
        """Unsolved sky pixels should keep their original values."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.npix_full

        # Apply mask (first half only)
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_pixel_mask(mask)

        # Store original values
        sky_input = np.asarray(sky_coeffs_true)
        sky_unsolved = sky_input[~mask].copy()

        # Fit
        gain_params, _ = cal.fit_gains_linear(sky_input)
        sky_out, _ = cal.fit_sky_dirty(sky_input, gain_params, n_iter=1)

        # Check unsolved pixels unchanged
        np.testing.assert_array_almost_equal(sky_out[~mask], sky_unsolved, decimal=5)

    def test_fit_alternating_dirty_returns_full_sky(self, calibrator_gains_setup):
        """fit_alternating_dirty should always return full-sky parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.npix_full

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[: npix_full//2] = True
        cal.apply_pixel_mask(mask)

        # Fit
        params = cal.init_params()
        params_out, _ = cal.fit_alternating_dirty(params, n_iter=1, verbose=False)

        # Output should be full-sky
        assert params_out['sky_coeffs'].shape[0] == npix_full


class TestSkyBeamWeighting:
    def test_beam_weighting_shape_full_sky(self, calibrator_gains_setup):
        """get_sky_beam_weighting should return per-pixel weights for full sky."""
        cal, _ = calibrator_gains_setup
        weights = cal.get_sky_beam_weighting()

        assert weights.shape == (cal.fwd.npix_sky,)

    def test_beam_weighting_positive(self, calibrator_gains_setup):
        """Beam weights should be non-negative (squared magnitudes)."""
        cal, _ = calibrator_gains_setup
        weights = cal.get_sky_beam_weighting()

        assert np.all(weights >= 0.0)

    def test_beam_weighting_data_independent(self, calibrator_gains_setup):
        """Beam weights should be independent of data, only depend on beam/geometry."""
        cal, _ = calibrator_gains_setup
        weights1 = cal.get_sky_beam_weighting()
        weights2 = cal.get_sky_beam_weighting()

        np.testing.assert_array_equal(weights1, weights2)

    def test_beam_weighting_expanded_after_horizon_cut(self, calibrator_gains_setup, sky_coeffs_true):
        """After horizon cut, beam weights should be expanded to full sky with zeros for masked pixels."""
        cal, _ = calibrator_gains_setup
        npix_full_before = cal.npix_full
        npix_active = cal.apply_horizon_cut()

        weights_after_cut = cal.get_sky_beam_weighting()

        # After horizon cut, result is expanded back to full sky
        assert weights_after_cut.shape == (npix_full_before,)
        # Masked pixels have zero weight
        assert np.all(weights_after_cut[~cal.pixel_mask] == 0.0)
        # Active pixels have nonzero weight (from the reduced set)
        assert np.sum(weights_after_cut[cal.pixel_mask]) > 0.0
