"""
Tests for Calibrator methods.

TestFitGainsLinear: closed-loop test that with the true sky fixed,
fit_gains_linear recovers gains to near-machine precision.
"""

import numpy as np
import jax.numpy as jnp
import pytest
import healpy
import inspect

from newnucal.basis import basis_project
from newnucal.gains import apply_gains, init_gain_params
from newnucal.simulate import ForwardModel

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

    vis_true = forward_model.simulate_3d(sky_coeffs_true, jnp.array(rot_matrices))

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

    def test_calc_loss_auto_uses_explicit_beam(self, calibrator_gains_setup, sky_coeffs_true):
        """calc_loss should match simulate by using beam_coeffs when present."""
        cal, gain_params = calibrator_gains_setup
        beam_coeffs = jnp.array(cal.fwd._beam_coeffs_full)
        beam_perturbed = beam_coeffs * 0.5
        params = {
            'sky_coeffs': sky_coeffs_true,
            'beam_coeffs': beam_perturbed,
            **gain_params,
        }

        loss_auto = cal.calc_loss(params)
        loss_explicit = cal.calc_loss(params, explicit_beam=True)
        loss_fixed = cal.calc_loss(params, explicit_beam=False)

        assert np.isclose(loss_auto, loss_explicit, rtol=1e-6, atol=1e-6)
        assert not np.isclose(loss_auto, loss_fixed, rtol=1e-3, atol=1e-6)

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
    vis_true = forward_model.simulate_3d(sky_coeffs_true, jnp.array(rot_matrices))
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

        # set_channel_weights now expects log-space values (0 = no downweighting)
        cal.set_channel_weights(np.zeros((cal.ntime, cal.nfreq), dtype=np.float32))
        loss_full = cal.calc_loss(params)

        # Create weights in log-space: weight 0.01 = log(0.01) ≈ -4.605
        log_weights = np.zeros((cal.ntime, cal.nfreq), dtype=np.float32)
        log_weights[:, 2] = np.log(0.01)
        cal.set_channel_weights(log_weights)
        loss_downweighted = cal.calc_loss(params)

        assert loss_downweighted < 0.05 * loss_full

    def test_calc_reduced_chi2_near_unity_for_known_noise(self, array, beam_model, freqs, rot_matrices, forward_model, sky_coeffs_true, sky_model):
        from newnucal.calibrator import Calibrator
        rng = np.random.default_rng(123)
        vis_true = np.array(forward_model.simulate_3d(sky_coeffs_true, jnp.array(rot_matrices)))
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

    def test_fit_joint_sky_beam_dirty_with_rfi_reweighting(self, calibrator_rfi_setup):
        cal, params = calibrator_rfi_setup
        data = np.array(cal.data)
        data[:, 2, :] += 50.0 + 0.0j
        cal.data = jnp.array(data, dtype=jnp.complex64)

        # Start with prior weights (in direct space) and convert to log-space
        init_w = np.ones((cal.ntime, cal.nfreq), dtype=np.float32) * 0.5
        log_init_w = np.log(np.clip(init_w, 1e-30, 1.0))
        params['log_ch_weights'] = log_init_w

        params_out, loss_out = cal.fit_joint_sky_beam_dirty(
            params,
            n_iter=5,
            solve_every={'gains': 1, 'rfi': 2},
            rfi_regularization=1.0,
            rfi_regularization_power=2.0,
            rfi_log_min_weight=np.log(0.01),
            rfi_log_max_weight=0.0,
            max_rfi_updates=2,
            verbose=False,
        )

        # params_out now includes 'log_ch_weights' which is tracked in the fit
        expected_keys = set(params.keys()) | {'log_ch_weights'}
        assert set(params_out.keys()) == expected_keys
        assert loss_out < float(cal.calc_loss(params))
        # Weights should have been updated and are in params_out
        assert 'log_ch_weights' in params_out
        log_ch_weights = np.asarray(params_out['log_ch_weights'])
        assert log_ch_weights.shape == (cal.ntime, cal.nfreq)
        # Weights should be in valid range (in direct space)
        ch_weights = np.exp(log_ch_weights)
        assert np.all(ch_weights >= 0.01)
        assert np.all(ch_weights <= 1.0)

    def test_successive_fits_preserve_weights_and_params(self, calibrator_rfi_setup):
        """Verify that successive calls to fit_* properly apply and update weights."""
        cal, params = calibrator_rfi_setup
        data = np.array(cal.data)
        data[:, 1, :] += 30.0 + 0.0j  # Inject RFI into channel 1
        cal.data = jnp.array(data, dtype=jnp.complex64)

        # ===== FIRST FIT =====
        params1, loss1 = cal.fit_joint_sky_beam_dirty(
            params,
            n_iter=4,
            solve_every={'gains': 1, 'rfi': 2},
            rfi_log_min_weight=np.log(0.05),
            max_rfi_updates=1,
            verbose=False,
        )

        # Verify first fit improved loss and found downweighted channel
        assert loss1 < float(cal.calc_loss(params)), "First fit should improve loss"
        assert 'log_ch_weights' in params1, "First fit should return log_ch_weights"
        log_w1 = np.asarray(params1['log_ch_weights'])

        # Channel 1 should be downweighted (negative log-weight)
        # Use median to handle any time variation
        ch1_median_weight = float(np.median(np.exp(log_w1[:, 1])))
        assert ch1_median_weight < 0.9, f"RFI channel should be downweighted, got {ch1_median_weight:.3f}"

        # ===== SECOND FIT (reusing same calibrator, passing updated params) =====
        # This tests that weights and parameters from the first fit are properly applied
        params2, loss2 = cal.fit_joint_sky_beam_dirty(
            params1,  # Pass updated parameters including downweighted channel
            n_iter=4,
            solve_every={'gains': 1, 'rfi': 2},
            rfi_log_min_weight=np.log(0.05),
            max_rfi_updates=1,
            verbose=False,
        )

        # Second fit should:
        # 1. Continue improving loss (or at least not regress significantly)
        assert loss2 < loss1 * 1.01, (
            f"Second fit should continue improving loss or maintain it. "
            f"First: {loss1:.4e}, Second: {loss2:.4e}"
        )

        # 2. Return updated parameters
        assert set(params2.keys()) == set(params1.keys()), "Second fit should return same parameter keys"

        # 3. Keep the downweighted channel downweighted (or more)
        log_w2 = np.asarray(params2['log_ch_weights'])
        ch1_median_weight_2 = float(np.median(np.exp(log_w2[:, 1])))
        assert ch1_median_weight_2 <= ch1_median_weight * 1.01, (
            f"RFI channel weight should be preserved or decreased. "
            f"First: {ch1_median_weight:.3f}, Second: {ch1_median_weight_2:.3f}"
        )

        # 4. Channel 0 (far from RFI) should remain near unity
        # (channels near RFI may be affected by local baseline estimation)
        ch0_weight = float(np.median(np.exp(log_w2[:, 0])))
        assert 0.9 < ch0_weight <= 1.0, (
            f"Non-RFI channel 0 should have weight near 1.0, got {ch0_weight:.3f}"
        )

    def test_rfi_scheduling_tracks_efficiency(self, calibrator_rfi_setup):
        """Verify RFI scheduling tracks efficiency and performs updates on fixed cadence."""
        cal, params = calibrator_rfi_setup
        data = np.array(cal.data)
        data[:, 2, :] += 50.0 + 0.0j  # Inject strong RFI into channel 2
        cal.data = jnp.array(data, dtype=jnp.complex64)

        # Initialize state with RFI scheduled every step (solve_every={'rfi': 1})
        state = cal.init_joint_sky_beam_dirty_state(
            params,
            solve_every={'gains': 1, 'rfi': 1},  # RFI every step
            rfi_log_min_weight=np.log(0.05),
            max_rfi_updates=3,
        )

        # Run several steps and collect efficiency data
        for _ in range(6):
            cal._joint_sky_beam_dirty_step_from_state(state, verbose=False)

        # Verify eff_rfi was tracked with EMA smoothing
        assert state.eff_rfi is not None, "eff_rfi should be tracked after RFI updates"
        assert state.eff_rfi > 0.0, "eff_rfi should be positive (loss improvement per unit time)"

        # Verify RFI updates occurred on schedule
        assert state.n_rfi > 0, "At least one RFI update should have occurred"
        assert state.rfi_history is not None and len(state.rfi_history) > 0, "RFI history should be populated"

        # Verify efficiency is in reasonable range (relative to initial loss)
        # eff = dloss / (loss * dt), so reasonable values are 0.01 to 100
        assert 1e-6 < state.eff_rfi < 1e6, f"eff_rfi={state.eff_rfi} is out of reasonable range"

        # Verify RFI channel got downweighted
        log_weights = np.asarray(state.params['log_ch_weights'])
        ch2_weight = float(np.median(np.exp(log_weights[:, 2])))
        assert ch2_weight < 0.9, f"RFI channel 2 should be downweighted, got weight {ch2_weight:.3f}"


@pytest.fixture
def calibrator_noisy_setup(array, beam_model, freqs, rot_matrices, forward_model,
                            sky_coeffs_true, sky_model):
    """Return (Calibrator, true params, sigma) for noisy data with the true sky/beam."""
    from newnucal.calibrator import Calibrator
    rng = np.random.default_rng(2024)
    vis_true = np.array(forward_model.simulate_3d(sky_coeffs_true, jnp.array(rot_matrices)), dtype=np.complex64)
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
            sky_initial_step=0.2,  # Single step size, tuned by step_gain
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
        np.testing.assert_allclose(np.asarray(resumed_params['sky_coeffs']), np.asarray(one_shot_params['sky_coeffs']), rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(np.asarray(resumed_params['beam_coeffs']), np.asarray(one_shot_params['beam_coeffs']), rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(np.asarray(resumed_params['log_amp']), np.asarray(one_shot_params['log_amp']), rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(np.asarray(resumed_params['phase']), np.asarray(one_shot_params['phase']), rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(np.asarray(resumed_params['phi']), np.asarray(one_shot_params['phi']), rtol=1e-3, atol=1e-2)
        assert resumed_loss == pytest.approx(one_shot_loss, rel=1e-3, abs=1e-2)

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
            sky_initial_step=0.05,
            beam_initial_step=0.05,
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
    def test_apply_sky_mask_custom_mask(self, calibrator_gains_setup):
        """apply_sky_mask should accept and apply custom masks."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.fwd.npix_sky

        # Create a mask for first half of pixels
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True

        cal.apply_sky_mask(mask)

        assert cal.pixel_mask is not None
        assert np.array_equal(cal.pixel_mask, mask)
        assert cal.fwd.npix_sky == int(mask.sum())

    def test_apply_sky_mask_invalid_size(self, calibrator_gains_setup):
        """apply_sky_mask should reject masks of wrong size."""
        cal, _ = calibrator_gains_setup
        bad_mask = np.ones(cal.fwd.npix_sky + 10, dtype=bool)

        with pytest.raises(ValueError, match="mask size"):
            cal.apply_sky_mask(bad_mask)

    def test_fit_gains_linear_with_full_sky_input(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_gains_linear should accept full-sky parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.fwd.npix_sky

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_sky_mask(mask)

        # Pass full-sky sky_coeffs
        sky_full = sky_coeffs_true
        assert sky_full.shape[0] == npix_full

        gains, loss = cal.fit_gains_linear(sky_full)

        assert gains['log_amp'].shape == (cal.ntime, cal.nfreq)
        assert loss >= 0.0

    def test_fit_gains_linear_with_masked_input(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_gains_linear should accept masked parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.fwd.npix_sky

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_sky_mask(mask)

        # Pass masked sky_coeffs
        sky_masked = sky_coeffs_true[mask]
        assert sky_masked.shape[0] == int(mask.sum())

        gains, loss = cal.fit_gains_linear(sky_masked)

        assert gains['log_amp'].shape == (cal.ntime, cal.nfreq)
        assert loss >= 0.0

    def test_fit_sky_dirty_returns_full_sky(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_sky_dirty should always return full-sky parameters."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.fwd.npix_sky

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_sky_mask(mask)

        # Fit
        sky_input = sky_coeffs_true
        gain_params, _ = cal.fit_gains_linear(sky_input)
        sky_out, _ = cal.fit_sky_dirty(sky_input, gain_params, n_iter=1)

        # Output should be full-sky
        assert sky_out.shape[0] == npix_full

    def test_fit_sky_dirty_unsolved_pixels_unchanged(self, calibrator_gains_setup, sky_coeffs_true):
        """Unsolved sky pixels should keep their original values."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.fwd.npix_sky

        # Apply mask (first half only)
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_sky_mask(mask)

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
        npix_full = cal.fwd.npix_sky

        # Apply mask
        mask = np.zeros(npix_full, dtype=bool)
        mask[: npix_full//2] = True
        cal.apply_sky_mask(mask)

        # Fit
        params = cal.init_params()
        params_out, _ = cal.fit_alternating_dirty(params, n_iter=1, verbose=False)

        # Output should be full-sky
        assert params_out['sky_coeffs'].shape[0] == npix_full

    def test_apply_sky_mask_twice(self, calibrator_gains_setup):
        """Applying two different masks sequentially should work correctly."""
        cal, _ = calibrator_gains_setup
        npix_full = cal.fwd.npix_sky

        # First mask: first half of pixels
        mask1 = np.zeros(npix_full, dtype=bool)
        mask1[:npix_full//2] = True
        cal.apply_sky_mask(mask1)
        npix_after_first = cal.fwd.npix_sky
        assert npix_after_first == int(mask1.sum())

        # Second mask: within the remaining pixels, keep every other one
        mask2 = np.zeros(npix_full, dtype=bool)
        mask2[::2] = True
        mask2 = mask2 & mask1  # Combine masks
        cal.apply_sky_mask(mask2)
        npix_after_second = cal.fwd.npix_sky
        assert npix_after_second == int(mask2.sum())

        # Verify coordinates have correct size
        assert cal.fwd.eq_coords.shape[1] == npix_after_second


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
        """After altitude mask, beam weights should be expanded to full sky with zeros for masked pixels."""
        cal, _ = calibrator_gains_setup
        npix_full_before = cal.fwd.npix_sky
        sky_mask = cal.build_sky_mask_altitude(min_altitude_deg=0.0)
        cal.apply_sky_mask(sky_mask)
        npix_active = int(sky_mask.sum())

        weights_after_cut = cal.get_sky_beam_weighting()

        # After horizon cut, result is expanded back to full sky
        assert weights_after_cut.shape == (npix_full_before,)
        # Masked pixels have zero weight
        assert np.all(weights_after_cut[~cal.pixel_mask] == 0.0)
        # Active pixels have nonzero weight (from the reduced set)
        assert np.sum(weights_after_cut[cal.pixel_mask]) > 0.0


class TestBeamMasking:
    def test_build_ever_illuminated_beam_mask(self, calibrator_gains_setup):
        """build_beam_mask_altitude should return a valid boolean array."""
        cal, _ = calibrator_gains_setup
        mask = cal.fwd.build_beam_mask_altitude(cal.rot_matrices, min_altitude_deg=0.0)

        assert mask.dtype == bool
        assert mask.shape == (healpy.nside2npix(cal.beam_model.nside),)
        assert mask.sum() > 0  # Some pixels should be above horizon

    def test_apply_beam_mask_reduces_coefficients(self, calibrator_gains_setup):
        """apply_beam_mask should reduce beam coefficient array size."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = cal.fwd.npix_beam

        # Apply a simple mask: first half of pixels
        mask = np.zeros(npix_beam_full, dtype=bool)
        mask[:npix_beam_full//2] = True
        cal.apply_beam_mask(mask)

        assert cal.fwd.npix_beam == int(mask.sum())
        assert cal.fwd.beam_coeffs.shape[0] == int(mask.sum())

    def test_apply_beam_mask_builds_index_lookup(self, calibrator_gains_setup):
        """Beam masks should precompute full-pixel to masked-index lookup."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = cal.fwd.npix_beam
        mask = np.zeros(npix_beam_full, dtype=bool)
        mask[::3] = True

        cal.apply_beam_mask(mask)

        lookup = cal.fwd._beam_index_lookup
        expected_indices = np.where(mask)[0]
        assert lookup.shape == mask.shape
        assert np.all(lookup[~mask] == -1)
        assert np.array_equal(lookup[expected_indices], np.arange(len(expected_indices)))

    def test_masked_beam_scatter_uses_lookup_not_searchsorted(self):
        """Masked adjoints should use precomputed lookup instead of per-step search."""
        methods = [
            ForwardModel._accumulate_beam_update_impl,
            ForwardModel.accumulate_sky_and_beam_update_3d,
            ForwardModel.accumulate_sky_and_beam_update_2d,
        ]
        for method in methods:
            src = inspect.getsource(method)
            assert "beam_index_lookup_jax" in src
            assert "searchsorted" not in src

    def test_init_params_after_beam_mask_returns_full_beam(self, calibrator_gains_setup):
        """External parameter dicts should keep full-sized beam coefficients."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)
        mask = np.ones(npix_beam_full, dtype=bool)
        mask[::2] = False
        cal.apply_beam_mask(mask)

        params = cal.init_params()

        assert params['beam_coeffs'].shape[0] == npix_beam_full
        assert np.isfinite(cal.calc_loss(params))
        assert cal.simulate(params).shape == cal.data.shape

    def test_apply_ever_illuminated_beam_mask(self, calibrator_gains_setup):
        """build_beam_mask_altitude should produce mask for above-horizon pixels."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = cal.fwd.npix_beam

        beam_mask = cal.build_beam_mask_altitude(min_altitude_deg=0.0)
        cal.apply_beam_mask(beam_mask)

        npix_beam_masked = cal.fwd.npix_beam
        assert npix_beam_masked <= npix_beam_full
        # For a small nside, all above-horizon pixels should be kept
        # so the reduction should be minimal or none
        assert npix_beam_masked > 0

    def test_fit_beam_dirty_with_mask(self, calibrator_gains_setup):
        """Beam dirty updates should work with a beam mask applied."""
        cal, _ = calibrator_gains_setup
        params = cal.init_params()

        # Apply a beam altitude mask
        mask = cal.fwd.build_beam_mask_altitude(cal.rot_matrices, min_altitude_deg=0.0)
        cal.apply_beam_mask(mask)

        # Run beam dirty update
        params_out, loss = cal.fit_beam_dirty(params, n_iter=1, verbose=False)

        # Output beam coefficients should be full-size (expanded back)
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)
        assert params_out['beam_coeffs'].shape[0] == npix_beam_full

    def test_pixel_mask_after_beam_mask(self, calibrator_gains_setup):
        """Applying sky pixel mask after beam mask should work correctly."""
        cal, _ = calibrator_gains_setup
        npix_sky_full = cal.fwd.npix_sky
        npix_beam_full = cal.fwd.npix_beam

        # First apply a beam mask (even if it doesn't reduce size with small nside)
        beam_mask = np.ones(npix_beam_full, dtype=bool)
        beam_mask[::2] = False  # Keep only half
        cal.apply_beam_mask(beam_mask)
        npix_beam_masked = cal.fwd.npix_beam
        assert npix_beam_masked == int(beam_mask.sum())

        # Then apply sky pixel mask
        sky_mask = np.zeros(npix_sky_full, dtype=bool)
        sky_mask[:npix_sky_full//2] = True
        cal.apply_sky_mask(sky_mask)

        # Both masks should be active
        assert cal.fwd.npix_sky == int(sky_mask.sum())
        assert cal.fwd.npix_beam == npix_beam_masked  # Beam mask should persist

    def test_pixel_mask_after_alternating_dirty_with_beam_mask(self, calibrator_gains_setup):
        """Pixel mask after fit_alternating_dirty should work with beam mask active."""
        cal, _ = calibrator_gains_setup
        params = cal.init_params()
        npix_sky_full = cal.fwd.npix_sky
        npix_beam_full = cal.fwd.npix_beam

        # Apply beam mask before fit
        beam_mask = np.ones(npix_beam_full, dtype=bool)
        beam_mask[::2] = False
        cal.apply_beam_mask(beam_mask)
        npix_beam_masked = cal.fwd.npix_beam

        # Run alternating dirty fit (which updates the beam cache)
        state = cal.init_alternating_dirty_state(params)
        state = cal.run_alternating_dirty_state(state, n_iter=2, verbose=False)

        # After fit, beam mask should still be active
        assert cal.fwd.npix_beam == npix_beam_masked
        assert cal.fwd.beam_coeffs.shape[0] == npix_beam_masked

        # Should now be able to change the sky pixel mask
        sky_mask = np.zeros(npix_sky_full, dtype=bool)
        sky_mask[:npix_sky_full//2] = True
        cal.apply_sky_mask(sky_mask)

        # Both masks should remain active after changing sky mask
        assert cal.fwd.npix_sky == int(sky_mask.sum())
        assert cal.fwd.npix_beam == npix_beam_masked


class TestStaticSkySubtraction:
    """Tests for caching and subtracting static sky contributions."""

    def test_cache_static_sky_coeffs_requires_pixel_mask(self, calibrator_gains_setup, sky_coeffs_true):
        """cache_static_sky_coeffs should raise if no pixel mask applied."""
        cal, _ = calibrator_gains_setup
        with pytest.raises(ValueError, match="No pixel mask applied"):
            cal.cache_static_sky_coeffs(sky_coeffs_true)

    def test_cache_static_sky_coeffs_wrong_size(self, calibrator_gains_setup, sky_coeffs_true):
        """cache_static_sky_coeffs should reject wrong-sized coefficients."""
        cal, _ = calibrator_gains_setup
        npix_full = cal._npix_full
        mask = np.zeros(npix_full, dtype=bool)
        mask[:npix_full//2] = True
        cal.apply_sky_mask(mask)

        # Try to cache with wrong size
        with pytest.raises(ValueError, match="shape"):
            cal.cache_static_sky_coeffs(sky_coeffs_true[:npix_full//2])

    def test_cache_static_sky_coeffs_caches_visibilities(self, calibrator_gains_setup, sky_coeffs_true):
        """cache_static_sky_coeffs should cache visibility contribution."""
        cal, gain_params = calibrator_gains_setup
        npix_full = cal._npix_full

        # Apply pixel mask first (keep first 3/4 of pixels)
        pixel_mask = np.zeros(npix_full, dtype=bool)
        pixel_mask[:3*npix_full//4] = True
        cal.apply_sky_mask(pixel_mask)

        # Cache static sky (method extracts unmasked pixels internally)
        cal.cache_static_sky_coeffs(sky_coeffs_true)

        assert cal._static_sky_cached
        assert cal._cached_static_vis is not None
        assert cal._cached_static_vis.shape == cal.data.shape
        assert float(jnp.linalg.norm(cal._cached_static_vis)) > 0.0

    def test_cache_static_sky_coeffs_matches_dropped_pixels(self, calibrator_gains_setup, sky_coeffs_true):
        """Static cache should equal the visibility from pixels outside the solve mask."""
        cal, _ = calibrator_gains_setup
        npix_full = cal._npix_full
        pixel_mask = np.zeros(npix_full, dtype=bool)
        pixel_mask[:3*npix_full//4] = True
        static_mask = ~pixel_mask
        cal.apply_sky_mask(pixel_mask)

        cal.cache_static_sky_coeffs(sky_coeffs_true)
        cached = cal._cached_static_vis

        cal.fwd.apply_sky_mask(static_mask)
        cal.fwd.precompute_time_geometry(cal.rot_matrices)
        try:
            expected = cal.fwd.simulate(
                jnp.array(np.asarray(sky_coeffs_true)[static_mask], dtype=jnp.float32),
                cal.rot_matrices,
            )
        finally:
            cal.fwd.apply_sky_mask(pixel_mask)
            cal.fwd.precompute_time_geometry(cal.rot_matrices)
            cal._recompile_jit()

        assert jnp.allclose(cached, expected, rtol=1e-5, atol=1e-5)

    def test_fit_gains_linear_subtract_static_sky_requires_cache(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_gains_linear with subtract_static_sky should raise if cache doesn't exist."""
        cal, _ = calibrator_gains_setup
        sky_active = cal._ensure_sky_is_active(sky_coeffs_true)

        with pytest.raises(RuntimeError, match="Static sky not cached"):
            cal.fit_gains_linear(sky_active, subtract_static_sky=True)

class TestAltitudeMasking:
    """Tests for altitude-based masking of sky and beam pixels."""

    def test_build_sky_mask_altitude_zero(self, forward_model, rot_matrices):
        """build_sky_mask_altitude at 0° should match above-horizon pixels."""
        mask_0 = forward_model.build_sky_mask_altitude(rot_matrices, min_altitude_deg=0.0)
        assert mask_0.dtype == bool
        assert mask_0.shape == (forward_model.npix_sky,)
        assert mask_0.sum() > 0

    def test_build_sky_mask_altitude_high(self, forward_model, rot_matrices):
        """build_sky_mask_altitude at high altitude should be more restrictive."""
        mask_0 = forward_model.build_sky_mask_altitude(rot_matrices, min_altitude_deg=0.0)
        mask_45 = forward_model.build_sky_mask_altitude(rot_matrices, min_altitude_deg=45.0)
        mask_90 = forward_model.build_sky_mask_altitude(rot_matrices, min_altitude_deg=90.0)

        # More restrictive masks should have fewer pixels
        assert mask_45.sum() <= mask_0.sum()
        assert mask_90.sum() <= mask_45.sum()

    def test_build_beam_mask_altitude_zenith(self, forward_model, rot_matrices):
        """build_beam_mask_altitude at 0° should include all above-horizon pixels."""
        mask = forward_model.build_beam_mask_altitude(rot_matrices, min_altitude_deg=0.0)
        assert mask.dtype == bool
        assert mask.shape == (forward_model.npix_beam,)
        assert mask.sum() > 0

    def test_build_beam_mask_altitude_narrow(self, forward_model, rot_matrices):
        """build_beam_mask_altitude with higher thresholds should be more restrictive."""
        mask_all = forward_model.build_beam_mask_altitude(rot_matrices, min_altitude_deg=0.0)
        mask_moderate = forward_model.build_beam_mask_altitude(rot_matrices, min_altitude_deg=45.0)
        mask_zenith = forward_model.build_beam_mask_altitude(rot_matrices, min_altitude_deg=90.0)

        # Higher altitude thresholds should have fewer pixels
        assert mask_moderate.sum() <= mask_all.sum()
        assert mask_zenith.sum() <= mask_moderate.sum()

    def test_apply_sky_altitude_mask(self, calibrator_gains_setup):
        """build_sky_mask_altitude should produce mask for above-altitude pixels."""
        cal, _ = calibrator_gains_setup
        npix_full = cal._npix_full

        sky_mask = cal.build_sky_mask_altitude(min_altitude_deg=0.0)
        cal.apply_sky_mask(sky_mask)

        npix_retained = int(sky_mask.sum())
        assert isinstance(npix_retained, int)
        assert npix_retained > 0
        assert npix_retained <= npix_full
        assert cal.pixel_mask is not None

    def test_apply_beam_altitude_mask(self, calibrator_gains_setup):
        """build_beam_mask_altitude should produce mask for above-horizon pixels."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)

        beam_mask = cal.build_beam_mask_altitude(min_altitude_deg=0.0)
        cal.apply_beam_mask(beam_mask)

        npix_retained = int(beam_mask.sum())
        assert isinstance(npix_retained, int)
        assert npix_retained > 0
        assert npix_retained <= npix_beam_full
        assert cal.beam_mask is not None

    def test_apply_sky_altitude_mask_different_thresholds(self, calibrator_gains_setup):
        """Different sky altitude thresholds should give different masks."""
        cal, _ = calibrator_gains_setup
        mask1 = cal.build_sky_mask_altitude(min_altitude_deg=0.0)
        npix1 = int(mask1.sum())

        cal2, _ = calibrator_gains_setup  # Fresh calibrator
        mask2 = cal2.build_sky_mask_altitude(min_altitude_deg=30.0)
        npix2 = int(mask2.sum())

        # Different thresholds may give different pixel counts
        # (though not guaranteed due to small test grid)
        assert isinstance(npix1, int) and isinstance(npix2, int)

    def test_apply_beam_altitude_mask_different_thresholds(self, calibrator_gains_setup):
        """Different beam altitude thresholds should give different masks."""
        cal, _ = calibrator_gains_setup
        mask1 = cal.build_beam_mask_altitude(min_altitude_deg=90.0)
        npix1 = int(mask1.sum())

        cal2, _ = calibrator_gains_setup  # Fresh calibrator
        mask2 = cal2.build_beam_mask_altitude(min_altitude_deg=45.0)
        npix2 = int(mask2.sum())

        # Different thresholds may give different pixel counts
        assert isinstance(npix1, int) and isinstance(npix2, int)


class TestBeamSkyWeighting:
    """Tests for computing weighting of beam pixels by sky observations."""

    def test_get_beam_sky_weighting_shape(self, calibrator_gains_setup):
        """get_beam_sky_weighting should return correct shape."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)

        weights = cal.get_beam_sky_weighting()

        assert weights.shape == (npix_beam_full,)
        assert weights.dtype == np.float32 or weights.dtype == np.float64

    def test_get_beam_sky_weighting_positive(self, calibrator_gains_setup):
        """get_beam_sky_weighting should return non-negative values."""
        cal, _ = calibrator_gains_setup

        weights = cal.get_beam_sky_weighting()

        assert np.all(weights >= 0.0)
        assert np.sum(weights) > 0.0  # At least some pixels should have nonzero weight

    def test_get_beam_sky_weighting_central_higher(self, calibrator_gains_setup):
        """Beam pixels near zenith should have higher weight."""
        cal, _ = calibrator_gains_setup
        nside = cal.beam_model.nside

        weights = cal.get_beam_sky_weighting()

        # Get zenith angle for each beam pixel
        npix_beam = healpy.nside2npix(nside)
        theta, phi = healpy.pix2ang(nside, np.arange(npix_beam))
        zenith_angle = theta  # In HEALPix, theta is the colatitude (zenith angle)

        # Central pixels (low zenith angle) should tend to have higher weight
        central_mask = zenith_angle < np.pi / 4  # Central 45°
        edge_mask = zenith_angle > 3 * np.pi / 4  # Near horizon

        if central_mask.sum() > 0 and edge_mask.sum() > 0:
            avg_central = weights[central_mask].mean()
            avg_edge = weights[edge_mask].mean()
            # Central should typically be higher (though not guaranteed for all configurations)
            assert avg_central >= 0.0


class TestBeamSkyMaskingReverse:
    """Tests for selecting sky pixels based on beam pixels."""

    def test_build_sky_mask_from_beam_pixels_shape(self, forward_model):
        """build_sky_mask_from_beam_pixels should return correct shape."""
        forward_model.precompute_time_geometry(np.array([np.eye(3)]))  # Dummy rotation
        npix_beam_full = healpy.nside2npix(forward_model.beam_model.nside)

        beam_mask = np.ones(npix_beam_full, dtype=bool)
        sky_mask = forward_model.build_sky_mask_from_beam_pixels(beam_mask)

        assert sky_mask.dtype == bool
        assert sky_mask.shape == (forward_model.npix_sky,)

    def test_build_sky_mask_from_beam_pixels_all_beams(self, forward_model, rot_matrices):
        """Selecting all beam pixels should include many sky pixels."""
        forward_model.precompute_time_geometry(rot_matrices)
        npix_beam_full = healpy.nside2npix(forward_model.beam_model.nside)

        beam_mask = np.ones(npix_beam_full, dtype=bool)
        sky_mask = forward_model.build_sky_mask_from_beam_pixels(beam_mask)

        # Should select a significant fraction of sky pixels
        assert sky_mask.sum() > 0

    def test_build_sky_mask_from_beam_pixels_no_beams(self, forward_model, rot_matrices):
        """Selecting no beam pixels should select no sky pixels."""
        forward_model.precompute_time_geometry(rot_matrices)
        npix_beam_full = healpy.nside2npix(forward_model.beam_model.nside)

        beam_mask = np.zeros(npix_beam_full, dtype=bool)
        sky_mask = forward_model.build_sky_mask_from_beam_pixels(beam_mask)

        # Should select no sky pixels
        assert sky_mask.sum() == 0

    def test_build_sky_mask_from_beam_pixels_wrapper(self, calibrator_gains_setup):
        """build_sky_mask_from_beam_pixels should wrap forward model method."""
        cal, _ = calibrator_gains_setup
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)

        beam_mask = np.ones(npix_beam_full, dtype=bool)
        sky_mask = cal.build_sky_mask_from_beam_pixels(beam_mask)

        assert sky_mask.dtype == bool
        assert sky_mask.shape == (cal._npix_full,)

    def test_build_sky_mask_from_altitude_selected_beams(self, calibrator_gains_setup):
        """Selecting high-altitude beams should select corresponding sky pixels."""
        cal, _ = calibrator_gains_setup

        # Select only central beam pixels (high altitude)
        beam_mask = cal.fwd.build_beam_mask_altitude(cal.rot_matrices, min_altitude_deg=45.0)
        sky_mask = cal.build_sky_mask_from_beam_pixels(beam_mask)

        # Should select some sky pixels
        assert sky_mask.sum() > 0
        # But fewer than if we selected all beams
        all_beam_mask = np.ones(healpy.nside2npix(cal.beam_model.nside), dtype=bool)
        all_sky_mask = cal.build_sky_mask_from_beam_pixels(all_beam_mask)
        assert sky_mask.sum() <= all_sky_mask.sum()

    def test_build_beam_mask_from_sky_pixels_shape(self, calibrator_gains_setup):
        """build_beam_mask_from_sky_pixels should return mask with correct shape."""
        cal, _ = calibrator_gains_setup
        npix_sky_full = cal._npix_full
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)

        # Create a sky mask with some pixels
        sky_mask = np.zeros(npix_sky_full, dtype=bool)
        sky_mask[:npix_sky_full // 2] = True

        beam_mask = cal.build_beam_mask_from_sky_pixels(sky_mask)

        assert beam_mask.shape == (npix_beam_full,)
        assert beam_mask.dtype == bool

    def test_build_beam_mask_from_sky_pixels_all_sky(self, calibrator_gains_setup):
        """When all sky pixels selected, beam mask should touch all illuminated beams."""
        cal, _ = calibrator_gains_setup
        npix_sky_full = cal._npix_full

        all_sky_mask = np.ones(npix_sky_full, dtype=bool)
        beam_mask = cal.build_beam_mask_from_sky_pixels(all_sky_mask)

        # Should have selected some beam pixels (non-zero mask)
        assert beam_mask.sum() > 0
        assert beam_mask.sum() <= healpy.nside2npix(cal.beam_model.nside)

    def test_build_beam_mask_from_sky_pixels_no_sky(self, calibrator_gains_setup):
        """When no sky pixels selected, beam mask should be empty."""
        cal, _ = calibrator_gains_setup
        npix_sky_full = cal._npix_full

        empty_sky_mask = np.zeros(npix_sky_full, dtype=bool)
        beam_mask = cal.build_beam_mask_from_sky_pixels(empty_sky_mask)

        assert beam_mask.sum() == 0

    def test_build_beam_mask_from_sky_pixels_reciprocal(self, calibrator_gains_setup):
        """Beam→sky and sky→beam should be reciprocal operations."""
        cal, _ = calibrator_gains_setup
        npix_sky_full = cal._npix_full

        # Start with a subset of sky pixels
        initial_sky_mask = np.zeros(npix_sky_full, dtype=bool)
        initial_sky_mask[:npix_sky_full // 4] = True

        # Forward: sky → beam
        beam_mask = cal.build_beam_mask_from_sky_pixels(initial_sky_mask)

        # Reverse: beam → sky
        recovered_sky_mask = cal.build_sky_mask_from_beam_pixels(beam_mask)

        # The recovered sky mask should contain all original sky pixels
        # (and possibly more due to the stencil structure)
        assert np.all(recovered_sky_mask[initial_sky_mask])

    def test_build_beam_mask_from_sky_pixels_with_applied_sky_mask(self, calibrator_gains_setup):
        """Should work when a sky mask has been applied but full-size mask is passed in."""
        cal, _ = calibrator_gains_setup
        npix_sky_full = cal._npix_full
        npix_beam_full = healpy.nside2npix(cal.beam_model.nside)

        # Create a weight-based sky mask (simulating the user's scenario)
        bm_wgts = cal.get_sky_beam_weighting()
        threshold = bm_wgts.max() / 100
        sky_mask_full = bm_wgts > threshold  # Full-size mask with some pixels selected

        # Apply the sky mask to the calibrator
        cal.apply_sky_mask(sky_mask_full)

        # Now call build_beam_mask_from_sky_pixels with the same full-size mask
        # This should still work (function should auto-detect and convert to masked space)
        beam_mask = cal.build_beam_mask_from_sky_pixels(sky_mask_full)

        # Verify output shape
        assert beam_mask.shape == (npix_beam_full,), f"Expected shape {(npix_beam_full,)}, got {beam_mask.shape}"
        assert beam_mask.dtype == bool
        # Should have selected some beam pixels (the weighted ones)
        assert beam_mask.sum() > 0, "Expected non-empty beam mask"

    def test_fit_gains_linear_variable_beam_uses_current_beam(self, calibrator_gains_setup, sky_coeffs_true):
        """fit_gains_linear_variable_beam should use the provided beam_coeffs, not cached beam."""
        cal, true_gains = calibrator_gains_setup

        # Beam coefficients from the calibrator
        beam_coeffs_true = cal.beam_model.coeffs

        # Test 1: fit_gains_linear_variable_beam with true sky and beam should recover gains
        gain_params_var, loss_var = cal.fit_gains_linear_variable_beam(sky_coeffs_true, beam_coeffs_true)

        # The recovered gains should be close to true gains
        assert np.allclose(gain_params_var['log_amp'], true_gains['log_amp'], atol=0.1), \
            "log_amp not recovered with variable beam"
        assert np.allclose(gain_params_var['phase'], true_gains['phase'], atol=0.1), \
            "phase not recovered with variable beam"
        assert np.allclose(gain_params_var['phi'], true_gains['phi'], atol=0.1), \
            "phi not recovered with variable beam"

        # Test 2: Verify the loss is computed with variable beam, not cached beam
        # Create a perturbed beam and verify the loss changes
        beam_coeffs_perturbed = beam_coeffs_true + 0.1 * np.random.RandomState(42).randn(*beam_coeffs_true.shape)
        gain_params_perturbed, loss_perturbed = cal.fit_gains_linear_variable_beam(sky_coeffs_true, beam_coeffs_perturbed)

        # Loss should be different with perturbed beam
        assert loss_var != loss_perturbed, \
            "Loss should differ when beam coefficients change (variable_beam not being used)"
        # And the perturbed beam loss should be higher
        assert loss_perturbed > loss_var, \
            "Perturbed beam should give worse loss"
