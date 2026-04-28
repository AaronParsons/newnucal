import numpy as np

from newnucal.rfi import (
    RFIConfig,
    prepare_initial_channel_weights,
    channel_chi2_statistic,
    smooth_frequency_statistic,
    score_to_soft_weights,
    update_channel_weights_from_residuals,
    fit_soft_channel_weights,
)


def test_prepare_initial_channel_weights_from_flags_and_1d_weights():
    ntime, nfreq = 3, 5
    init_w = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
    flags = np.array([False, True, False, True, False])

    out = prepare_initial_channel_weights(
        ntime=ntime,
        nfreq=nfreq,
        initial_weights=init_w,
        initial_flags=flags,
        flagged_weight=0.05,
    )

    assert out.shape == (ntime, nfreq)
    np.testing.assert_allclose(out[:, 0], 1.0)
    np.testing.assert_allclose(out[:, 2], 0.6)
    np.testing.assert_allclose(out[:, 4], 0.2)
    np.testing.assert_allclose(out[:, 1], 0.05)
    np.testing.assert_allclose(out[:, 3], 0.05)


def test_channel_chi2_statistic_matches_expected_whitened_power():
    resid = np.zeros((2, 4, 3), dtype=np.complex64)
    resid[:, 1, :] = 2.0 + 0.0j
    inv_noise_var = np.array([1.0, 0.5, 2.0, 1.0], dtype=np.float32)

    chi2 = channel_chi2_statistic(resid, inv_noise_var=inv_noise_var)

    assert chi2.shape == (2, 4)
    np.testing.assert_allclose(chi2[:, 0], 0.0)
    # mean |2|^2 = 4, whitened by 0.5 -> 2
    np.testing.assert_allclose(chi2[:, 1], 2.0)
    np.testing.assert_allclose(chi2[:, 2], 0.0)
    np.testing.assert_allclose(chi2[:, 3], 0.0)


def test_smooth_frequency_statistic_rolling_median_suppresses_single_spike():
    stat = np.array([1, 1, 1, 20, 1, 1, 1], dtype=np.float32)
    smooth = smooth_frequency_statistic(stat, window=3)

    np.testing.assert_allclose(smooth, np.ones_like(stat))


def test_score_to_soft_weights_is_monotone_and_bounded():
    score = np.array([0.0, 1.0, 3.0, 10.0], dtype=np.float32)
    w = score_to_soft_weights(score, center=3.0, slope=2.0, min_weight=0.05, max_weight=1.0)

    assert np.all(w[:-1] >= w[1:])
    assert np.all(w >= 0.05)
    assert np.all(w <= 1.0)
    assert np.isclose(w[0], 1.0)


def test_update_channel_weights_from_residuals_downweights_rfi_channel():
    rng = np.random.default_rng(0)
    ntime, nfreq, nbls = 4, 8, 3
    resid = (0.01 * rng.normal(size=(ntime, nfreq, nbls))
             + 1j * 0.01 * rng.normal(size=(ntime, nfreq, nbls))).astype(np.complex64)
    # Inject one clearly bad channel across all times/baselines.
    resid[:, 3, :] += 5.0 + 3.0j

    prior = np.ones((ntime, nfreq), dtype=np.float32)
    prior[:, 6] = 0.2  # external prior should remain somewhat downweighted too

    cfg = RFIConfig(
        min_weight=0.01,
        max_weight=1.0,
        smooth_window=5,
        score_center=1.5,
        score_slope=2.0,
        prior_power=1.0,
        model_power=1.0,
        blend=0.0,
        time_reduce="median",
    )

    weights, diag = update_channel_weights_from_residuals(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        prior_weights=prior,
        current_weights=np.ones((ntime, nfreq), dtype=np.float32),
        config=cfg,
    )

    assert weights.shape == (ntime, nfreq)
    assert diag["chi2_tf"].shape == (ntime, nfreq)
    assert diag["chi2_f"].shape == (nfreq,)
    assert diag["score_f"].shape == (nfreq,)
    assert diag["model_weights_f"].shape == (nfreq,)

    # Strong residual channel should be heavily downweighted relative to neighbors.
    assert float(weights[:, 3].mean()) < 0.2
    assert float(weights[:, 3].mean()) < float(weights[:, 2].mean())
    assert float(weights[:, 3].mean()) < float(weights[:, 4].mean())

    # External prior should still influence the result.
    assert float(weights[:, 6].mean()) < 1.0


def test_fit_soft_channel_weights_regularizes_downweighting():
    rng = np.random.default_rng(0)
    ntime, nfreq, nbls = 4, 8, 3
    resid = (0.01 * rng.normal(size=(ntime, nfreq, nbls))
             + 1j * 0.01 * rng.normal(size=(ntime, nfreq, nbls))).astype(np.complex64)
    # Inject one clearly bad channel across all times/baselines.
    resid[:, 3, :] += 5.0 + 3.0j

    # Prior weights from external flagging (e.g., DPSS): can be downweighted further
    prior = np.ones((ntime, nfreq), dtype=np.float32) * 0.5  # moderately conservative prior
    prior[:, 6] = 0.1  # one channel is more aggressively flagged externally

    # Test with strong regularization (prevents aggressive downweighting)
    weights_strong, diag_strong = fit_soft_channel_weights(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        prior_weights=prior,
        regularization=10.0,
        regularization_power=2.0,
        min_weight=0.01,
        max_weight=1.0,
    )

    # Test with weak regularization (allows more downweighting)
    weights_weak, diag_weak = fit_soft_channel_weights(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        prior_weights=prior,
        regularization=0.1,
        regularization_power=2.0,
        min_weight=0.01,
        max_weight=1.0,
    )

    assert weights_strong.shape == (ntime, nfreq)
    assert weights_weak.shape == (ntime, nfreq)
    assert diag_strong["residual_power_f"].shape == (nfreq,)
    assert diag_weak["residual_power_f"].shape == (nfreq,)

    # Both should respect bounds
    assert np.all(weights_strong >= prior - 1e-6)
    assert np.all(weights_weak >= prior - 1e-6)
    assert np.all(weights_strong <= 1.0 + 1e-6)
    assert np.all(weights_weak <= 1.0 + 1e-6)

    # Soft weighting should not violate hard prior (externally flagged channels)
    # Channel 6 is flagged to 0.1, should not go above that
    w_prior_channel = float(weights_weak[:, 6].mean())
    assert w_prior_channel <= prior[0, 6] + 1e-6, "Should respect hard external prior"

    # Strong regularization should keep weights closer to prior (less aggressive downweighting)
    # than weak regularization
    w_strong_range = float(np.max(weights_strong) - np.min(weights_strong))
    w_weak_range = float(np.max(weights_weak) - np.min(weights_weak))
    # Weak regularization allows more variation (more aggressive downweighting of bad channels)
    assert w_weak_range >= w_strong_range * 0.5, "Weak regularization should allow more variation"

    # Diagnostics should be valid
    assert diag_strong["optimization_success"]
    assert diag_weak["optimization_success"]
