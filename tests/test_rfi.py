import numpy as np

from newnucal.rfi import (
    prepare_initial_channel_weights,
    channel_chi2_statistic,
    fit_channel_weights_local_chi2_exponential,
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


def test_channel_chi2_statistic_respects_visibility_weights():
    resid = np.zeros((2, 4, 3), dtype=np.complex64)
    resid[:, 1, 0] = 100.0 + 0.0j
    resid[:, 1, 1:] = 2.0 + 0.0j
    visibility_weights = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    chi2 = channel_chi2_statistic(
        resid,
        inv_noise_var=np.ones((2, 4), dtype=np.float32),
        visibility_weights=visibility_weights,
    )

    # The large masked baseline should not contribute; mean(|2|^2, |2|^2) = 4.
    np.testing.assert_allclose(chi2[:, 1], 4.0)


def test_fit_channel_weights_local_chi2_exponential_detects_outliers():
    """Verify local chi² exponential approach detects and downweights outliers."""
    rng = np.random.default_rng(0)
    ntime, nfreq, nbls = 4, 8, 3
    resid = (0.01 * rng.normal(size=(ntime, nfreq, nbls))
             + 1j * 0.01 * rng.normal(size=(ntime, nfreq, nbls))).astype(np.complex64)
    # Inject one clearly bad channel (very strong outlier in chi²)
    resid[:, 3, :] += 20.0 + 15.0j

    weights, diag = fit_channel_weights_local_chi2_exponential(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        old_weights=np.ones(nfreq, dtype=np.float32),
        prior_weights=None,
        min_weight=0.05,
        max_weight=1.0,
        smooth_width_chans=5,
        threshold_ratio=2.0,
        gamma=1.0,
        alpha_down=0.30,
        alpha_up=0.75,
        max_drop_per_update=0.50,
    )

    assert weights.shape == (ntime, nfreq)
    assert "chi2_f" in diag
    assert "log_ratio_f" in diag
    assert "weights_f" in diag

    # All weights should be in valid range
    assert np.all(weights >= 0.05)  # min_weight
    assert np.all(weights <= 1.0)   # max_weight

    # Bad channel (3) should be downweighted compared to normal channels
    w_channel3 = float(np.mean(weights[:, 3]))
    # Normal channels should retain higher weight
    normal_weights = [np.mean(weights[:, i]) for i in [0, 1, 2, 4, 5, 6, 7]]
    avg_normal = np.mean(normal_weights)
    assert w_channel3 < avg_normal, f"Expected bad channel ({w_channel3:.3f}) to be lower than normal ({avg_normal:.3f})"


def test_fit_channel_weights_local_chi2_exponential_respects_min_max():
    """Verify weights stay within min/max bounds."""
    rng = np.random.default_rng(42)
    ntime, nfreq, nbls = 3, 6, 3
    resid = (0.01 * rng.normal(size=(ntime, nfreq, nbls))
             + 1j * 0.01 * rng.normal(size=(ntime, nfreq, nbls))).astype(np.complex64)
    # Make all channels very bad
    resid[:, :, :] += 10.0 + 5.0j

    min_w = 0.10
    max_w = 0.95

    weights, _ = fit_channel_weights_local_chi2_exponential(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        old_weights=np.ones(nfreq, dtype=np.float32),
        prior_weights=None,
        min_weight=min_w,
        max_weight=max_w,
        smooth_width_chans=3,
        threshold_ratio=2.0,
        gamma=0.75,
        alpha_down=0.20,
        alpha_up=0.75,
        max_drop_per_update=0.20,
    )

    # Check bounds are respected
    assert np.all(weights >= min_w - 1e-6), f"Weights below min: {np.min(weights)}"
    assert np.all(weights <= max_w + 1e-6), f"Weights above max: {np.max(weights)}"
