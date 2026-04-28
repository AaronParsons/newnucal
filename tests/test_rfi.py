import numpy as np

from newnucal.rfi import (
    prepare_initial_channel_weights,
    channel_chi2_statistic,
    fit_soft_channel_weights,
    fit_soft_channel_weights_jax,
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


def test_fit_soft_channel_weights_respects_prior_bounds():
    rng = np.random.default_rng(0)
    ntime, nfreq, nbls = 4, 8, 3
    resid = (0.01 * rng.normal(size=(ntime, nfreq, nbls))
             + 1j * 0.01 * rng.normal(size=(ntime, nfreq, nbls))).astype(np.complex64)
    # Inject one clearly bad channel across all times/baselines.
    resid[:, 3, :] += 5.0 + 3.0j

    # Prior weights from external flagging: act as upper bound caps
    prior = np.ones((ntime, nfreq), dtype=np.float32) * 0.5
    prior[:, 6] = 0.1  # channel 6 is more aggressively flagged

    weights, diag = fit_soft_channel_weights(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        prior_weights=prior,
        regularization=1.0,
        regularization_power=2.0,
        min_weight=0.01,
        max_weight=1.0,
    )

    assert weights.shape == (ntime, nfreq)
    assert diag["residual_power_f"].shape == (nfreq,)

    # Prior acts as upper bound: no weight should exceed prior
    assert np.all(weights <= prior + 1e-6), "Weights must not exceed prior bounds"

    # All weights should be in valid range
    assert np.all(weights >= 0.01)  # min_weight
    assert np.all(weights <= 1.0)   # max_weight

    # Hard-flagged channel should be at or below its prior
    w_channel6 = float(np.mean(weights[:, 6]))
    assert w_channel6 <= prior[0, 6] + 1e-6

    # Bad channel (3) should be downweighted
    w_channel3 = float(np.mean(weights[:, 3]))
    assert w_channel3 < 0.4

    # Diagnostics should be valid
    assert diag["optimization_success"]


def test_fit_soft_channel_weights_jax_produces_valid_weights():
    """Verify JAX version produces valid, reasonable weights."""
    rng = np.random.default_rng(42)
    ntime, nfreq, nbls = 4, 8, 3
    resid = (0.01 * rng.normal(size=(ntime, nfreq, nbls))
             + 1j * 0.01 * rng.normal(size=(ntime, nfreq, nbls))).astype(np.complex64)
    # Inject one clearly bad channel across all times/baselines.
    resid[:, 3, :] += 5.0 + 3.0j

    prior = np.ones((ntime, nfreq), dtype=np.float32) * 0.5
    prior[:, 6] = 0.1

    # Run JAX version
    weights_jax, diag_jax = fit_soft_channel_weights_jax(
        residual=resid,
        inv_noise_var=np.ones((ntime, nfreq), dtype=np.float32),
        prior_weights=prior,
        regularization=1.0,
        regularization_power=2.0,
        min_weight=0.01,
        max_weight=1.0,
        use_jax=True,
    )

    # Convert JAX arrays to NumPy for inspection
    weights_jax_np = np.asarray(weights_jax)

    # Check that weights are valid
    assert weights_jax_np.shape == (ntime, nfreq)
    assert np.all(weights_jax_np >= 0.01)
    assert np.all(weights_jax_np <= 1.0)
    # Check that prior is respected
    assert np.all(weights_jax_np <= prior + 1e-6)
    # Check that bad channel is downweighted
    assert float(np.mean(weights_jax_np[:, 3])) < 0.4
    # Check that diagnostics are present
    assert "final_loss" in diag_jax
    assert "num_iterations" in diag_jax
    assert diag_jax["num_iterations"] > 0
