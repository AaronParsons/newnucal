import numpy as np
import jax.numpy as jnp
import pytest
from newnucal.gains import apply_gains, init_gain_params


NFREQ = 16
NBLS = 10
NTIME = 3


@pytest.fixture
def dummy_vis():
    rng = np.random.default_rng(42)
    return jnp.array(
        rng.standard_normal((NTIME, NFREQ, NBLS))
        + 1j * rng.standard_normal((NTIME, NFREQ, NBLS)),
        dtype=jnp.complex64,
    )


@pytest.fixture
def dummy_bls():
    rng = np.random.default_rng(0)
    return jnp.array(rng.uniform(-100, 100, (NBLS, 3)), dtype=jnp.float32)


def test_zero_gains_unity(dummy_vis, dummy_bls):
    """Zero gain params → gain factor = 1 → output equals input."""
    params = init_gain_params(NTIME, NFREQ)
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert jnp.allclose(out, dummy_vis, atol=1e-5)


def test_log_amp_scales_amplitude(dummy_vis, dummy_bls):
    """log_amp = log(2) → amplitude doubles at every time and frequency."""
    params = init_gain_params(NTIME, NFREQ)
    params = {**params, "log_amp": jnp.full((NTIME, NFREQ), np.log(2.0), dtype=jnp.float32)}
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert jnp.allclose(jnp.abs(out), 2.0 * jnp.abs(dummy_vis), rtol=1e-4)


def test_phase_rotates(dummy_vis, dummy_bls):
    """phase = π/4 → all visibilities rotated by π/4 in phase at every time."""
    params = init_gain_params(NTIME, NFREQ)
    phs = float(np.pi / 4)
    params = {**params, "phase": jnp.full((NTIME, NFREQ), phs, dtype=jnp.float32)}
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    phase_diff = jnp.angle(out / dummy_vis)
    assert jnp.allclose(phase_diff, phs, atol=0.01)


def test_output_shape(dummy_vis, dummy_bls):
    params = init_gain_params(NTIME, NFREQ)
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert out.shape == dummy_vis.shape


def test_phi_baseline_dependence(dummy_bls):
    """Phase gradient should create different phases for different baselines."""
    vis = jnp.ones((NTIME, NFREQ, NBLS), dtype=jnp.complex64)
    params = init_gain_params(NTIME, NFREQ)
    phi_val = jnp.full((NTIME, 2, NFREQ), 1e-3, dtype=jnp.float32)
    params = {**params, "phi": phi_val}
    out = apply_gains(vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    # Different baselines → different phases at each time
    phases = jnp.angle(out[0, 0])  # first time, first frequency
    assert not jnp.allclose(phases, phases[0])


def test_per_time_independence(dummy_vis, dummy_bls):
    """Different log_amp at each time produces different amplitude scaling per time."""
    params = init_gain_params(NTIME, NFREQ)
    log_amp = jnp.array(
        np.arange(NTIME, dtype=np.float32)[:, None] * np.ones((1, NFREQ), dtype=np.float32)
    )
    params = {**params, "log_amp": log_amp}
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    for t in range(NTIME):
        expected_scale = float(np.exp(t))
        ratio = jnp.abs(out[t]) / (jnp.abs(dummy_vis[t]) + 1e-12)
        assert jnp.allclose(ratio, expected_scale, rtol=1e-4)
