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
    params = init_gain_params(NFREQ)
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert jnp.allclose(out, dummy_vis, atol=1e-5)


def test_log_amp_scales_amplitude(dummy_vis, dummy_bls):
    """log_amp = log(2) → amplitude doubles."""
    params = init_gain_params(NFREQ)
    params = {**params, "log_amp": jnp.full(NFREQ, np.log(2.0), dtype=jnp.float32)}
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert jnp.allclose(jnp.abs(out), 2.0 * jnp.abs(dummy_vis), rtol=1e-4)


def test_phase_rotates(dummy_vis, dummy_bls):
    """phase = π/4 → all visibilities rotated by π/4 in phase."""
    params = init_gain_params(NFREQ)
    phs = float(np.pi / 4)
    params = {**params, "phase": jnp.full(NFREQ, phs, dtype=jnp.float32)}
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    # Phase difference via ratio avoids ±π wrapping issues
    phase_diff = jnp.angle(out / dummy_vis)
    assert jnp.allclose(phase_diff, phs, atol=0.01)


def test_output_shape_3d(dummy_vis, dummy_bls):
    params = init_gain_params(NFREQ)
    out = apply_gains(dummy_vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert out.shape == dummy_vis.shape


def test_output_shape_2d(dummy_bls):
    rng = np.random.default_rng(7)
    vis_2d = jnp.array(
        rng.standard_normal((NFREQ, NBLS)) + 1j * rng.standard_normal((NFREQ, NBLS)),
        dtype=jnp.complex64,
    )
    params = init_gain_params(NFREQ)
    out = apply_gains(vis_2d, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    assert out.shape == (NFREQ, NBLS)


def test_phi_baseline_dependence(dummy_bls):
    """Phase gradient should create different phases for different baselines."""
    rng = np.random.default_rng(1)
    vis = jnp.ones((NFREQ, NBLS), dtype=jnp.complex64)
    params = init_gain_params(NFREQ)
    phi_val = jnp.full((2, NFREQ), 1e-3, dtype=jnp.float32)
    params = {**params, "phi": phi_val}
    out = apply_gains(vis, params["log_amp"], params["phase"], params["phi"], dummy_bls)
    # Different baselines → different phases (unless all baselines are identical)
    phases = jnp.angle(out[0])  # at first frequency
    assert not jnp.allclose(phases, phases[0])  # at least two differ
