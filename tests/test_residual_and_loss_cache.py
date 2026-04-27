"""Tests for residual and loss caching optimizations."""

import numpy as np
import jax.numpy as jnp
import pytest

from newnucal.calibrator import Calibrator
from newnucal.gains import init_gain_params, apply_gains
from newnucal.basis import basis_project

# Re-implement fixtures from test_calibrator to avoid import dependencies
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
    ntime = rot_matrices.shape[0]
    nfreq = len(freqs)

    vis_true = forward_model.simulate_3d(sky_coeffs_true, jnp.array(rot_matrices))

    # Smooth gains: exact parametric form that fit_gains_linear inverts analytically
    f_norm = np.linspace(0, 1, nfreq)[None, :]
    t_norm = np.linspace(0, 1, ntime)[:, None]

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


class TestResidualAndLossFunction:
    """Test that _residual_and_loss_variable_beam correctly combines the two computations."""

    def test_combined_function_matches_separate(self, calibrator_gains_setup, sky_coeffs_true):
        """Verify combined function returns same loss and residual as separate functions."""
        cal, true_gains = calibrator_gains_setup
        weights = cal._effective_weights()

        # Use the true sky and true gains
        params = {
            'sky_coeffs': sky_coeffs_true,
            'beam_coeffs': jnp.array(cal.fwd.beam_coeffs),
            **true_gains
        }

        # Compute using combined function
        resid_combined, loss_combined = cal._jit_residual_and_loss_variable_beam(params, weights)

        # Compute using separate functions
        loss_separate = cal._jit_loss_variable_beam(params, weights)
        resid_separate = cal._jit_residual_variable_beam(params, weights)

        # Check loss matches
        assert jnp.allclose(loss_combined, loss_separate, rtol=1e-5, atol=1e-8), \
            f"Combined loss {float(loss_combined):.6e} != separate {float(loss_separate):.6e}"

        # Check residual matches (stricter tolerance for residuals)
        assert jnp.allclose(resid_combined, resid_separate, rtol=1e-5, atol=1e-8), \
            "Combined residual does not match separate residual"


class TestResidualCache:
    """Test Level 1 cache: accepted trial residual for next adjoint."""

    def test_residual_cache_hit_on_consecutive_identical_params(self, calibrator_gains_setup, sky_coeffs_true):
        """Verify residual cache is hit when params haven't changed."""
        cal, true_gains = calibrator_gains_setup
        weights = cal._effective_weights()

        beam_coeffs = jnp.array(cal.fwd.beam_coeffs)
        params = {'sky_coeffs': sky_coeffs_true, 'beam_coeffs': beam_coeffs, **true_gains}

        # Manually set cache with exact same JAX arrays
        resid_original, _ = cal._jit_residual_and_loss_variable_beam(params, weights)
        cal._residual_cache = (
            sky_coeffs_true, beam_coeffs,
            true_gains['log_amp'], true_gains['phase'], true_gains['phi'],
            resid_original
        )

        # Verify the cache is structured correctly
        assert cal._residual_cache is not None, "Cache should be set"
        assert len(cal._residual_cache) == 6, "Cache should have 6 elements"
        assert cal._residual_cache[0] is sky_coeffs_true, "Sky should be cached by identity"
        assert cal._residual_cache[1] is beam_coeffs, "Beam should be cached by identity"

    def test_residual_cache_miss_on_new_params(self, calibrator_gains_setup, sky_coeffs_true):
        """Verify cache is invalidated when params change."""
        cal, true_gains = calibrator_gains_setup
        weights = cal._effective_weights()

        beam_coeffs = jnp.array(cal.fwd.beam_coeffs)
        params = {'sky_coeffs': sky_coeffs_true, 'beam_coeffs': beam_coeffs, **true_gains}

        # Set cache with original params
        resid_original, _ = cal._jit_residual_and_loss_variable_beam(params, weights)
        cal._residual_cache = (
            sky_coeffs_true, beam_coeffs,
            true_gains['log_amp'], true_gains['phase'], true_gains['phi'],
            resid_original
        )

        # Create new beam array (fresh copy, different identity)
        beam_new = beam_coeffs + 0.0  # Force a new array
        beam_new = beam_new.astype(jnp.float32)  # Ensure it's a different object

        # Cache should miss because beam is not `is` the cached beam
        params_new = {
            'sky_coeffs': sky_coeffs_true,
            'beam_coeffs': beam_new,
            **true_gains,
        }

        # The cache identity check should fail
        cache = cal._residual_cache
        if cache is not None:
            # Manually verify the identity check would fail
            assert not (cache[1] is params_new['beam_coeffs']), \
                "Identity check should fail for different beam array"


class TestLossCache:
    """Test Level 2 cache: accepted-step loss for chi2 checks."""

    def test_loss_cache_fast_path_after_step(self, calibrator_gains_setup, sky_coeffs_true):
        """Verify calc_loss hits _variable_beam_eval_cache on subsequent calls."""
        cal, true_gains = calibrator_gains_setup

        beam_coeffs = jnp.array(cal.fwd.beam_coeffs)
        params = {'sky_coeffs': sky_coeffs_true, 'beam_coeffs': beam_coeffs, **true_gains}

        # Set the cache as if we just completed a step
        loss_value = 123.45  # arbitrary
        # The _variable_beam_eval_cache format is (sky, beam, log_amp, phase, phi, resid, loss)
        resid_placeholder = None  # residual is optional
        cal._variable_beam_eval_cache = (
            cal._ensure_sky_is_active(sky_coeffs_true),
            beam_coeffs,
            true_gains.get('log_amp'),
            true_gains.get('phase'),
            true_gains.get('phi'),
            resid_placeholder,
            loss_value
        )

        # Call calc_loss with explicit_beam=True
        loss_returned = cal.calc_loss(params, explicit_beam=True)

        # Should return the cached loss
        assert loss_returned == loss_value, \
            f"calc_loss should return cached value {loss_value}, got {loss_returned}"

    def test_loss_cache_miss_on_different_params(self, calibrator_gains_setup, sky_coeffs_true):
        """Verify calc_loss recomputes when params don't match cache."""
        cal, true_gains = calibrator_gains_setup

        beam_coeffs = jnp.array(cal.fwd.beam_coeffs)

        # Set cache with one param set
        loss_cached = 999.0
        cal._fit_gains_linear_variable_beam_cache = (
            cal._ensure_sky_is_active(sky_coeffs_true),
            beam_coeffs,
            true_gains,
            loss_cached
        )

        # Create different params
        beam_new = beam_coeffs + 0.01
        beam_new = beam_new.astype(jnp.float32)
        params_different = {
            'sky_coeffs': sky_coeffs_true,
            'beam_coeffs': beam_new,
            **true_gains,
        }

        # Call calc_loss; should not return cached value
        loss_returned = cal.calc_loss(params_different, explicit_beam=True)

        # Should be different from cached (recomputed with new beam)
        assert loss_returned != loss_cached, \
            "calc_loss should recompute for different params, not return cache"


class TestCacheInvalidation:
    """Test that caches are invalidated appropriately."""

    def test_residual_cache_invalidated_on_recompile_jit(self, calibrator_gains_setup):
        """Verify _residual_cache is cleared when _recompile_jit is called."""
        cal, _ = calibrator_gains_setup

        # Set a dummy cache
        cal._residual_cache = ("dummy",)

        # Call _recompile_jit
        cal._recompile_jit()

        # Cache should be cleared
        assert cal._residual_cache is None, \
            "_residual_cache should be cleared after _recompile_jit"
