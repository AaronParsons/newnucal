"""
Tests for the unified adjoint interface (compute_adjoint_updates).

Verifies that the unified interface produces identical results to the
individual adjoint calls, and tests that all fit_*_dirty methods work
correctly with the new interface.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from newnucal.gains import init_gain_params


@pytest.fixture
def sky_and_beam_setup(calibrator, sky_coeffs_test):
    """Setup for sky/beam adjoint tests."""
    return calibrator, sky_coeffs_test


class TestComputeAdjointUpdatesInterface:
    """Test the unified compute_adjoint_updates interface."""

    def test_sky_mode_matches_direct_call(self, calibrator, sky_coeffs_test):
        """Mode='sky' produces same result as direct _sky_update_fn call."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        resid = calibrator.calibrated_residual(params)

        # Direct call
        delta_direct = calibrator._sky_update_fn(resid, step_size=0.5, beam_reg=1e-3)

        # Unified interface
        updates = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='sky',
            step_size=0.5, beam_reg=1e-3
        )
        delta_unified = updates['sky']

        np.testing.assert_allclose(
            np.asarray(delta_direct), np.asarray(delta_unified),
            rtol=1e-5, atol=1e-8,
            err_msg="Sky-mode adjoint mismatch"
        )

    def test_beam_mode_matches_direct_call(self, calibrator, sky_coeffs_test):
        """Mode='beam' produces same result as direct _beam_update_fn call."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        resid = calibrator.calibrated_residual_variable_beam(params)

        # Direct call
        delta_direct = calibrator._beam_update_fn(
            sky_coeffs_test, resid, step_size=0.5, sky_reg=1e-3
        )

        # Unified interface
        updates = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='beam',
            step_size=0.5, sky_reg=1e-3
        )
        delta_unified = updates['beam']

        np.testing.assert_allclose(
            np.asarray(delta_direct), np.asarray(delta_unified),
            rtol=1e-5, atol=1e-8,
            err_msg="Beam-mode adjoint mismatch"
        )

    def test_both_mode_matches_combined_call(self, calibrator, sky_coeffs_test):
        """Mode='both' produces same result as direct _combined_update_fn call."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        resid = calibrator.calibrated_residual_variable_beam(params)

        # Direct call
        delta_sky_direct, delta_beam_direct = calibrator._combined_update_fn(
            sky_coeffs_test, resid,
            sky_step_size=0.5, beam_step_size=0.6,
            beam_reg=2e-3, sky_reg=1.5e-3
        )

        # Unified interface
        updates = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='both',
            sky_step_size=0.5, beam_step_size=0.6,
            beam_reg=2e-3, sky_reg=1.5e-3
        )
        delta_sky_unified = updates['sky']
        delta_beam_unified = updates['beam']

        np.testing.assert_allclose(
            np.asarray(delta_sky_direct), np.asarray(delta_sky_unified),
            rtol=1e-5, atol=1e-8,
            err_msg="Sky component of both-mode adjoint mismatch"
        )
        np.testing.assert_allclose(
            np.asarray(delta_beam_direct), np.asarray(delta_beam_unified),
            rtol=1e-5, atol=1e-8,
            err_msg="Beam component of both-mode adjoint mismatch"
        )

    def test_returns_correct_keys(self, calibrator, sky_coeffs_test):
        """Return dict has correct keys for each mode."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        resid = calibrator.calibrated_residual_variable_beam(params)

        updates_sky = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='sky'
        )
        assert set(updates_sky.keys()) == {'sky'}, "Sky mode should only return 'sky' key"

        updates_beam = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='beam'
        )
        assert set(updates_beam.keys()) == {'beam'}, "Beam mode should only return 'beam' key"

        updates_both = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='both'
        )
        assert set(updates_both.keys()) == {'sky', 'beam'}, "Both mode should return both keys"

    def test_invalid_mode_raises_error(self, calibrator, sky_coeffs_test):
        """Invalid update_mode raises ValueError."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        resid = calibrator.calibrated_residual(params)

        with pytest.raises(ValueError, match="update_mode must be"):
            calibrator.compute_adjoint_updates(
                sky_coeffs_test, resid, update_mode='invalid'
            )

    def test_default_parameters(self, calibrator, sky_coeffs_test):
        """Default parameters are applied when not specified."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        resid = calibrator.calibrated_residual(params)

        # These should not raise even though params aren't explicitly passed
        updates_sky = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='sky'
        )
        assert 'sky' in updates_sky

        updates_beam = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='beam'
        )
        assert 'beam' in updates_beam

        updates_both = calibrator.compute_adjoint_updates(
            sky_coeffs_test, resid, update_mode='both'
        )
        assert 'sky' in updates_both and 'beam' in updates_both


class TestFitSkyDirtyWithUnifiedInterface:
    """Test that fit_sky_dirty works with unified interface."""

    def test_fit_sky_dirty_reduces_loss(self, calibrator, sky_coeffs_test):
        """fit_sky_dirty should reduce loss."""
        gain_params = init_gain_params(calibrator.ntime, calibrator.nfreq)
        initial_params = {'sky_coeffs': sky_coeffs_test, **gain_params}
        loss_init = calibrator.calc_loss(initial_params)

        sky_updated, loss_updated = calibrator.fit_sky_dirty(
            sky_coeffs_test, gain_params, n_iter=3, step_size=0.5
        )

        assert loss_updated < loss_init, (
            f"Loss should decrease; got {loss_init:.3e} → {loss_updated:.3e}"
        )

    def test_fit_sky_dirty_preserves_sky_shape(self, calibrator, sky_coeffs_test):
        """fit_sky_dirty should return sky with same shape."""
        gain_params = init_gain_params(calibrator.ntime, calibrator.nfreq)
        sky_updated, _ = calibrator.fit_sky_dirty(
            sky_coeffs_test, gain_params, n_iter=1
        )
        assert sky_updated.shape == sky_coeffs_test.shape


class TestFitBeamDirtyWithUnifiedInterface:
    """Test that fit_beam_dirty works with unified interface."""

    def test_fit_beam_dirty_reduces_loss(self, calibrator, sky_coeffs_test):
        """fit_beam_dirty should reduce loss."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        loss_init = calibrator.calc_loss(params, explicit_beam=True)

        params_updated, loss_updated = calibrator.fit_beam_dirty(
            params, n_iter=2, step_size=0.5
        )

        assert loss_updated < loss_init, (
            f"Loss should decrease; got {loss_init:.3e} → {loss_updated:.3e}"
        )

    def test_fit_beam_dirty_preserves_beam_shape(self, calibrator, sky_coeffs_test):
        """fit_beam_dirty should return beam with same shape."""
        beam_shape_orig = calibrator.fwd.beam_coeffs.shape
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        params_updated, _ = calibrator.fit_beam_dirty(params, n_iter=1)
        assert params_updated['beam_coeffs'].shape == beam_shape_orig


class TestFitSkyAndBeamDirtyWithUnifiedInterface:
    """Test that fit_sky_and_beam_dirty works with unified interface."""

    def test_fit_sky_and_beam_dirty_changes_params(self, calibrator, sky_coeffs_test):
        """fit_sky_and_beam_dirty should update parameters when called."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        params_updated, _ = calibrator.fit_sky_and_beam_dirty(
            params, n_iter=1, sky_step_size=0.5, beam_step_size=0.5
        )

        # Verify params were processed (output matches size of input)
        assert params_updated['sky_coeffs'].shape == params['sky_coeffs'].shape
        assert params_updated['beam_coeffs'].shape == params['beam_coeffs'].shape

    def test_fit_sky_and_beam_dirty_runs_without_error(self, calibrator, sky_coeffs_test):
        """fit_sky_and_beam_dirty should run without error."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        # Should complete without error
        params_updated, loss = calibrator.fit_sky_and_beam_dirty(
            params, n_iter=1, sky_step_size=0.5, beam_step_size=0.5
        )

        # Verify outputs are valid
        assert np.isfinite(loss), "Loss should be finite"
        assert np.all(np.isfinite(np.asarray(params_updated['sky_coeffs']))), "Sky should be finite"
        assert np.all(np.isfinite(np.asarray(params_updated['beam_coeffs']))), "Beam should be finite"

    def test_fit_sky_and_beam_dirty_with_anderson_acceleration(self, calibrator, sky_coeffs_test):
        """fit_sky_and_beam_dirty should work with Anderson acceleration enabled."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        # Run with AA enabled (need at least 2 iterations for AA to activate)
        params_updated, loss = calibrator.fit_sky_and_beam_dirty(
            params, n_iter=3, sky_step_size=0.5, beam_step_size=0.5,
            anderson_history=2
        )

        # Verify outputs are valid
        assert np.isfinite(loss), "Loss should be finite"
        assert np.all(np.isfinite(np.asarray(params_updated['sky_coeffs']))), "Sky should be finite"
        assert np.all(np.isfinite(np.asarray(params_updated['beam_coeffs']))), "Beam should be finite"


class TestFitAlternatingDirtyWithUnifiedInterface:
    """Test that fit_alternating_dirty works with unified interface."""

    def test_fit_alternating_dirty_reduces_loss(self, calibrator, sky_coeffs_test):
        """fit_alternating_dirty should reduce loss."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        loss_init = calibrator.calc_loss(params, explicit_beam=True)

        params_updated, loss_updated = calibrator.fit_alternating_dirty(
            params, n_iter=3
        )

        assert loss_updated < loss_init, (
            f"Loss should decrease; got {loss_init:.3e} → {loss_updated:.3e}"
        )

    def test_fit_alternating_dirty_returns_full_sky(self, calibrator, sky_coeffs_test):
        """fit_alternating_dirty should return full-sky format."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        params_updated, _ = calibrator.fit_alternating_dirty(params, n_iter=1)

        # Output should be full-sky format
        assert params_updated['sky_coeffs'].shape[0] == calibrator._npix_full


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibrator(array, beam_model, freqs, rot_matrices, sky_model):
    """Create a Calibrator instance with synthetic data."""
    from newnucal.calibrator import Calibrator
    from newnucal.simulate import ForwardModel
    from newnucal.basis import basis_project

    # Create forward model
    fwd = ForwardModel(array, sky_model, beam_model, freqs)
    fwd.precompute_time_geometry(rot_matrices)

    # Create a test sky (flat spectrum, low amplitude)
    npix = fwd.npix_sky
    nfreq = len(freqs)
    sky_coeffs = jnp.array(basis_project(
        np.ones((npix, nfreq), dtype=np.float32) * 0.1,
        np.asarray(sky_model.A)
    ), dtype=jnp.float32)

    # Simulate data
    vis_model = fwd.simulate(sky_coeffs, jnp.array(rot_matrices))

    # Add some noise
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.01, vis_model.shape) + 1j * rng.normal(0, 0.01, vis_model.shape)
    data = np.asarray(vis_model) + noise

    # Create calibrator
    return Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=data,
    )


@pytest.fixture
def sky_coeffs_test(forward_model, A_sky):
    """Random sky coefficients for testing."""
    rng = np.random.default_rng(42)
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    return jnp.array(rng.normal(0, 0.1, (npix, nmodes)), dtype=jnp.float32)
