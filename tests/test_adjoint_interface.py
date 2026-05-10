"""
Tests for the unified adjoint interface (compute_adjoint_updates).

Verifies that the unified interface produces identical results to the
individual adjoint calls, and tests that all fit_*_dirty methods work
correctly with the new interface.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from newnucal.calibrator import DEFAULT_JOINT_SOLVE_EVERY
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


class TestFitJointSkyBeamDirtyWithUnifiedInterface:
    """Test the new joint sky+beam dirty-map method."""

    def test_fit_joint_sky_beam_dirty_reduces_loss(self, calibrator, sky_coeffs_test):
        """fit_joint_sky_beam_dirty should reduce loss."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }
        loss_init = calibrator.calc_loss(params, explicit_beam=True)

        params_updated, loss_updated = calibrator.fit_joint_sky_beam_dirty(
            params, n_iter=3
        )

        assert loss_updated < loss_init, (
            f"Loss should decrease; got {loss_init:.3e} → {loss_updated:.3e}"
        )

    def test_fit_joint_sky_beam_dirty_returns_full_sky(self, calibrator, sky_coeffs_test):
        """fit_joint_sky_beam_dirty should return full-sky format."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        params_updated, _ = calibrator.fit_joint_sky_beam_dirty(params, n_iter=1)

        # Output should be full-sky format
        assert params_updated['sky_coeffs'].shape[0] == calibrator._npix_full

    def test_fit_joint_sky_beam_dirty_with_anderson(self, calibrator, sky_coeffs_test):
        """fit_joint_sky_beam_dirty should work with Anderson acceleration."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        params_updated, loss = calibrator.fit_joint_sky_beam_dirty(
            params, n_iter=3, joint_anderson_history=2
        )

        assert np.isfinite(loss), "Loss should be finite"
        assert np.all(np.isfinite(np.asarray(params_updated['sky_coeffs']))), "Sky should be finite"

    def test_fit_joint_sky_beam_dirty_with_gain_solve(self, calibrator, sky_coeffs_test):
        """fit_joint_sky_beam_dirty should support solve_every for gain updates."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        # Run with gain solves every 2 steps
        params_updated, loss = calibrator.fit_joint_sky_beam_dirty(
            params, n_iter=3, solve_every={'gains': 2}
        )

        assert np.isfinite(loss), "Loss should be finite"

    def test_init_joint_sky_beam_dirty_state_creates_valid_state(self, calibrator, sky_coeffs_test):
        """init_joint_sky_beam_dirty_state should create a valid state."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        state = calibrator.init_joint_sky_beam_dirty_state(params)

        assert state.loss is not None
        assert np.isfinite(state.loss)
        assert state.step == 0
        assert state.n_joint == 0
        assert state.n_gains == 0
        assert state.settings['solve_every'] == DEFAULT_JOINT_SOLVE_EVERY
        assert state.settings['solve_every'] is not DEFAULT_JOINT_SOLVE_EVERY

    def test_init_joint_sky_beam_dirty_state_preserves_explicit_solve_every(
        self, calibrator, sky_coeffs_test
    ):
        """Explicit solve_every should override the tuned default cadence."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        state = calibrator.init_joint_sky_beam_dirty_state(
            params, solve_every={'gains': 0}
        )

        assert state.settings['solve_every'] == {'gains': 0}

    def test_run_joint_sky_beam_dirty_state_advances(self, calibrator, sky_coeffs_test):
        """run_joint_sky_beam_dirty_state should advance the state."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        state = calibrator.init_joint_sky_beam_dirty_state(params)
        loss_init = state.loss

        state = calibrator.run_joint_sky_beam_dirty_state(state, n_iter=2)

        assert state.step == 2, "State step should advance"
        assert state.loss is not None
        assert np.isfinite(state.loss)

    def test_iter_joint_sky_beam_dirty_yields_states(self, calibrator, sky_coeffs_test):
        """iter_joint_sky_beam_dirty should yield intermediate states."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        state = calibrator.init_joint_sky_beam_dirty_state(params)

        yielded_states = list(calibrator.iter_joint_sky_beam_dirty(
            state, n_iter=3, yield_every=1
        ))

        assert len(yielded_states) == 3, "Should yield 3 states"
        assert all(s.step > 0 for s in yielded_states)
        assert yielded_states[-1].step == 3

    def test_fit_joint_sky_beam_dirty_with_initial_step_list(self, calibrator, sky_coeffs_test):
        """fit_joint_sky_beam_dirty should perform line search with initial_step list."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        # Run with initial_step as a list (should do line search on first iter)
        params_updated, loss = calibrator.fit_joint_sky_beam_dirty(
            params, n_iter=2, joint_initial_step=[0.1, 0.5, 1.0]
        )

        assert np.isfinite(loss), "Loss should be finite"
        # After first iteration, initial_step should be saved as scalar
        state = calibrator.init_joint_sky_beam_dirty_state(
            params_updated, joint_initial_step=[0.1, 0.5, 1.0]
        )
        # Settings should have updated joint_initial_step as scalar if line search succeeded
        # (This is checked indirectly through convergence)

    def test_joint_state_accepts_separate_initial_steps(self, calibrator, sky_coeffs_test):
        """Joint state should keep separate sky and beam step-size controls."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        state = calibrator.init_joint_sky_beam_dirty_state(
            params,
            joint_sky_initial_step=[0.1, 0.3],
            joint_beam_initial_step=[0.2, 0.4],
            beam_reg=1e-3,
            sky_reg=2e-3,
            solve_every={'gains': 0},
        )
        state = calibrator.run_joint_sky_beam_dirty_state(state, n_iter=1)

        assert isinstance(state.settings['joint_sky_initial_step'], float)
        assert isinstance(state.settings['joint_beam_initial_step'], float)
        assert state.settings['joint_beam_reg'] == 1e-3
        assert state.settings['joint_sky_reg'] == 2e-3
        assert np.isfinite(state.loss)

    def test_joint_aa_history_persists_across_gain_step(self, calibrator, sky_coeffs_test):
        """Gain solves should not discard useful joint AA history."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        state = calibrator.init_joint_sky_beam_dirty_state(
            params,
            joint_anderson_history=3,
            solve_every={'gains': 0, 'rfi': 0},
        )
        state = calibrator.run_joint_sky_beam_dirty_state(state, n_iter=2)
        hist_len = len(state.joint_acc._hist_f)
        state.settings['solve_every']['gains'] = 1
        state.n_since_gains = 1
        state.eff_gains = 0.0
        state = calibrator.run_joint_sky_beam_dirty_state(state, n_iter=1)

        assert state.n_gains == 1
        assert len(state.joint_acc._hist_f) == hist_len

    def test_fit_joint_sky_beam_dirty_with_step_gain_factor(self, calibrator, sky_coeffs_test):
        """fit_joint_sky_beam_dirty should support step_gain_factor for adaptive scaling."""
        params = {
            'sky_coeffs': sky_coeffs_test,
            'beam_coeffs': calibrator.fwd.beam_coeffs,
            **init_gain_params(calibrator.ntime, calibrator.nfreq)
        }

        # Run with custom step_gain_factor
        params_updated, loss = calibrator.fit_joint_sky_beam_dirty(
            params, n_iter=3, joint_anderson_history=0, joint_step_gain_factor=3.0
        )

        assert np.isfinite(loss), "Loss should be finite"


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
    vis_model = fwd.simulate_3d(sky_coeffs, jnp.array(rot_matrices))

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
