"""Closed-loop convergence baselines for joint sky/beam fitting.

These tests are intentionally numerical regression tests rather than proof-style
unit tests.  They keep a lightweight, deterministic version of the notebook
workflow in CI so convergence changes have a baseline to beat.  For proposals
that add more per-step compute, such as BB or multi-candidate searches, compare
loss reduction per wall time rather than loss at a fixed iteration count alone.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from newnucal.basis import basis_project
from newnucal.calibrator import Calibrator
from newnucal.gains import apply_gains, init_gain_params
from newnucal.simulate import ForwardModel


OLD_BASELINE_JOINT_SETTINGS = dict(
    sky_beam_reg=1e-3,
    joint_initial_step=1.0,
    solve_every={'gains': 8, 'rfi': 3},
    rfi_smooth_width_chans=5,
    rfi_log_threshold=np.log(3.0),
    rfi_gamma=0.75,
    rfi_log_min_weight=np.log(1e-5),
    rfi_alpha_down=0.5,
    rfi_alpha_up=0.9,
    rfi_min_retention_per_update=0.5,
    max_rfi_updates=None,
)

DEFAULT_BENCHMARK_JOINT_SETTINGS = dict(
    OLD_BASELINE_JOINT_SETTINGS,
    sky_beam_reg=3e-3,
    joint_initial_step=1.2,
    solve_every=None,
)


def _smooth_gain_params(ntime, nfreq):
    f_norm = np.linspace(0, 1, nfreq, endpoint=False)[None, :]
    t_norm = np.linspace(0, 1, ntime)[:, None]
    log_amp = (
        0.05 * np.cos(2 * np.pi * f_norm)
        + 0.02 * np.sin(2 * np.pi * t_norm)
    ).astype(np.float32)
    phase = (
        0.15 * np.sin(2 * np.pi * f_norm)
        + 0.05 * np.cos(2 * np.pi * t_norm)
    ).astype(np.float32)
    phi = np.zeros((ntime, 2, nfreq), dtype=np.float32)
    phi[:, 0, :] = (
        1e-4 * np.cos(2 * np.pi * f_norm)
        + 3e-5 * np.sin(2 * np.pi * t_norm)
    )
    phi[:, 1, :] = (
        5e-5 * np.sin(2 * np.pi * f_norm)
        + 2e-5 * np.cos(2 * np.pi * t_norm)
    )
    return {
        'log_amp': jnp.array(log_amp),
        'phase': jnp.array(phase),
        'phi': jnp.array(phi),
    }


def _make_closed_loop_case(
    array, beam_model, freqs, rot_matrices, sky_model, *, apply_true_gains=False
):
    fwd = ForwardModel(array, sky_model, beam_model, freqs)
    fwd.precompute_time_geometry(rot_matrices)

    true_sky = jnp.array(
        basis_project(
            np.ones((fwd.npix_sky, len(freqs)), dtype=np.float32) * 0.1,
            np.asarray(sky_model.A),
        ),
        dtype=jnp.float32,
    )
    vis_model = fwd.simulate_3d(true_sky, jnp.array(rot_matrices))
    if apply_true_gains:
        vis_model = apply_gains(
            vis_model,
            **_smooth_gain_params(rot_matrices.shape[0], len(freqs)),
            bls=jnp.array(array.bls, dtype=jnp.float32),
        )

    rng = np.random.default_rng(123)
    noise = (
        rng.normal(0, 0.01, vis_model.shape)
        + 1j * rng.normal(0, 0.01, vis_model.shape)
    )
    cal = Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=np.asarray(vis_model) + noise,
    )

    rng = np.random.default_rng(42)
    sky_start = jnp.array(
        rng.normal(0, 0.1, (fwd.npix_sky, sky_model.A.shape[1])),
        dtype=jnp.float32,
    )
    params = {
        'sky_coeffs': sky_start,
        'beam_coeffs': cal.fwd.beam_coeffs,
        **init_gain_params(cal.ntime, cal.nfreq),
    }
    return cal, params


def _joint_loss_trace(
    array,
    beam_model,
    freqs,
    rot_matrices,
    sky_model,
    *,
    joint_anderson_history,
    n_steps=12,
    settings_overrides=None,
    apply_true_gains=False,
):
    cal, params = _make_closed_loop_case(
        array,
        beam_model,
        freqs,
        rot_matrices,
        sky_model,
        apply_true_gains=apply_true_gains,
    )
    settings = dict(DEFAULT_BENCHMARK_JOINT_SETTINGS)
    if settings_overrides is not None:
        settings.update(settings_overrides)
    state = cal.init_joint_sky_beam_dirty_state(
        params,
        joint_anderson_history=joint_anderson_history,
        **settings,
    )
    losses = [float(state.loss)]
    for _ in range(n_steps):
        state = cal.run_joint_sky_beam_dirty_state(state, n_iter=1)
        losses.append(float(state.loss))
    return np.asarray(losses), state


@pytest.mark.convergence
def test_closed_loop_default_joint_convergence_baseline(
    array, beam_model, freqs, rot_matrices, sky_model
):
    aa_losses, aa_state = _joint_loss_trace(
        array,
        beam_model,
        freqs,
        rot_matrices,
        sky_model,
        joint_anderson_history=2,
    )
    plain_losses, plain_state = _joint_loss_trace(
        array,
        beam_model,
        freqs,
        rot_matrices,
        sky_model,
        joint_anderson_history=0,
    )

    assert aa_state.step == 12
    assert aa_state.n_joint == 9
    assert aa_state.n_gains == 3
    assert aa_state.n_rfi == 4
    assert plain_state.step == aa_state.step

    assert np.all(np.diff(aa_losses) <= 1e-6)
    assert aa_losses[4] < 0.72
    assert aa_losses[8] < 0.46
    assert aa_losses[12] < 0.03

    assert aa_losses[12] < 0.96 * plain_losses[12]


@pytest.mark.convergence
def test_closed_loop_default_parameters_beat_old_baseline(
    array, beam_model, freqs, rot_matrices, sky_model
):
    old_losses, old_state = _joint_loss_trace(
        array,
        beam_model,
        freqs,
        rot_matrices,
        sky_model,
        joint_anderson_history=2,
        settings_overrides=OLD_BASELINE_JOINT_SETTINGS,
    )
    default_losses, default_state = _joint_loss_trace(
        array,
        beam_model,
        freqs,
        rot_matrices,
        sky_model,
        joint_anderson_history=2,
    )

    assert default_state.step == old_state.step == 12
    assert default_state.n_joint == 9
    assert old_state.n_joint == 10
    assert np.all(np.diff(default_losses) <= 1e-6)
    assert default_losses[12] < 0.60 * old_losses[12]


@pytest.mark.convergence
def test_closed_loop_default_joint_convergence_with_nonunity_gains(
    array, beam_model, freqs, rot_matrices, sky_model
):
    losses, state = _joint_loss_trace(
        array,
        beam_model,
        freqs,
        rot_matrices,
        sky_model,
        joint_anderson_history=2,
        apply_true_gains=True,
    )

    assert state.step == 12
    assert np.all(np.isfinite(losses))
    assert state.n_joint == 9
    assert state.n_gains == 3
    assert state.n_rfi == 4
    assert losses[12] < 0.07
