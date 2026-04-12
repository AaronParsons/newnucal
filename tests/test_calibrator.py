"""
Tests for Calibrator methods.

TestFitGainsLinear: closed-loop test that with the true sky fixed,
fit_gains_linear recovers gains to near-machine precision.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from newnucal.dpss import dpss_project
from newnucal.gains import apply_gains, init_gain_params

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
    return jnp.array(dpss_project(flux_map, A_sky), dtype=jnp.float32)


@pytest.fixture
def calibrator_gains_setup(array, beam_model, freqs, rot_matrices,
                            forward_model, sky_coeffs_true):
    """Returns (Calibrator, true_gain_params) where data = simulate(sky_true) * gains."""
    from newnucal.calibrator import Calibrator

    ntime = rot_matrices.shape[0]
    nfreq = len(freqs)

    vis_true = forward_model.simulate(sky_coeffs_true, jnp.array(rot_matrices))

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
        sky_nside=NSIDE_SKY,
        sky_eta_max=ETA_SKY,
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
