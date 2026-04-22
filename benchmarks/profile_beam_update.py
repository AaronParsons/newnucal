"""
Benchmark and profile the beam update step to identify bottlenecks.

Reproduces the closed_loop_test.ipynb scenario with sky masking active,
then profiles a single beam dirty update iteration.
"""

import cProfile
import pstats
import numpy as np
import jax.numpy as jnp
from io import StringIO

from newnucal import HERAArray, BeamModel, BeamBasis, SkyBasis, SkyModel, Calibrator, init_gain_params
from newnucal.simulate import compute_rotation_matrices
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u


def setup_calibrator():
    """Set up the same scenario as closed_loop_test.ipynb with sky masking."""
    # --- Array ---
    array = HERAArray.from_hex(hexnum=5, sep=14.6)
    print(f"Array: {array.nants} antennas, {array.nbls} baselines")

    # --- Frequencies ---
    nfreq = 32
    freqs = np.linspace(50e6, 225e6, nfreq, dtype=np.float32)

    # --- Times ---
    hera_loc = EarthLocation(lat=-30.7215 * u.deg, lon=21.4283 * u.deg, height=1073.0 * u.m)
    t0 = Time("2023-03-21T04:00:00", scale="utc")
    ntime = 24
    dt = 240 * u.min / (ntime + 1)
    times = t0 + np.arange(ntime) * dt
    rot_matrices = compute_rotation_matrices(times, hera_loc)

    # --- Beam and sky models ---
    sky_nside = 32
    beam_basis = BeamBasis.from_dpss(freqs, eta_max=20e-9)
    beam_model = BeamModel(nside=16, freqs=freqs, basis=beam_basis)

    sky_basis = SkyBasis.from_dpss(freqs, eta_max=40e-9)
    sky_model = SkyModel(nside=sky_nside, freqs=freqs, basis=sky_basis)

    # --- Synthetic data ---
    fwd_temp = __import__("newnucal").ForwardModel(array, sky_model, beam_model, freqs, eps=1e-5)

    npix_sky = fwd_temp.npix_sky
    ref_freq = 150e6
    rng = np.random.default_rng(42)
    spectral_indices = rng.normal(-0.7, 0.1, npix_sky).astype(np.float32)
    ref_flux = rng.exponential(scale=1.0, size=npix_sky).astype(np.float32)
    flux_true = ref_flux[:, None] * (freqs[None, :] / ref_freq) ** spectral_indices[:, None]
    sky_coeffs_true = jnp.array(sky_basis.project(flux_true), dtype=jnp.float32)

    vis_true = fwd_temp.simulate(sky_coeffs_true, jnp.array(rot_matrices))

    # Smooth gains
    t_norm = np.linspace(0, 1, ntime)[:, None]
    f_norm = np.linspace(0, 1, nfreq)[None, :]
    true_log_amp = (0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)).astype(np.float32)
    true_phase = (0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)).astype(np.float32)
    true_phi = np.zeros((ntime, 2, nfreq), dtype=np.float32)
    true_phi[:, 0, :] = (1e-4 * np.cos(2 * np.pi * f_norm) + 3e-5 * np.sin(2 * np.pi * t_norm))
    true_phi[:, 1, :] = (5e-5 * np.sin(2 * np.pi * f_norm) + 2e-5 * np.cos(2 * np.pi * t_norm))

    from newnucal.gains import apply_gains
    vis_data = apply_gains(vis_true, jnp.array(true_log_amp), jnp.array(true_phase),
                           jnp.array(true_phi), jnp.array(array.bls, dtype=jnp.float32))

    # --- Create calibrator ---
    cal = Calibrator(array=array, sky_model=sky_model, beam_model=beam_model, freqs=freqs,
                     rot_matrices=rot_matrices, data=vis_data, eps=1e-5)

    # --- Apply sky masking (like in the notebook) ---
    bm_wgts = cal.get_sky_beam_weighting()
    threshold = bm_wgts.max() / 1000
    mask = bm_wgts > threshold
    cal.apply_pixel_mask(mask)

    npix_active = int(mask.sum())
    print(f"Sky masking: {npix_active}/{npix_sky} pixels active")

    # --- Perturbed parameters ---
    rng2 = np.random.default_rng(99)
    sky_coeffs_perturbed = sky_coeffs_true + 0.30 * jnp.array(
        rng2.standard_normal(sky_coeffs_true.shape).astype(np.float32)
    ) * float(jnp.abs(sky_coeffs_true).mean())

    beam_coeffs_true = jnp.array(beam_model.coeffs, dtype=jnp.float32)
    rng3 = np.random.default_rng(77)
    beam_coeffs_perturbed = beam_coeffs_true + 0.30 * jnp.array(
        rng3.standard_normal(beam_coeffs_true.shape).astype(np.float32)
    ) * float(jnp.abs(beam_coeffs_true).mean())

    params = {
        "sky_coeffs": sky_coeffs_perturbed,
        "beam_coeffs": beam_coeffs_perturbed,
        **init_gain_params(ntime, nfreq),
    }

    return cal, params


def profile_sky_update(cal, params):
    """Profile a single sky dirty update."""
    print("\n" + "="*60)
    print("PROFILING SKY UPDATE (1 iteration)")
    print("="*60)

    sky_coeffs = params['sky_coeffs']
    gain_params = {k: params[k] for k in ['log_amp', 'phase', 'phi']}

    pr = cProfile.Profile()
    pr.enable()
    sky_out, loss = cal.fit_sky_dirty(sky_coeffs, gain_params, n_iter=1, verbose=True)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_beam_update(cal, params):
    """Profile a single beam dirty update."""
    print("\n" + "="*60)
    print("PROFILING BEAM UPDATE (1 iteration)")
    print("="*60)

    pr = cProfile.Profile()
    pr.enable()
    params_out, loss = cal.fit_beam_dirty(params, n_iter=1, verbose=True)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    cal, params = setup_calibrator()

    # Warm up JIT
    print("\nWarming up JIT...")
    _ = cal.calc_loss(params)

    # Profile sky and beam updates WITHOUT beam masking
    print("\n" + "="*60)
    print("WITHOUT BEAM MASKING")
    print("="*60)
    profile_sky_update(cal, params)
    profile_beam_update(cal, params)

    # Now apply beam masking and profile again
    print("\n" + "="*60)
    print("WITH BEAM MASKING (ever-illuminated)")
    print("="*60)
    cal2, params2 = setup_calibrator()
    cal2.apply_ever_illuminated_beam_mask()
    print(f"\nBeam reduced from {cal.fwd.npix_beam} to {cal2.fwd.npix_beam} pixels")

    # Warm up JIT for masked version
    _ = cal2.calc_loss(params2)

    profile_sky_update(cal2, params2)
    profile_beam_update(cal2, params2)
