#!/usr/bin/env python
"""
Line Search Efficiency Benchmark

Measures χ² reduction rate (χ²/second) for different line search configurations
to determine if the extra loss evaluations are worth the cost.

Compares:
  1. No line search (fixed step size only)
  2. Narrow line search (cheaper, [0.8, 1.0, 1.2, 1.4])
  3. Default line search (standard, [0.5, 1.0, 1.5, 2.0])

Usage:
    python benchmarks/line_search_efficiency.py --n-iter 20

This will run 20 iterations of each configuration and report:
  - Initial χ² and final χ²
  - Total wall-clock time
  - χ² reduction rate (χ² change per second)
  - Effective speedup relative to no line search
"""
import os
import sys
import argparse
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

jax.config.update("jax_enable_x64", False)

from newnucal import (
    HERAArray, BeamModel, BeamBasis, SkyBasis, SkyModel,
    ForwardModel, Calibrator, apply_gains, init_gain_params
)
from newnucal.simulate import compute_rotation_matrices


def setup_problem():
    """Setup the calibration problem (shared for all configurations)."""
    # --- Array ---
    array = HERAArray.from_hex(hexnum=5, sep=14.6)

    # --- Frequencies ---
    nfreq = 64
    freqs = np.linspace(50e6, 225e6, nfreq, dtype=np.float32)
    ref_freq = 150e6

    # --- Times ---
    hera_loc = EarthLocation(lat=-30.7215 * u.deg, lon=21.4283 * u.deg, height=1073.0 * u.m)
    t0 = Time("2023-03-21T04:00:00", scale="utc")
    ntime = 24
    dt = 240 * u.min / (ntime + 1)
    times = t0 + np.arange(ntime) * dt
    rot_matrices = compute_rotation_matrices(times, hera_loc)

    # --- Bases ---
    try:
        beam_basis = BeamBasis.from_file('benchmarks/beam_basis_airy.npz', new_freqs=freqs)
        sky_basis = SkyBasis.from_file('benchmarks/sky_basis_gsm_gleam.npz', new_freqs=freqs)
    except FileNotFoundError as e:
        print(f"Error: Could not find basis files. {e}")
        sys.exit(1)

    sky_nside = 32
    beam_model = BeamModel(nside=16, freqs=freqs, basis=beam_basis)
    sky_model = SkyModel(nside=sky_nside, freqs=freqs, basis=sky_basis)

    # --- Simulate sky and gains ---
    import healpy
    npix_sky = healpy.nside2npix(sky_nside)
    rng = np.random.default_rng(42)
    spectral_indices = rng.normal(-0.7, 0.1, npix_sky).astype(np.float32)
    ref_flux = rng.exponential(scale=1.0, size=npix_sky).astype(np.float32)
    flux_true = ref_flux[:, None] * (freqs[None, :] / ref_freq) ** spectral_indices[:, None]
    sky_coeffs_true = jnp.array(sky_basis.project(flux_true), dtype=jnp.float32)
    beam_coeffs_true = jnp.array(beam_model.coeffs, dtype=jnp.float32)

    # --- Simulate visibilities with gains ---
    fwd = ForwardModel(array, sky_model, beam_model, freqs, eps=1e-5)
    vis_true = fwd.simulate_2d(sky_coeffs_true, jnp.array(rot_matrices))

    t_norm = np.linspace(0, 1, ntime)[:, None]
    f_norm = np.linspace(0, 1, nfreq)[None, :]
    true_log_amp = (0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)).astype(np.float32)
    true_phase = (0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)).astype(np.float32)
    true_phi = np.zeros((ntime, 2, nfreq), dtype=np.float32)
    vis_data = apply_gains(vis_true, jnp.array(true_log_amp), jnp.array(true_phase),
                          jnp.array(true_phi), jnp.array(array.bls, dtype=jnp.float32))

    # --- Build calibrator ---
    cal = Calibrator(
        array=array, sky_model=sky_model, beam_model=beam_model,
        freqs=freqs, rot_matrices=rot_matrices, data=vis_data, eps=1e-5, method='2d'
    )

    # --- Apply pixel masking ---
    bm_wgts = cal.get_sky_beam_weighting()
    threshold = bm_wgts.max() / 100
    mask = bm_wgts > threshold
    cal.apply_pixel_mask(mask)
    cal.apply_ever_illuminated_beam_mask()

    # --- Perturbed start ---
    rng2 = np.random.default_rng(99)
    sky_coeffs_perturbed = sky_coeffs_true + 0.30 * jnp.array(
        rng2.standard_normal(sky_coeffs_true.shape).astype(np.float32)
    ) * float(jnp.abs(sky_coeffs_true).mean())

    rng3 = np.random.default_rng(77)
    beam_coeffs_perturbed = beam_coeffs_true + 0.30 * jnp.array(
        rng3.standard_normal(beam_coeffs_true.shape).astype(np.float32)
    ) * float(jnp.abs(beam_coeffs_true).mean())

    prms0 = {
        "sky_coeffs": sky_coeffs_perturbed,
        "beam_coeffs": beam_coeffs_perturbed,
        **init_gain_params(ntime, nfreq),
    }

    return cal, prms0


def run_config(cal, prms0, config_name, sky_line_search_steps, n_iter):
    """Run one configuration and measure efficiency."""
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"  sky_line_search_steps: {sky_line_search_steps}")
    print(f"  Running {n_iter} iterations...")
    print(f"{'='*70}")

    loss_before = cal.calc_loss(prms0)
    chi2_before = loss_before  # Loss IS χ²

    t_start = time.perf_counter()

    # Initialize state with line search configuration
    state = cal.init_alternating_dirty_state(
        prms0,
        sky_line_search_steps=sky_line_search_steps,
        sky_anderson_history=0,
        sky_aa_start=2,
        sky_aa_damping=0.45,
        sky_aa_ridge=1e-4,
        beam_line_search_steps=[0.7],
        beam_anderson_history=0,
        beam_aa_start=2,
        beam_aa_damping=0.65,
        beam_aa_ridge=1e-4,
        solve_every={'gains': 20, 'gains_max': 20, 'sky_max': 5, 'beam_max': 4},
    )

    # Run fit through state
    cal.run_alternating_dirty_state(state, n_iter=n_iter, verbose=False)
    prms = state.params
    t_elapsed = time.perf_counter() - t_start
    loss_after = cal.calc_loss(prms)
    chi2_after = loss_after

    # --- Compute metrics ---
    chi2_reduction = chi2_before - chi2_after
    chi2_reduction_rate = chi2_reduction / t_elapsed
    chi2_reduction_pct = 100 * chi2_reduction / chi2_before

    print(f"\nResults:")
    print(f"  χ² before: {chi2_before:.4e}")
    print(f"  χ² after:  {chi2_after:.4e}")
    print(f"  χ² reduction: {chi2_reduction:.4e} ({chi2_reduction_pct:.1f}%)")
    print(f"  Wall-clock time: {t_elapsed:.1f}s")
    print(f"  χ² reduction rate: {chi2_reduction_rate:.2e} χ²/s")

    return {
        'config': config_name,
        'n_steps': n_iter,
        'chi2_before': chi2_before,
        'chi2_after': chi2_after,
        'chi2_reduction': chi2_reduction,
        'chi2_reduction_pct': chi2_reduction_pct,
        'wall_time': t_elapsed,
        'chi2_rate': chi2_reduction_rate,
        'line_search_steps': sky_line_search_steps,
    }


def main(args):
    """Run efficiency benchmark comparing line search configurations."""

    print("=" * 70)
    print("Line Search Efficiency Benchmark")
    print("=" * 70)

    # Setup shared problem
    print("\n[Setup] Preparing calibration problem...")
    cal, prms0 = setup_problem()
    print("[Setup] Problem ready. Running configurations...")

    # Configurations to test
    configs = [
        ("No line search (fixed step=0.9)", [0.9]),
        ("Narrow line search", [0.8, 1.0, 1.2, 1.4]),
        ("Default line search", [0.5, 1.0, 1.5, 2.0]),
    ]

    results = []
    for config_name, step_sizes in configs:
        result = run_config(cal, prms0, config_name, step_sizes, args.n_iter)
        results.append(result)

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print("SUMMARY: Efficiency Comparison")
    print(f"{'='*70}\n")

    baseline = results[0]  # No line search as baseline
    baseline_rate = baseline['chi2_rate']

    print(f"{'Config':<30} {'χ²/s':>12} {'Speedup':>10} {'Time (s)':>10}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*10}")

    for result in results:
        speedup = result['chi2_rate'] / baseline_rate
        print(f"{result['config']:<30} {result['chi2_rate']:>12.2e} {speedup:>10.2f}x {result['wall_time']:>10.1f}s")

    print(f"\n{'='*70}")
    if results[2]['chi2_rate'] > results[0]['chi2_rate']:
        print("✓ Default line search is MORE efficient (χ²/s speedup > 1)")
    else:
        print("✗ Default line search is LESS efficient (χ²/s speedup < 1)")
        print("  Suggests that the 4× loss evaluations are not offset by better step selection.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure line search efficiency by χ² reduction per second",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/line_search_efficiency.py --n-iter 20
  python benchmarks/line_search_efficiency.py --n-iter 30
        """,
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of fit iterations per configuration (default: 20)",
    )

    args = parser.parse_args()
    main(args)
