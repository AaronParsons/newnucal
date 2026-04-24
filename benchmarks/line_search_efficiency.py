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
import argparse
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'

import jax

jax.config.update("jax_enable_x64", False)

from bench_utils import setup_benchmark_calibrator


def run_config(cal, prms0, config_name, sky_line_search_steps, n_iter):
    """Run one configuration and measure efficiency."""
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"  sky_line_search_steps: {sky_line_search_steps}")
    print(f"  Running {n_iter} iterations...")
    print(f"{'='*70}")

    chi2_before = cal.calc_loss(prms0)
    t_start = time.perf_counter()

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
    cal.run_alternating_dirty_state(state, n_iter=n_iter, verbose=False)

    t_elapsed = time.perf_counter() - t_start
    chi2_after = cal.calc_loss(state.params)
    chi2_reduction = chi2_before - chi2_after
    chi2_reduction_rate = chi2_reduction / t_elapsed
    chi2_reduction_pct = 100 * chi2_reduction / chi2_before

    print(f"\nResults:")
    print(f"  χ² before: {chi2_before:.4e}")
    print(f"  χ² after:  {chi2_after:.4e}")
    print(f"  χ² reduction: {chi2_reduction:.4e} ({chi2_reduction_pct:.1f}%)")
    print(f"  Wall-clock time: {t_elapsed:.1f}s")
    print(f"  χ² reduction rate: {chi2_reduction_rate:.2e} χ²/s")

    return dict(
        config=config_name, n_steps=n_iter,
        chi2_before=chi2_before, chi2_after=chi2_after,
        chi2_reduction=chi2_reduction, chi2_reduction_pct=chi2_reduction_pct,
        wall_time=t_elapsed, chi2_rate=chi2_reduction_rate,
        line_search_steps=sky_line_search_steps,
    )


def main(args):
    print("=" * 70)
    print("Line Search Efficiency Benchmark")
    print("=" * 70)

    print("\n[Setup] Preparing calibration problem...")
    cal, prms0, info = setup_benchmark_calibrator()
    print(f"[Setup] {info['nants']} antennas, {info['nbls']} baselines, "
          f"{info['nfreq']} freqs.  Problem ready.")

    configs = [
        ("No line search (fixed step=0.9)", [0.9]),
        ("Narrow line search", [0.8, 1.0, 1.2, 1.4]),
        ("Default line search", [0.5, 1.0, 1.5, 2.0]),
    ]

    results = []
    for config_name, step_sizes in configs:
        results.append(run_config(cal, prms0, config_name, step_sizes, args.n_iter))

    print(f"\n{'='*70}")
    print("SUMMARY: Efficiency Comparison")
    print(f"{'='*70}\n")
    baseline_rate = results[0]['chi2_rate']
    print(f"{'Config':<30} {'χ²/s':>12} {'Speedup':>10} {'Time (s)':>10}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*10}")
    for r in results:
        speedup = r['chi2_rate'] / baseline_rate
        print(f"{r['config']:<30} {r['chi2_rate']:>12.2e} {speedup:>10.2f}x {r['wall_time']:>10.1f}s")

    print(f"\n{'='*70}")
    if results[2]['chi2_rate'] > results[0]['chi2_rate']:
        print("✓ Default line search is MORE efficient (χ²/s speedup > 1)")
    else:
        print("✗ Default line search is LESS efficient (χ²/s speedup < 1)")
        print("  Suggests 4× loss evaluations are not offset by better step selection.")
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
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Number of fit iterations per configuration (default: 20).")
    args = parser.parse_args()
    main(args)
