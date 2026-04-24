#!/usr/bin/env python
"""
Profiling benchmark for fit_joint_sky_beam_dirty.

Uses cProfile to measure per-subroutine time and call counts during
joint sky+beam dirty-map fitting.
"""
import argparse
import cProfile
import os
import pstats
from io import StringIO

os.environ['OMP_NUM_THREADS'] = '1'

import jax
jax.config.update("jax_enable_x64", False)

from bench_utils import setup_benchmark_calibrator


def main(args):
    print("=" * 80)
    print("Joint Sky+Beam Dirty-Map Profiling Benchmark")
    print("=" * 80)

    print("\n[Setup] Building calibration problem...")
    cal, prms0, info = setup_benchmark_calibrator()
    loss_before = cal.calc_loss(prms0)
    print(f"[Setup] {info['nants']} antennas, {info['nbls']} baselines, "
          f"{info['nfreq']} freqs")
    print(f"[Setup] Initial χ²: {loss_before:.4e}")
    print(f"[Setup] Running {args.n_iter} iterations with aa_history={args.aa_history}\n")

    # JIT warm-up
    print("[Warmup] Compiling JAX functions...")
    _ = cal.calc_loss(prms0)
    print("[Warmup] Done.\n")

    # Profile the fit
    print("=" * 80)
    print(f"Profiling fit_joint_sky_beam_dirty ({args.n_iter} iterations)")
    print("=" * 80 + "\n")

    pr = cProfile.Profile()
    pr.enable()

    params, loss_after = cal.fit_joint_sky_beam_dirty(
        prms0,
        n_iter=args.n_iter,
        joint_anderson_history=args.aa_history,
        joint_aa_start=2,
        joint_aa_damping=0.5,
        joint_aa_ridge=1e-8,
        joint_initial_step=1.0,
        solve_every={'gains': 5},
        verbose=False,
    )

    pr.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(args.top_n)
    print(s.getvalue())

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Loss:          {loss_before:.4e} → {loss_after:.4e}")
    print(f"Reduction:     {100*(loss_before - loss_after)/loss_before:.1f}%")
    print(f"Iterations:    {args.n_iter}")
    print(f"AA History:    {args.aa_history}")

    # Total time
    total_time = sum(stat[3] for stat in pr.getstats().values())
    print(f"Total time:    {total_time:.2f}s")
    print(f"Time/iter:     {total_time/args.n_iter*1e3:.0f}ms")
    print("=" * 80 + "\n")

    # Breakdown by method name
    print("Top methods by cumulative time:")
    print("-" * 80)
    stats_dict = pstats.Stats(pr)
    stats_dict.strip_dirs()
    stats_dict.sort_stats('cumulative')

    # Extract and display top methods
    for func_key, stat in sorted(
        stats_dict.stats.items(),
        key=lambda x: x[1][3],
        reverse=True
    )[:args.top_n]:
        filename, line, func_name = func_key
        ncalls, nrecursive, tottime, cumtime, callers = stat
        file_short = filename.split('/')[-1]
        print(f"{func_name:<40} {cumtime:>10.3f}s  {tottime:>10.3f}s  "
              f"calls={ncalls:>6} ({file_short}:{line})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile fit_joint_sky_beam_dirty")
    parser.add_argument("--n-iter", type=int, default=5,
                        help="Number of fit iterations (default: 5)")
    parser.add_argument("--aa-history", type=int, default=3,
                        help="Anderson acceleration history (default: 3)")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of top functions to display (default: 30)")
    args = parser.parse_args()
    main(args)
