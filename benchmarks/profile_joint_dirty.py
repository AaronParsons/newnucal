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
import time
from io import StringIO

os.environ['OMP_NUM_THREADS'] = '1'

import jax
jax.config.update("jax_enable_x64", False)

from bench_utils import block_until_ready, setup_benchmark_calibrator, time_call


def main(args):
    print("=" * 80)
    print("Joint Sky+Beam Dirty-Map Profiling Benchmark")
    print("=" * 80)

    print("\n[Setup] Building calibration problem...")
    cal, prms0, info = setup_benchmark_calibrator(
        nfreq=args.nfreq,
        ntime=args.ntime,
        sky_nside=args.sky_nside,
        beam_nside=args.beam_nside,
        use_dpss_basis=args.use_dpss_basis,
        apply_masks=args.apply_masks,
    )
    loss_before = cal.calc_loss(prms0)
    print(f"[Setup] {info['nants']} antennas, {info['nbls']} baselines, "
          f"{info['nfreq']} freqs")
    print(f"[Setup] Initial χ²: {loss_before:.4e}")
    print(f"[Setup] Running {args.n_iter} iterations with aa_history={args.aa_history}\n")

    # JIT warm-up
    print("[Warmup] Compiling JAX functions...")
    time_call("calc_loss warmup", lambda: cal.calc_loss(prms0), repeats=1)
    print("[Warmup] Done.\n")

    # Profile the fit
    print("=" * 80)
    print(f"Profiling fit_joint_sky_beam_dirty ({args.n_iter} iterations)")
    print("=" * 80 + "\n")

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()

    fit_kwargs = {}
    if args.legacy_cadence:
        fit_kwargs.update(joint_initial_step=1.0, solve_every={'gains': 5})

    params, loss_after = cal.fit_joint_sky_beam_dirty(
        prms0,
        n_iter=args.n_iter,
        joint_anderson_history=args.aa_history,
        joint_aa_start=2,
        joint_aa_damping=0.5,
        joint_aa_ridge=1e-8,
        verbose=False,
        **fit_kwargs,
    )
    block_until_ready(params)

    pr.disable()
    total_time = time.perf_counter() - t0

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
    parser.add_argument("--nfreq", type=int, default=32,
                        help="Number of frequency channels (default: 32)")
    parser.add_argument("--ntime", type=int, default=12,
                        help="Number of time samples (default: 12)")
    parser.add_argument("--sky-nside", type=int, default=16,
                        help="Sky HEALPix nside (default: 16)")
    parser.add_argument("--beam-nside", type=int, default=8,
                        help="Beam HEALPix nside (default: 8)")
    parser.add_argument("--apply-masks", default="standard",
                        choices=["standard", "alt", "none"],
                        help="Mask mode for setup_benchmark_calibrator (default: standard)")
    parser.add_argument("--use-dpss-basis", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use generated DPSS bases instead of benchmark basis files")
    parser.add_argument("--legacy-cadence", action="store_true",
                        help="Use the old joint_initial_step=1.0 and solve_every={'gains': 5}")
    args = parser.parse_args()
    if args.apply_masks == "none":
        args.apply_masks = None
    main(args)
