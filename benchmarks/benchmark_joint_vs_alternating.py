#!/usr/bin/env python
"""
Joint vs. Alternating Dirty-Map Benchmark

Compares two solvers over a fixed number of iterations:
  - fit_alternating_dirty  — sky and beam updated in alternating steps
  - fit_joint_sky_beam_dirty — sky+beam updated simultaneously from a shared adjoint

Metrics reported per solver:
  - χ² before/after
  - Wall-clock time
  - χ² reduction rate (χ²/s)
  - Time per iteration

Each solver is tested with and without Anderson acceleration.

Usage:
    python benchmarks/benchmark_joint_vs_alternating.py
    python benchmarks/benchmark_joint_vs_alternating.py --n-iter 20 --aa-history 3
"""
import argparse
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'

import jax

jax.config.update("jax_enable_x64", False)

from bench_utils import setup_benchmark_calibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_alternating(cal, prms0, n_iter, aa_history, verbose=False):
    t0 = time.perf_counter()
    state = cal.init_alternating_dirty_state(
        prms0,
        sky_initial_step=[0.9],
        sky_anderson_history=aa_history,
        sky_aa_start=2,
        sky_aa_damping=0.45,
        sky_aa_ridge=1e-8,
        beam_initial_step=[0.7],
        beam_anderson_history=aa_history,
        beam_aa_start=2,
        beam_aa_damping=0.45,
        beam_aa_ridge=1e-8,
        solve_every={'gains': 5, 'sky_max': 3, 'beam_max': 2},
    )
    cal.run_alternating_dirty_state(state, n_iter=n_iter, verbose=verbose)
    elapsed = time.perf_counter() - t0
    loss_after = cal.calc_loss(state.params)
    return loss_after, elapsed


def _run_joint(cal, prms0, n_iter, aa_history, verbose=False):
    t0 = time.perf_counter()
    state = cal.init_joint_sky_beam_dirty_state(
        prms0,
        joint_anderson_history=aa_history,
        joint_aa_start=2,
        joint_aa_damping=0.5,
        joint_aa_ridge=1e-8,
        joint_initial_step=1.0,
        solve_every={'gains': 5},
    )
    cal.run_joint_sky_beam_dirty_state(state, n_iter=n_iter, verbose=verbose)
    elapsed = time.perf_counter() - t0
    loss_after = cal.calc_loss(state.params, explicit_beam=True)
    return loss_after, elapsed


def _report(label, loss_before, loss_after, elapsed, n_iter):
    chi2_reduction = loss_before - loss_after
    chi2_rate = chi2_reduction / elapsed if elapsed > 0 else float('nan')
    pct = 100 * chi2_reduction / loss_before if loss_before > 0 else float('nan')
    print(f"  {label}")
    print(f"    χ² before:    {loss_before:.4e}")
    print(f"    χ² after:     {loss_after:.4e}  ({pct:.1f}% reduction)")
    print(f"    Wall time:    {elapsed:.1f}s  ({elapsed/n_iter*1e3:.0f} ms/iter)")
    print(f"    χ²/s rate:    {chi2_rate:.2e}")
    return dict(
        label=label, loss_before=loss_before, loss_after=loss_after,
        chi2_reduction=chi2_reduction, chi2_rate=chi2_rate,
        elapsed=elapsed, n_iter=n_iter,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("=" * 70)
    print("Joint vs. Alternating Dirty-Map Benchmark")
    print("=" * 70)

    print("\n[Setup] Building calibration problem...")
    cal, prms0, info = setup_benchmark_calibrator()
    loss_before = cal.calc_loss(prms0)
    print(f"[Setup] {info['nants']} antennas, {info['nbls']} baselines, "
          f"{info['nfreq']} freqs, {info['sky_nmodes']} sky modes, "
          f"{info['beam_nmodes']} beam modes")
    print(f"[Setup] Initial χ²: {loss_before:.4e}")
    print(f"[Setup] n_iter={args.n_iter}, aa_history={args.aa_history}")

    results = []

    # --- JIT warm-up ---
    print("\n[Warmup] Running 1 iteration of each solver to warm up JIT...")
    _run_alternating(cal, prms0, n_iter=1, aa_history=0)
    _run_joint(cal, prms0, n_iter=1, aa_history=0)
    print("[Warmup] Done.\n")

    # --- Alternating, no AA ---
    print(f"{'='*70}")
    print(f"Alternating dirty fit  (AA disabled)")
    print(f"{'='*70}")
    loss_alt, t_alt = _run_alternating(cal, prms0, args.n_iter, aa_history=0)
    results.append(_report("alternating, no AA", loss_before, loss_alt, t_alt, args.n_iter))

    # --- Alternating, with AA ---
    if args.aa_history > 0:
        print(f"\n{'='*70}")
        print(f"Alternating dirty fit  (AA history={args.aa_history})")
        print(f"{'='*70}")
        loss_alt_aa, t_alt_aa = _run_alternating(cal, prms0, args.n_iter, aa_history=args.aa_history)
        results.append(_report(f"alternating, AA={args.aa_history}", loss_before, loss_alt_aa, t_alt_aa, args.n_iter))

    # --- Joint, no AA ---
    print(f"\n{'='*70}")
    print(f"Joint sky+beam dirty fit  (AA disabled)")
    print(f"{'='*70}")
    loss_joint, t_joint = _run_joint(cal, prms0, args.n_iter, aa_history=0)
    results.append(_report("joint, no AA", loss_before, loss_joint, t_joint, args.n_iter))

    # --- Joint, with AA ---
    if args.aa_history > 0:
        print(f"\n{'='*70}")
        print(f"Joint sky+beam dirty fit  (AA history={args.aa_history})")
        print(f"{'='*70}")
        loss_joint_aa, t_joint_aa = _run_joint(cal, prms0, args.n_iter, aa_history=args.aa_history)
        results.append(_report(f"joint, AA={args.aa_history}", loss_before, loss_joint_aa, t_joint_aa, args.n_iter))

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    baseline = results[0]
    print(f"\n{'Solver':<30} {'χ²/s':>12} {'Speedup':>10} {'Final χ²':>14} {'Time(s)':>10}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*14} {'-'*10}")
    for r in results:
        speedup = r['chi2_rate'] / baseline['chi2_rate'] if baseline['chi2_rate'] > 0 else float('nan')
        print(f"{r['label']:<30} {r['chi2_rate']:>12.2e} {speedup:>10.2f}x "
              f"{r['loss_after']:>14.4e} {r['elapsed']:>10.1f}s")

    joint_idx = next(i for i, r in enumerate(results) if r['label'].startswith('joint, no AA'))
    alt_idx = 0
    speedup_joint_vs_alt = results[joint_idx]['chi2_rate'] / results[alt_idx]['chi2_rate']
    print(f"\n  Joint/Alternating χ²/s ratio (no AA): {speedup_joint_vs_alt:.2f}×")
    if speedup_joint_vs_alt > 1:
        print("  → Joint step is more χ²-efficient per second than alternating.")
    else:
        print("  → Alternating step is more χ²-efficient per second than joint.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark joint vs. alternating dirty-map solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/benchmark_joint_vs_alternating.py
  python benchmarks/benchmark_joint_vs_alternating.py --n-iter 20 --aa-history 3
  python benchmarks/benchmark_joint_vs_alternating.py --n-iter 5 --aa-history 0
        """,
    )
    parser.add_argument("--n-iter", type=int, default=10,
                        help="Iterations per solver configuration (default: 10).")
    parser.add_argument("--aa-history", type=int, default=3,
                        help="Anderson acceleration history depth (0 to skip AA runs, default: 3).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration output from each solver.")
    args = parser.parse_args()
    main(args)
