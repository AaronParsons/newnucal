#!/usr/bin/env python
"""
Anderson Acceleration diagnostic benchmark.

Reproduces the closed_loop_test.ipynb setup with realistic forward model,
pixel masking, and alternating dirty fit, but runs only 10 iterations and
skips all visualization. Focuses on Anderson Acceleration diagnostics.

Usage:
    python benchmarks/anderson_acceleration_test.py [--sky-aa-ridge RIDGE] [--disable-aa]

Example:
    python benchmarks/anderson_acceleration_test.py --sky-aa-ridge 1e-4 --disable-aa
"""
import argparse
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'

import jax

jax.config.update("jax_enable_x64", False)

from bench_utils import setup_benchmark_calibrator


def main(args):
    """Run Anderson Acceleration diagnostic benchmark."""

    print("=" * 70)
    print("Anderson Acceleration Diagnostic Benchmark")
    print("=" * 70)

    print("\n[Setup] Building calibration problem...")
    cal, prms0, info = setup_benchmark_calibrator(
        hexnum=5, nfreq=64, ntime=24, apply_masks='alt',
    )
    loss_before = cal.calc_loss(prms0)
    print(f"[Setup] Antennas: {info['nants']},  Baselines: {info['nbls']}")
    print(f"[Setup] Freqs: {info['nfreq']},  Sky nmodes={info['sky_nmodes']},  Beam nmodes={info['beam_nmodes']}")
    print(f"[Calibrate] Initial loss: {loss_before:.4e}")

    print(f"\n[Fit] Running 10 iterations of alternating dirty fit...")
    print(f"[Fit] sky_anderson_history={args.sky_anderson_history}, sky_aa_ridge={args.sky_aa_ridge:.0e}")
    print(f"[Fit] beam_anderson_history={args.beam_anderson_history}, beam_aa_ridge={args.beam_aa_ridge:.0e}")
    print("-" * 70)

    prms, history = cal.fit_alternating_dirty(
        prms0,
        n_iter=10,
        sky_line_search_steps=[0.9],
        sky_anderson_history=args.sky_anderson_history,
        sky_aa_start=2,
        sky_aa_damping=0.45,
        sky_aa_ridge=args.sky_aa_ridge,
        beam_line_search_steps=[0.7],
        beam_anderson_history=args.beam_anderson_history,
        beam_aa_start=2,
        beam_aa_damping=0.65,
        beam_aa_ridge=args.beam_aa_ridge,
        solve_every={'gains': 20, 'gains_max': 20, 'sky_max': 5, 'beam_max': 4},
        verbose=True,
    )

    loss_after = cal.calc_loss(prms)
    print("-" * 70)
    print(f"\n[Results] Loss: {loss_before:.4e} → {loss_after:.4e}")
    print(f"[Results] Reduction: {loss_before / loss_after:.2f}×")
    print(f"[Results] Final loss: {history:.4e}")
    print("\n[Summary] Benchmark complete. Check verbose output above for AA diagnostics.")
    print("[Summary] Look for '[AA rejected]' messages to diagnose AA behavior.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anderson Acceleration diagnostic benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/anderson_acceleration_test.py
  python benchmarks/anderson_acceleration_test.py --sky-aa-ridge 1e-4
  python benchmarks/anderson_acceleration_test.py --disable-aa
        """,
    )
    parser.add_argument("--sky-anderson-history", type=int, default=3,
                        help="History vectors for sky AA (0 to disable).")
    parser.add_argument("--sky-aa-ridge", type=float, default=1e-4,
                        help="Ridge regularization for sky AA (default: 1e-4).")
    parser.add_argument("--beam-anderson-history", type=int, default=3,
                        help="History vectors for beam AA (0 to disable).")
    parser.add_argument("--beam-aa-ridge", type=float, default=1e-4,
                        help="Ridge regularization for beam AA (default: 1e-4).")
    parser.add_argument("--disable-aa", action="store_true",
                        help="Disable both AA (sets history to 0).")

    args = parser.parse_args()
    if args.disable_aa:
        args.sky_anderson_history = 0
        args.beam_anderson_history = 0
        print("[Config] Anderson Acceleration disabled via --disable-aa")

    main(args)
