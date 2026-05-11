#!/usr/bin/env python
"""
Profiling benchmark for fit_joint_sky_beam_dirty.

Uses cProfile to measure per-subroutine time and call counts during
joint sky+beam dirty-map fitting.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import cProfile
import os
import pstats
import time
import traceback
from io import StringIO

os.environ['OMP_NUM_THREADS'] = '1'

import jax
jax.config.update("jax_enable_x64", False)

from bench_utils import block_until_ready, setup_benchmark_calibrator, time_call


@contextmanager
def _scalar_sync_tracer(enabled=False, top_n=12):
    """Trace JAX scalar/array host conversions during benchmark windows."""
    if not enabled:
        yield None
        return

    array_cls = type(jax.numpy.array(1.0))
    orig_float = array_cls.__float__
    orig_array = array_cls.__array__
    counts = Counter()

    def _caller_key(kind):
        stack = traceback.extract_stack()[:-2]
        for frame in reversed(stack):
            if frame.filename == __file__:
                continue
            if "site-packages/jax" in frame.filename:
                continue
            if "site-packages/jaxlib" in frame.filename:
                continue
            return (kind, frame.filename, frame.lineno, frame.name, frame.line or "")
        frame = stack[-1]
        return (kind, frame.filename, frame.lineno, frame.name, frame.line or "")

    def traced_float(self):
        counts[_caller_key("__float__")] += 1
        return orig_float(self)

    def traced_array(self, *args, **kwargs):
        counts[_caller_key("__array__")] += 1
        return orig_array(self, *args, **kwargs)

    array_cls.__float__ = traced_float
    array_cls.__array__ = traced_array
    try:
        yield counts
    finally:
        array_cls.__float__ = orig_float
        array_cls.__array__ = orig_array
        print("Scalar/array host conversion call sites:")
        print("-" * 80)
        for (kind, filename, lineno, name, line), count in counts.most_common(top_n):
            file_short = filename.split("/")[-1]
            print(f"{count:5d} {kind:<10} {file_short}:{lineno} {name}  {line}")
        if not counts:
            print("(none)")
        print("=" * 80 + "\n")


def _fit_kwargs(args):
    fit_kwargs = {}
    if args.legacy_cadence:
        fit_kwargs.update(joint_initial_step=1.0, solve_every={'gains': 5})
    return fit_kwargs


def _setup_case(args):
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
    print("[Warmup] Compiling JAX functions...")
    time_call("calc_loss warmup", lambda: cal.calc_loss(prms0), repeats=1)
    print("[Warmup] Done.\n")
    return cal, prms0, loss_before


def _print_profile(pr, args):
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(args.top_n)
    print(s.getvalue())


def _print_method_breakdown(pr, args):
    print("Top methods by cumulative time:")
    print("-" * 80)
    stats_dict = pstats.Stats(pr)
    stats_dict.strip_dirs()
    stats_dict.sort_stats('cumulative')

    for func_key, stat in sorted(
        stats_dict.stats.items(),
        key=lambda x: x[1][3],
        reverse=True,
    )[:args.top_n]:
        filename, line, func_name = func_key
        ncalls, nrecursive, tottime, cumtime, callers = stat
        file_short = filename.split('/')[-1]
        print(f"{func_name:<40} {cumtime:>10.3f}s  {tottime:>10.3f}s  "
              f"calls={ncalls:>6} ({file_short}:{line})")
    print("=" * 80 + "\n")


def _profile_fit(args):
    cal, prms0, loss_before = _setup_case(args)
    print("=" * 80)
    print(f"Profiling fit_joint_sky_beam_dirty ({args.n_iter} iterations, compile included)")
    print("=" * 80 + "\n")

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    with _scalar_sync_tracer(args.trace_scalar_syncs, args.top_n):
        pr.enable()
        params, loss_after = cal.fit_joint_sky_beam_dirty(
            prms0,
            n_iter=args.n_iter,
            joint_anderson_history=args.aa_history,
            joint_aa_start=2,
            joint_aa_damping=0.5,
            joint_aa_ridge=1e-8,
            verbose=False,
            **_fit_kwargs(args),
        )
        block_until_ready(params)
        pr.disable()
    total_time = time.perf_counter() - t0

    _print_profile(pr, args)

    print("=" * 80)
    print("SUMMARY: COMPILE-INCLUDED FIT")
    print("=" * 80)
    print(f"Loss:          {loss_before:.4e} → {loss_after:.4e}")
    print(f"Reduction:     {100*(loss_before - loss_after)/loss_before:.1f}%")
    print(f"Iterations:    {args.n_iter}")
    print(f"AA History:    {args.aa_history}")

    print(f"Total time:    {total_time:.2f}s")
    print(f"Time/iter:     {total_time/args.n_iter*1e3:.0f}ms")
    print("=" * 80 + "\n")
    _print_method_breakdown(pr, args)


def _profile_warmed_state(args):
    cal, prms0, loss_before = _setup_case(args)
    print("=" * 80)
    print(
        f"Warming joint dirty state for {args.warmup_steps} steps, "
        f"then profiling {args.n_iter} steps"
    )
    print("=" * 80 + "\n")

    state = cal.init_joint_sky_beam_dirty_state(
        prms0,
        joint_anderson_history=args.aa_history,
        joint_aa_start=2,
        joint_aa_damping=0.5,
        joint_aa_ridge=1e-8,
        **_fit_kwargs(args),
    )
    if args.warmup_steps > 0:
        t0_warm = time.perf_counter()
        state = cal.run_joint_sky_beam_dirty_state(
            state, n_iter=args.warmup_steps, verbose=False
        )
        block_until_ready(state.params)
        warm_time = time.perf_counter() - t0_warm
    else:
        block_until_ready(state.params)
        warm_time = 0.0
    loss_profile_start = float(state.loss)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    with _scalar_sync_tracer(args.trace_scalar_syncs, args.top_n):
        pr.enable()
        state = cal.run_joint_sky_beam_dirty_state(state, n_iter=args.n_iter, verbose=False)
        block_until_ready(state.params)
        pr.disable()
    total_time = time.perf_counter() - t0
    loss_after = float(state.loss)

    _print_profile(pr, args)

    print("=" * 80)
    print("SUMMARY: WARMED STATE")
    print("=" * 80)
    print(f"Initial loss:         {loss_before:.4e}")
    print(f"Profile window loss:  {loss_profile_start:.4e} → {loss_after:.4e}")
    print(f"Profile reduction:    {100*(loss_profile_start - loss_after)/loss_profile_start:.1f}%")
    print(f"Warmup steps:         {args.warmup_steps}")
    print(f"Profile iterations:   {args.n_iter}")
    print(f"AA History:           {args.aa_history}")
    print(f"Warmup time:          {warm_time:.2f}s")
    print(f"Profile time:         {total_time:.2f}s")
    print(f"Profile time/iter:    {total_time/args.n_iter*1e3:.0f}ms")
    print("=" * 80 + "\n")
    _print_method_breakdown(pr, args)


def main(args):
    print("=" * 80)
    print("Joint Sky+Beam Dirty-Map Profiling Benchmark")
    print("=" * 80)

    if args.profile_mode in ("fit", "both"):
        _profile_fit(args)
    if args.profile_mode in ("warmed-state", "both"):
        _profile_warmed_state(args)


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
    parser.add_argument("--profile-mode", default="fit",
                        choices=["fit", "warmed-state", "both"],
                        help=("Profile compile-included fit, warmed state-machine "
                              "steps, or both (default: fit)"))
    parser.add_argument("--warmup-steps", type=int, default=2,
                        help="State-machine steps to run before warmed-state profiling")
    parser.add_argument("--trace-scalar-syncs", action="store_true",
                        help="Print call sites for JAX scalar/array host conversions")
    args = parser.parse_args()
    if args.apply_masks == "none":
        args.apply_masks = None
    main(args)
