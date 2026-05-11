#!/usr/bin/env python
"""Profile coarse-to-fine joint dirty fitting."""

import argparse
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

from bench_utils import block_until_ready, setup_benchmark_calibrator
from newnucal import BeamModel, SkyModel, Calibrator, init_gain_params
from newnucal.multiresolution import init_resampled_joint_state, resample_params


def _fit_kwargs(args):
    return dict(
        joint_anderson_history=args.aa_history,
        joint_aa_start=2,
        joint_aa_damping=0.5,
        joint_aa_ridge=1e-8,
        solve_every={"gains": args.solve_gains_every, "rfi": args.solve_rfi_every},
    )


def _run_state(cal, state, n_iter, *, verbose=False):
    t0 = time.perf_counter()
    state = cal.run_joint_sky_beam_dirty_state(state, n_iter=n_iter, verbose=verbose)
    block_until_ready(state.params)
    return state, time.perf_counter() - t0


def _format_taper_summary(label, cal):
    summary = cal.baseline_resolution_taper_summary()
    if summary is None:
        return f"[Taper:{label}] none"
    factor = summary["nside_excess_factor"]
    factor_s = "inf" if not np.isfinite(factor) else f"{factor:.2f}x"
    return (
        f"[Taper:{label}] down={100.0 * summary['frac_downweighted']:.1f}% "
        f"zero={100.0 * summary['frac_zero']:.1f}% "
        f"mean_w={summary['mean_weight']:.3f} min_w={summary['min_weight']:.3f} "
        f"nside={summary['sky_nside']} required~{summary['required_nside']:.1f} "
        f"margin={factor_s}"
    )


def main(args):
    print("=" * 80)
    print("Multiresolution Joint Dirty Refinement Benchmark")
    print("=" * 80)

    cal_high, prms_high0, info = setup_benchmark_calibrator(
        nfreq=args.nfreq,
        ntime=args.ntime,
        sky_nside=args.high_sky_nside,
        beam_nside=args.high_beam_nside,
        use_dpss_basis=args.use_dpss_basis,
        apply_masks=args.apply_masks,
    )
    freqs = np.asarray(cal_high.freqs)
    data = np.asarray(cal_high.data)
    array = cal_high.fwd.array
    rot_matrices = np.asarray(cal_high.rot_matrices)

    low_sky = SkyModel(args.low_sky_nside, freqs, cal_high.fwd.sky_model.basis)
    low_beam = BeamModel(args.low_beam_nside, freqs, cal_high.beam_model.basis)
    cal_low = Calibrator(
        array=array,
        beam_model=low_beam,
        sky_model=low_sky,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=data,
        method=cal_high.method,
    )
    if args.apply_masks == "standard":
        bm_wgts = cal_low.get_sky_beam_weighting()
        cal_low.apply_sky_mask(bm_wgts > bm_wgts.max() / 100)

    cal_low.set_baseline_resolution_taper(
        sky_nside=args.low_sky_nside,
        ell_factor=args.ell_factor,
        transition=args.transition,
    )
    cal_high.set_baseline_resolution_taper(
        sky_nside=args.high_sky_nside,
        ell_factor=args.ell_factor,
        transition=args.transition,
    )

    if args.low_init == "resample-high":
        prms_low0 = resample_params(prms_high0, cal_high, cal_low, sky=True, beam=True)
    else:
        rng = np.random.default_rng(args.seed)
        prms_low0 = {
            "sky_coeffs": jnp.array(
                rng.normal(0, 0.1, (low_sky.npix, low_sky.nmodes)),
                dtype=jnp.float32,
            ),
            "beam_coeffs": jnp.array(low_beam.coeffs, dtype=jnp.float32),
            **init_gain_params(cal_low.ntime, cal_low.nfreq),
        }

    low_loss0 = cal_low.calc_loss(prms_low0, explicit_beam=True)
    low_state = cal_low.init_joint_sky_beam_dirty_state(prms_low0, **_fit_kwargs(args))
    low_state, low_time = _run_state(
        cal_low, low_state, args.low_iter, verbose=args.verbose_fit
    )

    t0 = time.perf_counter()
    high_state = init_resampled_joint_state(
        cal_low,
        low_state,
        cal_high,
        resample_sky=args.refine_sky,
        resample_beam=args.refine_beam,
        **_fit_kwargs(args),
    )
    block_until_ready(high_state.params)
    resample_time = time.perf_counter() - t0
    refine_loss0 = float(high_state.loss)
    high_state, refine_time = _run_state(
        cal_high, high_state, args.high_iter, verbose=args.verbose_fit
    )

    direct_loss0 = cal_high.calc_loss(prms_high0, explicit_beam=True)
    direct_state = cal_high.init_joint_sky_beam_dirty_state(prms_high0, **_fit_kwargs(args))
    direct_state, direct_time = _run_state(
        cal_high,
        direct_state,
        args.low_iter + args.high_iter,
        verbose=args.verbose_fit,
    )

    print(f"[Setup] {info['nants']} ants, {info['nbls']} baselines, {args.nfreq} freqs")
    print(f"[Resolution] sky {args.low_sky_nside} -> {args.high_sky_nside}; "
          f"beam {args.low_beam_nside} -> {args.high_beam_nside}")
    print(_format_taper_summary("low", cal_low))
    print(_format_taper_summary("high", cal_high))
    print(f"[Low]      loss {low_loss0:.4e} -> {float(low_state.loss):.4e}  "
          f"time={low_time:.2f}s")
    print(f"[Resample] time={resample_time:.2f}s  high-start loss={refine_loss0:.4e}")
    print(f"[Refine]   loss {refine_loss0:.4e} -> {float(high_state.loss):.4e}  "
          f"time={refine_time:.2f}s")
    print(f"[Direct]   loss {direct_loss0:.4e} -> {float(direct_state.loss):.4e}  "
          f"time={direct_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-iter", type=int, default=5)
    parser.add_argument("--high-iter", type=int, default=5)
    parser.add_argument("--aa-history", type=int, default=2)
    parser.add_argument("--nfreq", type=int, default=32)
    parser.add_argument("--ntime", type=int, default=12)
    parser.add_argument("--low-sky-nside", type=int, default=8)
    parser.add_argument("--high-sky-nside", type=int, default=16)
    parser.add_argument("--low-beam-nside", type=int, default=4)
    parser.add_argument("--high-beam-nside", type=int, default=8)
    parser.add_argument("--ell-factor", type=float, default=2.0)
    parser.add_argument("--transition", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--refine-sky", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-beam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-masks", default="standard", choices=["standard", "none"])
    parser.add_argument("--use-dpss-basis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-fit", action="store_true")
    parser.add_argument("--solve-gains-every", type=int, default=4)
    parser.add_argument("--solve-rfi-every", type=int, default=2)
    parser.add_argument(
        "--low-init",
        default="resample-high",
        choices=["resample-high", "random"],
        help="How to initialize the low-resolution rung.",
    )
    args = parser.parse_args()
    if args.apply_masks == "none":
        args.apply_masks = None
    main(args)
