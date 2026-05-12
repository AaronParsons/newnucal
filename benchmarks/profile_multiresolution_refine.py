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
from newnucal.multiresolution import (
    init_resampled_joint_state,
    resample_params,
)


def _fit_kwargs(args):
    return dict(
        joint_anderson_history=args.aa_history,
        joint_aa_start=2,
        joint_aa_damping=0.5,
        joint_aa_ridge=1e-8,
        solve_every={"gains": args.solve_gains_every, "rfi": args.solve_rfi_every},
    )


def _resample_kwargs(args):
    return dict(
        sky_healpix_power=args.sky_healpix_power,
        beam_healpix_power=args.beam_healpix_power,
        beam_normalize=args.beam_normalize,
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


def _reset_run_weights(cal):
    # Matrix cases share calibrators; reset RFI weights between runs.
    cal.set_channel_weights(None)


def _run_direct_reference(cal_high, prms_high0, args, n_iter):
    _reset_run_weights(cal_high)
    loss0 = cal_high.calc_loss(prms_high0, explicit_beam=True)
    state = cal_high.init_joint_sky_beam_dirty_state(prms_high0, **_fit_kwargs(args))
    state, runtime = _run_state(
        cal_high, state, n_iter, verbose=args.verbose_fit
    )
    return {
        "loss0": float(loss0),
        "final_loss": float(state.loss),
        "time": float(runtime),
        "iters": int(n_iter),
    }


def _run_multiresolution_case(
    label,
    cal_low,
    cal_high,
    prms_low0,
    args,
    *,
    params_high_reference=None,
    low_iter,
    high_iter,
    resample_sky,
    resample_beam,
    warm_iter=0,
    high_low_support_weights=None,
):
    _reset_run_weights(cal_low)
    _reset_run_weights(cal_high)
    high_support_weights = np.array(cal_high._visibility_weights_np, copy=True)

    low_loss0 = cal_low.calc_loss(prms_low0, explicit_beam=True)
    low_state = cal_low.init_joint_sky_beam_dirty_state(prms_low0, **_fit_kwargs(args))
    low_state, low_time = _run_state(
        cal_low, low_state, low_iter, verbose=args.verbose_fit
    )

    t0 = time.perf_counter()
    high_state = init_resampled_joint_state(
        cal_low,
        low_state,
        cal_high,
        resample_sky=resample_sky,
        resample_beam=resample_beam,
        **_resample_kwargs(args),
        **_fit_kwargs(args),
    )
    block_until_ready(high_state.params)
    resample_time = time.perf_counter() - t0

    if params_high_reference is not None and (not resample_sky or not resample_beam):
        params = dict(high_state.params)
        if not resample_sky and params_high_reference.get("sky_coeffs") is not None:
            params["sky_coeffs"] = params_high_reference["sky_coeffs"]
        if not resample_beam and params_high_reference.get("beam_coeffs") is not None:
            params["beam_coeffs"] = params_high_reference["beam_coeffs"]
        high_state = cal_high.init_joint_sky_beam_dirty_state(
            params, **_fit_kwargs(args)
        )

    high_start_loss = float(high_state.loss)
    high_start_low_support_loss = None
    warm_time = 0.0
    warm_loss = None

    if high_low_support_weights is not None:
        cal_high.set_visibility_weights(high_low_support_weights)
        high_start_low_support_loss = cal_high.calc_loss(
            high_state.params, explicit_beam=True
        )
        if warm_iter > 0:
            high_state = cal_high.init_joint_sky_beam_dirty_state(
                high_state.params, **_fit_kwargs(args)
            )
            high_state, warm_time = _run_state(
                cal_high, high_state, warm_iter, verbose=args.verbose_fit
            )
            warm_loss = float(high_state.loss)
        cal_high.set_visibility_weights(high_support_weights)
        high_state = cal_high.init_joint_sky_beam_dirty_state(
            high_state.params, **_fit_kwargs(args)
        )

    high_state, refine_time = _run_state(
        cal_high, high_state, high_iter, verbose=args.verbose_fit
    )

    return {
        "label": label,
        "low_loss0": float(low_loss0),
        "low_loss": float(low_state.loss),
        "high_start_loss": high_start_loss,
        "high_start_low_support_loss": (
            None if high_start_low_support_loss is None
            else float(high_start_low_support_loss)
        ),
        "warm_loss": warm_loss,
        "final_loss": float(high_state.loss),
        "low_time": float(low_time),
        "resample_time": float(resample_time),
        "warm_time": float(warm_time),
        "refine_time": float(refine_time),
        "total_time": float(low_time + resample_time + warm_time + refine_time),
        "low_iter": int(low_iter),
        "high_iter": int(high_iter),
        "warm_iter": int(warm_iter),
        "resample_sky": bool(resample_sky),
        "resample_beam": bool(resample_beam),
    }


def _print_matrix_table(rows, direct):
    print("\n[Matrix]")
    print(
        "case          low warm high sky beam     low_start       low_end"
        " high_start(low) high_start(high)       final      time  final/direct"
    )
    for row in rows:
        ratio = row["final_loss"] / max(direct["final_loss"], 1e-30)
        high_low = row["high_start_low_support_loss"]
        high_low_s = "-" if high_low is None else f"{high_low:.4e}"
        print(
            f"{row['label']:<13}"
            f"{row['low_iter']:>5d} {row['warm_iter']:>4d} {row['high_iter']:>4d}"
            f" {str(row['resample_sky'])[0]:>3} {str(row['resample_beam'])[0]:>4}"
            f" {row['low_loss0']:>13.4e} {row['low_loss']:>13.4e}"
            f" {high_low_s:>15} {row['high_start_loss']:>16.4e}"
            f" {row['final_loss']:>11.4e}"
            f" {row['total_time']:>9.2f}s {ratio:>12.3f}"
        )
    print(
        f"{'direct':<13}{0:>5d} {0:>4d} {direct['iters']:>4d}"
        f" {'-':>3} {'-':>4}"
        f" {'-':>13} {'-':>13}"
        f" {'-':>15} {direct['loss0']:>16.4e} {direct['final_loss']:>11.4e}"
        f" {direct['time']:>9.2f}s {1.0:>12.3f}"
    )


def _roundtrip_row(label, params_high, cal_high, cal_low, args, *, sky, beam):
    t0 = time.perf_counter()
    params_low = resample_params(
        params_high,
        cal_high,
        cal_low,
        sky=sky,
        beam=beam,
        **_resample_kwargs(args),
    )
    params_roundtrip = resample_params(
        params_low,
        cal_low,
        cal_high,
        sky=sky,
        beam=beam,
        **_resample_kwargs(args),
    )
    if not sky and params_high.get("sky_coeffs") is not None:
        params_roundtrip["sky_coeffs"] = params_high["sky_coeffs"]
    if not beam and params_high.get("beam_coeffs") is not None:
        params_roundtrip["beam_coeffs"] = params_high["beam_coeffs"]
    block_until_ready(params_roundtrip)
    runtime = time.perf_counter() - t0

    high_loss = cal_high.calc_loss(params_high, explicit_beam=True)
    low_loss = cal_low.calc_loss(params_low, explicit_beam=True)
    roundtrip_loss = cal_high.calc_loss(params_roundtrip, explicit_beam=True)
    return {
        "params_low": params_low,
        "params_roundtrip": params_roundtrip,
        "label": label,
        "sky": bool(sky),
        "beam": bool(beam),
        "high_loss": float(high_loss),
        "low_loss": float(low_loss),
        "roundtrip_loss": float(roundtrip_loss),
        "ratio": float(roundtrip_loss / max(high_loss, 1e-30)),
        "time": float(runtime),
    }


def _print_roundtrip_bins(cal_high, params_high, params_roundtrip, *, nbins):
    if nbins <= 0:
        return
    sky_ref = cal_high._ensure_sky_is_active(params_high["sky_coeffs"])
    sky_rt = cal_high._ensure_sky_is_active(params_roundtrip["sky_coeffs"])
    beam = cal_high._beam_coeffs_full(params_high.get("beam_coeffs"))

    vis_ref = cal_high._jit_simulate_variable_beam(
        sky_ref, beam, cal_high.rot_matrices
    )
    vis_rt = cal_high._jit_simulate_variable_beam(
        sky_rt, beam, cal_high.rot_matrices
    )
    vis_ref, vis_rt = jax.device_get((vis_ref, vis_rt))
    resid_power = np.mean(np.abs(vis_rt - vis_ref) ** 2, axis=(0, 1))
    ref_power = np.mean(np.abs(vis_ref) ** 2, axis=(0, 1))

    bl_len = np.linalg.norm(np.asarray(cal_high.bls), axis=1)
    edges = np.linspace(float(np.min(bl_len)), float(np.max(bl_len)), nbins + 1)
    print("\n[Sky Roundtrip Baseline Bins]")
    print("bin      bl_min      bl_max    nbl    rel_rms    resid_power     ref_power")
    for bi in range(nbins):
        if bi == nbins - 1:
            mask = (bl_len >= edges[bi]) & (bl_len <= edges[bi + 1])
        else:
            mask = (bl_len >= edges[bi]) & (bl_len < edges[bi + 1])
        if not np.any(mask):
            continue
        rp = float(np.mean(resid_power[mask]))
        fp = float(np.mean(ref_power[mask]))
        rel = np.sqrt(rp / max(fp, 1e-30))
        print(
            f"{bi:3d} {edges[bi]:11.3f} {edges[bi + 1]:11.3f}"
            f" {int(np.sum(mask)):6d} {rel:10.3e} {rp:14.4e} {fp:13.4e}"
        )


def _print_roundtrip_table(rows):
    print("\n[Roundtrip]")
    print(
        "case          sky beam      high_loss       low_loss"
        "   roundtrip_loss  round/high      time"
    )
    for row in rows:
        print(
            f"{row['label']:<13}"
            f" {str(row['sky'])[0]:>3} {str(row['beam'])[0]:>4}"
            f" {row['high_loss']:>14.4e} {row['low_loss']:>14.4e}"
            f" {row['roundtrip_loss']:>16.4e} {row['ratio']:>11.3f}"
            f" {row['time']:>9.2f}s"
        )


def _parse_int_schedule(text):
    if text is None or text.strip() == "":
        return []
    return [int(tok) for tok in text.split(",") if tok.strip()]


def _run_support_schedule(cal_low, cal_high, prms_low0, prms_high0, args):
    if not args.support_schedule:
        return

    _reset_run_weights(cal_low)
    _reset_run_weights(cal_high)
    support_nsides = _parse_int_schedule(args.support_schedule)
    if not support_nsides:
        raise ValueError("--support-schedule must contain at least one NSIDE")

    low_state = cal_low.init_joint_sky_beam_dirty_state(prms_low0, **_fit_kwargs(args))
    low_state, low_time = _run_state(
        cal_low, low_state, args.low_iter, verbose=args.verbose_fit
    )

    t0 = time.perf_counter()
    high_state = init_resampled_joint_state(
        cal_low,
        low_state,
        cal_high,
        resample_sky=True,
        resample_beam=True,
        **_resample_kwargs(args),
        **_fit_kwargs(args),
    )
    block_until_ready(high_state.params)
    resample_time = time.perf_counter() - t0

    rows = []
    total_time = float(low_time + resample_time)
    for stage, support_nside in enumerate(support_nsides):
        cal_high.set_baseline_resolution_taper(
            sky_nside=support_nside,
            ell_factor=args.ell_factor,
            transition=args.transition,
        )
        start_loss = cal_high.calc_loss(high_state.params, explicit_beam=True)
        state = cal_high.init_joint_sky_beam_dirty_state(
            high_state.params, **_fit_kwargs(args)
        )
        state, stage_time = _run_state(
            cal_high, state, args.stage_iter, verbose=args.verbose_fit
        )
        total_time += float(stage_time)
        high_state = state
        summary = cal_high.baseline_resolution_taper_summary()
        rows.append({
            "stage": stage,
            "support_nside": int(support_nside),
            "start_loss": float(start_loss),
            "end_loss": float(high_state.loss),
            "time": float(stage_time),
            "down": summary["frac_downweighted"] if summary else 0.0,
            "zero": summary["frac_zero"] if summary else 0.0,
            "mean_weight": summary["mean_weight"] if summary else 1.0,
        })

    direct = _run_direct_reference(
        cal_high, prms_high0, args, args.low_iter + args.stage_iter * len(support_nsides)
    )
    print("\n[Support Schedule]")
    print(
        f"low_time={low_time:.2f}s resample_time={resample_time:.2f}s "
        f"total_time={total_time:.2f}s direct_time={direct['time']:.2f}s"
    )
    print("stage support  down% zero% mean_w     start_loss       end_loss      time  end/direct")
    for row in rows:
        ratio = row["end_loss"] / max(direct["final_loss"], 1e-30)
        print(
            f"{row['stage']:5d} {row['support_nside']:7d}"
            f" {100.0 * row['down']:6.1f} {100.0 * row['zero']:5.1f}"
            f" {row['mean_weight']:6.3f}"
            f" {row['start_loss']:14.4e} {row['end_loss']:14.4e}"
            f" {row['time']:8.2f}s {ratio:11.3f}"
        )
    print(
        f"{'direct':>13} {'':>13} {'':>13} {'':>8}"
        f" {direct['loss0']:14.4e} {direct['final_loss']:14.4e}"
        f" {direct['time']:8.2f}s {1.0:11.3f}"
    )


def main(args):
    print("=" * 80)
    print("Multiresolution Joint Dirty Refinement Benchmark")
    print("=" * 80)

    cal_high, prms_high0, info = setup_benchmark_calibrator(
        hexnum=args.hexnum,
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
    high_low_support_weights = cal_high.baseline_resolution_taper(
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
        prms_low0 = resample_params(
            prms_high0,
            cal_high,
            cal_low,
            sky=True,
            beam=True,
            **_resample_kwargs(args),
        )
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

    print(f"[Setup] {info['nants']} ants, {info['nbls']} baselines, {args.nfreq} freqs")
    print(f"[Resolution] sky {args.low_sky_nside} -> {args.high_sky_nside}; "
          f"beam {args.low_beam_nside} -> {args.high_beam_nside}")
    print(
        "[Resample] "
        f"sky_power={args.sky_healpix_power:g} "
        f"beam_power={args.beam_healpix_power:g} "
        f"beam_normalize={args.beam_normalize}"
    )
    print(_format_taper_summary("low", cal_low))
    if args.inherited_taper:
        inherited_summary = cal_high._baseline_resolution_taper_metadata(
            high_low_support_weights,
            sky_nside=args.low_sky_nside,
            ell_factor=args.ell_factor,
            transition=args.transition,
        )
        inherited_factor = inherited_summary["nside_excess_factor"]
        inherited_factor_s = (
            "inf" if not np.isfinite(inherited_factor)
            else f"{inherited_factor:.2f}x"
        )
        print(
            "[Taper:high-inherited] "
            f"down={100.0 * inherited_summary['frac_downweighted']:.1f}% "
            f"zero={100.0 * inherited_summary['frac_zero']:.1f}% "
            f"mean_w={inherited_summary['mean_weight']:.3f} "
            f"min_w={inherited_summary['min_weight']:.3f} "
            f"nside={inherited_summary['sky_nside']} "
            f"required~{inherited_summary['required_nside']:.1f} "
            f"margin={inherited_factor_s}"
        )
    print(_format_taper_summary("high", cal_high))

    if args.support_schedule:
        _run_support_schedule(cal_low, cal_high, prms_low0, prms_high0, args)
    elif args.roundtrip:
        _reset_run_weights(cal_low)
        _reset_run_weights(cal_high)
        rows = [
            _roundtrip_row("both", prms_high0, cal_high, cal_low, args, sky=True, beam=True),
            _roundtrip_row("sky-only", prms_high0, cal_high, cal_low, args, sky=True, beam=False),
            _roundtrip_row("beam-only", prms_high0, cal_high, cal_low, args, sky=False, beam=True),
        ]
        _print_roundtrip_table(rows)
        sky_row = next(row for row in rows if row["label"] == "sky-only")
        _print_roundtrip_bins(
            cal_high,
            prms_high0,
            sky_row["params_roundtrip"],
            nbins=args.roundtrip_bins,
        )
    elif args.matrix:
        rows = [
            _run_multiresolution_case(
                "both",
                cal_low,
                cal_high,
                prms_low0,
                args,
                params_high_reference=prms_high0,
                low_iter=args.low_iter,
                high_iter=args.high_iter,
                resample_sky=True,
                resample_beam=True,
                high_low_support_weights=(
                    high_low_support_weights if args.inherited_taper else None
                ),
            ),
            _run_multiresolution_case(
                "sky-only",
                cal_low,
                cal_high,
                prms_low0,
                args,
                params_high_reference=prms_high0,
                low_iter=args.low_iter,
                high_iter=args.high_iter,
                resample_sky=True,
                resample_beam=False,
                high_low_support_weights=(
                    high_low_support_weights if args.inherited_taper else None
                ),
            ),
            _run_multiresolution_case(
                "beam-only",
                cal_low,
                cal_high,
                prms_low0,
                args,
                params_high_reference=prms_high0,
                low_iter=args.low_iter,
                high_iter=args.high_iter,
                resample_sky=False,
                resample_beam=True,
                high_low_support_weights=(
                    high_low_support_weights if args.inherited_taper else None
                ),
            ),
            _run_multiresolution_case(
                "handoff-only",
                cal_low,
                cal_high,
                prms_low0,
                args,
                params_high_reference=prms_high0,
                low_iter=0,
                high_iter=args.high_iter,
                resample_sky=True,
                resample_beam=True,
                high_low_support_weights=(
                    high_low_support_weights if args.inherited_taper else None
                ),
            ),
            _run_multiresolution_case(
                "continuation",
                cal_low,
                cal_high,
                prms_low0,
                args,
                params_high_reference=prms_high0,
                low_iter=args.low_iter,
                high_iter=args.high_iter,
                warm_iter=args.warm_iter,
                resample_sky=True,
                resample_beam=True,
                high_low_support_weights=high_low_support_weights,
            ),
        ]
        direct = _run_direct_reference(
            cal_high, prms_high0, args, args.low_iter + args.high_iter
        )
        _print_matrix_table(rows, direct)
    else:
        row = _run_multiresolution_case(
            "single",
            cal_low,
            cal_high,
            prms_low0,
            args,
            params_high_reference=prms_high0,
            low_iter=args.low_iter,
            high_iter=args.high_iter,
            resample_sky=args.refine_sky,
            resample_beam=args.refine_beam,
            warm_iter=args.warm_iter,
            high_low_support_weights=(
                high_low_support_weights if args.inherited_taper else None
            ),
        )
        direct = _run_direct_reference(
            cal_high, prms_high0, args, args.low_iter + args.high_iter
        )
        print(f"[Low]      loss {row['low_loss0']:.4e} -> {row['low_loss']:.4e}  "
              f"time={row['low_time']:.2f}s")
        print(
            f"[Resample] time={row['resample_time']:.2f}s  "
            f"high-start loss={row['high_start_loss']:.4e}"
        )
        print(f"[Refine]   loss {row['high_start_loss']:.4e} -> {row['final_loss']:.4e}  "
              f"time={row['refine_time']:.2f}s")
        print(f"[Direct]   loss {direct['loss0']:.4e} -> {direct['final_loss']:.4e}  "
              f"time={direct['time']:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-iter", type=int, default=5)
    parser.add_argument("--high-iter", type=int, default=5)
    parser.add_argument("--aa-history", type=int, default=2)
    parser.add_argument("--hexnum", type=int, default=5)
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
    parser.add_argument("--sky-healpix-power", type=float, default=0.0)
    parser.add_argument("--beam-healpix-power", type=float, default=0.0)
    parser.add_argument(
        "--beam-normalize",
        default="none",
        choices=["none", "max", "mean"],
    )
    parser.add_argument("--apply-masks", default="standard", choices=["standard", "none"])
    parser.add_argument("--use-dpss-basis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-fit", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="Only measure high->low->high resampling loss without fitting.",
    )
    parser.add_argument(
        "--roundtrip-bins",
        type=int,
        default=0,
        help="Print sky-only roundtrip visibility residuals binned by baseline length.",
    )
    parser.add_argument(
        "--support-schedule",
        default="",
        help="Comma-separated effective sky NSIDEs for staged high-resolution support refinement.",
    )
    parser.add_argument(
        "--stage-iter",
        type=int,
        default=3,
        help="Iterations to run at each --support-schedule stage.",
    )
    parser.add_argument(
        "--inherited-taper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate resampled high models under the low-resolution support.",
    )
    parser.add_argument(
        "--warm-iter",
        type=int,
        default=0,
        help="High-resolution iterations to run with inherited low-resolution support before final support.",
    )
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
