#!/usr/bin/env python
"""
Anderson Acceleration diagnostic benchmark.

Reproduces the closed_loop_test.ipynb setup with realistic forward model,
pixel masking, and alternating dirty fit, but runs only 10 iterations and
skips all visualization. Focuses on Anderson Acceleration diagnostics.

Usage:
    python benchmark/anderson_acceleration_test.py [--sky-aa-ridge RIDGE] [--disable-aa]

Example:
    python benchmark/anderson_acceleration_test.py --sky-aa-ridge 1e-4 --disable-aa
"""
import os
import sys
import argparse
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


def main(args):
    """Run Anderson Acceleration diagnostic benchmark."""

    print("=" * 70)
    print("Anderson Acceleration Diagnostic Benchmark")
    print("=" * 70)

    # --- Array ---
    array = HERAArray.from_hex(hexnum=5, sep=14.6)
    print(f"\n[Setup] Antennas: {array.nants},  Baselines: {array.nbls}")

    # --- Frequencies ---
    nfreq = 64
    freqs = np.linspace(50e6, 225e6, nfreq, dtype=np.float32)
    ref_freq = 150e6
    print(f"[Setup] Freqs: {nfreq}, ({freqs[0]/1e6:.0f}--{freqs[-1]/1e6:.0f} MHz)")

    # --- Times ---
    hera_loc = EarthLocation(lat=-30.7215 * u.deg, lon=21.4283 * u.deg, height=1073.0 * u.m)
    t0 = Time("2023-03-21T04:00:00", scale="utc")
    ntime = 24
    dt = 240 * u.min / (ntime + 1)
    times = t0 + np.arange(ntime) * dt
    rot_matrices = compute_rotation_matrices(times, hera_loc)
    print(f"[Setup] Times: {ntime},  span: {(times[-1] - times[0]).to(u.min):.1f}")

    # --- Bases: load from files like the notebook ---
    # Resolve paths relative to this script's location
    script_dir = Path(__file__).parent
    beam_file = script_dir / 'beam_basis_airy.npz'
    sky_file = script_dir / 'sky_basis_gsm_gleam.npz'

    try:
        beam_basis = BeamBasis.from_file(str(beam_file), new_freqs=freqs)
        sky_basis = SkyBasis.from_file(str(sky_file), new_freqs=freqs)
    except FileNotFoundError as e:
        print(f"\nError: Could not find basis files.")
        print(f"  Expected: {beam_file} and {sky_file}")
        print(f"  {e}")
        sys.exit(1)

    sky_nside = 32
    beam_model = BeamModel(nside=16, freqs=freqs, basis=beam_basis)
    sky_model = SkyModel(nside=sky_nside, freqs=freqs, basis=sky_basis)
    print(f"[Setup] Sky nmodes={sky_model.nmodes}, Beam nmodes={beam_model.nmodes}")

    # --- Simulate sky and gains (same as notebook) ---
    import healpy
    npix_sky = healpy.nside2npix(sky_nside)
    rng = np.random.default_rng(42)
    spectral_indices = rng.normal(-0.7, 0.1, npix_sky).astype(np.float32)
    ref_flux = rng.exponential(scale=1.0, size=npix_sky).astype(np.float32)
    flux_true = ref_flux[:, None] * (freqs[None, :] / ref_freq) ** spectral_indices[:, None]
    sky_coeffs_true = jnp.array(sky_basis.project(flux_true), dtype=jnp.float32)
    beam_coeffs_true = jnp.array(beam_model.coeffs, dtype=jnp.float32)

    # --- Simulate visibilities with gains ---
    print(f"\n[Simulate] Building forward model...")
    fwd = ForwardModel(array, sky_model, beam_model, freqs, eps=1e-5)
    vis_true = fwd.simulate_2d(sky_coeffs_true, jnp.array(rot_matrices))
    print(f"[Simulate] vis_true: {vis_true.shape}, mean |V|: {float(jnp.abs(vis_true).mean()):.4f}")

    # Apply smooth gain perturbations (same as notebook)
    t_norm = np.linspace(0, 1, ntime)[:, None]
    f_norm = np.linspace(0, 1, nfreq)[None, :]
    true_log_amp = (0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)).astype(np.float32)
    true_phase = (0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)).astype(np.float32)
    true_phi = np.zeros((ntime, 2, nfreq), dtype=np.float32)
    vis_data = apply_gains(vis_true, jnp.array(true_log_amp), jnp.array(true_phase),
                          jnp.array(true_phi), jnp.array(array.bls, dtype=jnp.float32))

    # --- Build calibrator ---
    print(f"[Calibrate] Building calibrator...")
    cal = Calibrator(
        array=array, sky_model=sky_model, beam_model=beam_model,
        freqs=freqs, rot_matrices=rot_matrices, data=vis_data, eps=1e-5, method='2d'
    )

    # --- Remove permanently below-horizon pixels and apply beam weighting ---
    sky_alt_mask = cal.build_sky_mask_altitude(min_altitude_deg=0.0)
    bm_wgts = cal.get_sky_beam_weighting()
    threshold = bm_wgts.max() / 100
    beam_weight_mask = bm_wgts > threshold
    # Combine masks: keep pixels that pass both altitude and beam weighting
    sky_mask = sky_alt_mask & beam_weight_mask
    cal.apply_sky_mask(sky_mask)

    # --- Apply beam altitude mask ---
    beam_alt_mask = cal.build_beam_mask_altitude(max_zenith_angle_deg=90.0)
    cal.apply_beam_mask(beam_alt_mask)
    print(f"[Calibrate] Sky pixel mask: {sky_mask.sum()} pixels retained, {(~sky_mask).sum()} removed")

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

    loss_before = cal.calc_loss(prms0)
    print(f"[Calibrate] Initial loss: {loss_before:.4e}")

    # --- Run alternating dirty fit with diagnostics ---
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
    print("[Summary] Look for '[AA rejected]' and '[AA filtered]' messages to diagnose AA behavior.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anderson Acceleration diagnostic benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark/anderson_acceleration_test.py
  python benchmark/anderson_acceleration_test.py --sky-aa-ridge 1e-4
  python benchmark/anderson_acceleration_test.py --disable-aa
        """,
    )
    parser.add_argument(
        "--sky-anderson-history",
        type=int,
        default=3,
        help="Number of history vectors for sky AA (default: 3). Use 0 to disable.",
    )
    parser.add_argument(
        "--sky-aa-ridge",
        type=float,
        default=1e-4,
        help="Ridge regularization for sky AA Gram matrix (default: 1e-4)",
    )
    parser.add_argument(
        "--beam-anderson-history",
        type=int,
        default=3,
        help="Number of history vectors for beam AA (default: 3). Use 0 to disable.",
    )
    parser.add_argument(
        "--beam-aa-ridge",
        type=float,
        default=1e-4,
        help="Ridge regularization for beam AA Gram matrix (default: 1e-4)",
    )
    parser.add_argument(
        "--disable-aa",
        action="store_true",
        help="Disable both sky and beam Anderson Acceleration (sets history to 0)",
    )

    args = parser.parse_args()

    if args.disable_aa:
        args.sky_anderson_history = 0
        args.beam_anderson_history = 0
        print("[Config] Anderson Acceleration disabled via --disable-aa")

    main(args)
