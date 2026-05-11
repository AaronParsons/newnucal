"""Shared setup for newnucal benchmarks."""
import sys
import time
from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from newnucal import (
    HERAArray, BeamModel, BeamBasis, SkyBasis, SkyModel,
    ForwardModel, Calibrator, apply_gains, init_gain_params,
)
from newnucal.simulate import compute_rotation_matrices

_BENCH_DIR = Path(__file__).parent

HERA_LOC = EarthLocation(lat=-30.7215 * u.deg, lon=21.4283 * u.deg, height=1073.0 * u.m)


def make_rot_matrices(ntime=24, t0_str="2023-03-21T04:00:00", span_min=240.0):
    t0 = Time(t0_str, scale="utc")
    dt = span_min * u.min / (ntime + 1)
    times = t0 + np.arange(ntime) * dt
    return compute_rotation_matrices(times, HERA_LOC), times


def block_until_ready(x):
    """Block a JAX pytree so benchmark timings include device work."""
    return jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )


def time_call(label, fn, repeats=5):
    """Time a callable after one warmup call and print a compact summary."""
    block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    mean = float(np.mean(samples))
    print(f"{label:28s} mean={mean:.5f}s min={min(samples):.5f}s max={max(samples):.5f}s")
    return mean


def setup_lightweight_masked_joint_case(
    nfreq=16,
    ntime=2,
    sky_nside=8,
    beam_nside=8,
    hexnum=2,
    sky_eta_max=40e-9,
    beam_eta_max=20e-9,
    fmin_hz=100e6,
    fmax_hz=200e6,
    method="2d",
    seed=42,
):
    """Build a small masked closed-loop joint-fit case for profiling.

    The mask follows the notebook-style horizon beam mask and corresponding
    sky mask. The returned residual is weighted and ready for adjoint updates.

    Returns
    -------
    cal : Calibrator
    state : JointSkyBeamDirtyFitState
    resid : jnp.ndarray
    """
    freqs = np.linspace(fmin_hz, fmax_hz, nfreq, endpoint=False, dtype=np.float32)
    array = HERAArray.from_hex(hexnum=hexnum)
    sky_basis = SkyBasis.from_dpss(freqs, eta_max=sky_eta_max)
    beam_basis = BeamBasis.from_dpss(freqs, eta_max=beam_eta_max)
    sky_model = SkyModel(nside=sky_nside, freqs=freqs, basis=sky_basis)
    beam_model = BeamModel(nside=beam_nside, freqs=freqs, basis=beam_basis)

    t0 = Time("2018-08-31T04:02:30", format="isot", scale="utc")
    times = Time(np.linspace(t0.jd, t0.jd + 1 / 24, ntime), format="jd")
    rot_matrices = compute_rotation_matrices(times, HERA_LOC)

    true_sky = jnp.array(
        sky_basis.project(
            np.ones((sky_model.npix, nfreq), dtype=np.float32) * 0.1
        ),
        dtype=jnp.float32,
    )
    zero_data = np.zeros((ntime, nfreq, array.nbls), dtype=np.complex64)
    temp_cal = Calibrator(
        array, beam_model, sky_model, freqs, rot_matrices,
        data=zero_data, method=method,
    )
    true_vis = temp_cal.fwd.simulate_2d(true_sky, jnp.array(rot_matrices))

    t_norm = np.linspace(0, 1, ntime)[:, None]
    f_norm = np.linspace(0, 1, nfreq, endpoint=False)[None, :]
    data = apply_gains(
        true_vis,
        jnp.array(0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)),
        jnp.array(0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)),
        jnp.zeros((ntime, 2, nfreq), dtype=jnp.float32),
        jnp.array(array.bls, dtype=jnp.float32),
    )

    cal = Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=np.asarray(data),
        method=method,
    )
    beam_mask = cal.build_beam_mask_altitude(0)
    sky_mask = cal.build_sky_mask_from_beam_pixels(beam_mask)
    cal.apply_sky_mask(sky_mask)
    cal.apply_beam_mask(beam_mask)

    rng = np.random.default_rng(seed)
    params = {
        "sky_coeffs": jnp.array(
            rng.normal(0, 0.1, (sky_model.npix, sky_model.nmodes)),
            dtype=jnp.float32,
        ),
        "beam_coeffs": cal.fwd._beam_coeffs_full,
        **init_gain_params(ntime, nfreq),
    }
    state = cal.init_joint_sky_beam_dirty_state(
        params,
        joint_anderson_history=2,
        solve_every={"gains": 0, "rfi": 0},
    )
    resid, _ = cal._jit_residual_and_loss_variable_beam(
        state.params, cal._effective_weights()
    )
    block_until_ready(resid)
    return cal, state, resid


def setup_benchmark_calibrator(
    hexnum=5,
    nfreq=64,
    ntime=24,
    sky_nside=32,
    beam_nside=16,
    fmin_mhz=50.0,
    fmax_mhz=225.0,
    ref_freq=150e6,
    beam_file=None,
    sky_file=None,
    method='2d',
    eps=1e-5,
    sep=14.6,
    perturb_frac=0.30,
    sky_seed=42,
    sky_perturb_seed=99,
    beam_perturb_seed=77,
    apply_masks='standard',
):
    """Build a Calibrator and perturbed initial params for benchmarking.

    Parameters
    ----------
    apply_masks : str or None
        'standard' — horizon cut + beam-weighted sky mask (1/100 threshold) +
                     ever-illuminated beam mask.
        'alt'      — altitude + beam-weight combined sky mask + altitude beam mask
                     (matches anderson_acceleration_test.py notebook setup).
        None       — no masking.

    Returns
    -------
    cal : Calibrator
    prms0 : dict  — perturbed initial params
    info : dict   — setup metadata (freqs, ntime, etc.)
    """
    freqs = np.linspace(fmin_mhz * 1e6, fmax_mhz * 1e6, nfreq, dtype=np.float32)
    rot_matrices, times = make_rot_matrices(ntime=ntime)
    array = HERAArray.from_hex(hexnum=hexnum, sep=sep)

    beam_file = beam_file or str(_BENCH_DIR / 'beam_basis_airy.npz')
    sky_file = sky_file or str(_BENCH_DIR / 'sky_basis_gsm_gleam.npz')
    try:
        beam_basis = BeamBasis.from_file(beam_file, new_freqs=freqs)
        sky_basis = SkyBasis.from_file(sky_file, new_freqs=freqs)
    except FileNotFoundError as e:
        print(f"Error: Could not find basis files.\n  {e}")
        sys.exit(1)

    beam_model = BeamModel(nside=beam_nside, freqs=freqs, basis=beam_basis)
    sky_model = SkyModel(nside=sky_nside, freqs=freqs, basis=sky_basis)

    # Simulate sky
    import healpy
    npix_sky = healpy.nside2npix(sky_nside)
    rng = np.random.default_rng(sky_seed)
    spectral_indices = rng.normal(-0.7, 0.1, npix_sky).astype(np.float32)
    ref_flux = rng.exponential(scale=1.0, size=npix_sky).astype(np.float32)
    flux_true = ref_flux[:, None] * (freqs[None, :] / ref_freq) ** spectral_indices[:, None]
    sky_coeffs_true = jnp.array(sky_basis.project(flux_true), dtype=jnp.float32)
    beam_coeffs_true = jnp.array(beam_model.coeffs, dtype=jnp.float32)

    # Simulate visibilities with smooth gain perturbations
    fwd = ForwardModel(array, sky_model, beam_model, freqs, eps=eps)
    vis_true = fwd.simulate_2d(sky_coeffs_true, jnp.array(rot_matrices))
    t_norm = np.linspace(0, 1, ntime)[:, None]
    f_norm = np.linspace(0, 1, nfreq)[None, :]
    true_log_amp = (0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)).astype(np.float32)
    true_phase = (0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)).astype(np.float32)
    true_phi = np.zeros((ntime, 2, nfreq), dtype=np.float32)
    vis_data = apply_gains(
        vis_true,
        jnp.array(true_log_amp), jnp.array(true_phase),
        jnp.array(true_phi), jnp.array(array.bls, dtype=jnp.float32),
    )

    cal = Calibrator(
        array=array, sky_model=sky_model, beam_model=beam_model,
        freqs=freqs, rot_matrices=rot_matrices, data=vis_data,
        eps=eps, method=method,
    )

    if apply_masks == 'standard':
        bm_wgts = cal.get_sky_beam_weighting()
        cal.apply_sky_mask(bm_wgts > bm_wgts.max() / 100)
    elif apply_masks == 'alt':
        sky_alt_mask = cal.build_sky_mask_altitude(min_altitude_deg=0.0)
        bm_wgts = cal.get_sky_beam_weighting()
        cal.apply_sky_mask(sky_alt_mask & (bm_wgts > bm_wgts.max() / 100))

    # Perturbed initial params
    rng2 = np.random.default_rng(sky_perturb_seed)
    sky_coeffs_perturbed = sky_coeffs_true + perturb_frac * jnp.array(
        rng2.standard_normal(sky_coeffs_true.shape).astype(np.float32)
    ) * float(jnp.abs(sky_coeffs_true).mean())

    rng3 = np.random.default_rng(beam_perturb_seed)
    beam_coeffs_perturbed = beam_coeffs_true + perturb_frac * jnp.array(
        rng3.standard_normal(beam_coeffs_true.shape).astype(np.float32)
    ) * float(jnp.abs(beam_coeffs_true).mean())

    prms0 = {
        'sky_coeffs': sky_coeffs_perturbed,
        'beam_coeffs': beam_coeffs_perturbed,
        **init_gain_params(ntime, nfreq),
    }

    info = dict(
        freqs=freqs, ntime=ntime, nfreq=nfreq,
        sky_nside=sky_nside, beam_nside=beam_nside,
        hexnum=hexnum, nbls=array.nbls, nants=array.nants,
        sky_nmodes=sky_model.nmodes, beam_nmodes=beam_model.nmodes,
    )
    return cal, prms0, info
