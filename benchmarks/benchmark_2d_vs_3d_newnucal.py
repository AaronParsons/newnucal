"""
Benchmark the 2D hex-rect NUFFT path vs the 3D type-3 NUFFT path as
implemented in newnucal.

Usage
-----
python benchmarks/benchmark_2d_vs_3d_newnucal.py
python benchmarks/benchmark_2d_vs_3d_newnucal.py --hexnum 5 --nside 32 --nfreq 128 --ntime 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

import newnucal
from newnucal import HERAArray, BeamModel, SkyModel, ForwardModel
from newnucal.dpss import dpss_matrix, dpss_project
from newnucal.hexrect import critical_channel_spacing


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_freqs(nfreq, fmin_mhz=100.0, fmax_mhz=200.0):
    return np.linspace(fmin_mhz * 1e6, fmax_mhz * 1e6, nfreq, dtype=np.float32)


def _build_rot_matrices(ntime):
    """Random rotation matrices (not physically motivated — just for timing)."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, (ntime, 3))
    mats = []
    for a in angles:
        r = Rotation.from_euler('xyz', a)
        mats.append(r.as_matrix())
    return jnp.array(np.stack(mats, axis=0).astype(np.float32))


def _build_forward_model(hexnum, nside, freqs, eta_max_sky=40e-9, eta_max_beam=20e-9):
    from newnucal.basis import SkyBasis, BeamBasis
    array    = HERAArray.from_hex(hexnum=hexnum)
    sky_mdl  = SkyModel(nside=nside, freqs=freqs, basis=SkyBasis.from_dpss(freqs, eta_max_sky))
    beam_mdl = BeamModel(nside=nside, freqs=freqs, basis=BeamBasis.from_dpss(freqs, eta_max_beam))
    fwd      = ForwardModel(array, sky_mdl, beam_mdl, freqs)
    return fwd


def _warm_up(fwd, sky_coeffs, rot_matrices, method):
    """JIT-warm-up: run once and discard."""
    if method == '2d':
        _ = fwd.simulate_2d(sky_coeffs, rot_matrices).block_until_ready()
    else:
        _ = fwd.simulate(sky_coeffs, rot_matrices).block_until_ready()


def _time_fn(fn, n_repeat=5):
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn().block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.array(times)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_benchmark(hexnum=3, nside=16, nfreq=64, ntime=5, n_repeat=5):
    freqs = _build_freqs(nfreq)
    rot_matrices = _build_rot_matrices(ntime)

    print(f"\n{'='*60}")
    print(f"  hexnum={hexnum}, nside={nside}, nfreq={nfreq}, ntime={ntime}")

    fwd = _build_forward_model(hexnum, nside, freqs)
    fwd.precompute_time_geometry(rot_matrices)

    print(f"  nbls={fwd.nbls}, npix_sky={fwd.npix_sky}")

    dnu_actual  = float(freqs[1] - freqs[0])
    bmax        = float(np.max(np.linalg.norm(fwd.array.bls[:, :2], axis=1)))
    dnu_crit    = critical_channel_spacing(bmax)
    print(f"  dnu_actual={dnu_actual/1e6:.3f} MHz, dnu_crit={dnu_crit/1e6:.3f} MHz "
          f"({'alias-free for 2D path' if dnu_actual <= dnu_crit else 'exceeds 2D alias-free limit'})")
    print(f"{'='*60}")

    A_sky      = fwd.A_sky
    npix       = fwd.npix_sky
    nmodes     = A_sky.shape[1]
    rng        = np.random.default_rng(1)
    flat_flux  = np.ones((npix, nfreq), dtype=np.float32)
    sky_coeffs = jnp.array(dpss_project(flat_flux, A_sky), dtype=jnp.float32)

    # --- JIT warm-up -------------------------------------------------------
    print("  Warming up JIT ... ", end="", flush=True)
    _warm_up(fwd, sky_coeffs, rot_matrices, '3d')
    _warm_up(fwd, sky_coeffs, rot_matrices, '2d')
    print("done")

    # --- Time 3D -----------------------------------------------------------
    times_3d = _time_fn(
        lambda: fwd.simulate(sky_coeffs, rot_matrices),
        n_repeat=n_repeat,
    )
    mean_3d, std_3d = times_3d.mean(), times_3d.std()
    print(f"  3D path: {mean_3d*1e3:.1f} ± {std_3d*1e3:.1f} ms  (n={n_repeat})")

    # --- Time 2D -----------------------------------------------------------
    times_2d = _time_fn(
        lambda: fwd.simulate_2d(sky_coeffs, rot_matrices),
        n_repeat=n_repeat,
    )
    mean_2d, std_2d = times_2d.mean(), times_2d.std()
    print(f"  2D path: {mean_2d*1e3:.1f} ± {std_2d*1e3:.1f} ms  (n={n_repeat})")

    speedup = mean_3d / mean_2d
    print(f"  Speedup (3D/2D): {speedup:.1f}×")

    # NOTE on accuracy: the 3D path delay-truncates sky×beam to eta_max before
    # the NUFFT; the 2D path operates in frequency domain directly.  They
    # compute the same integral only when sky×beam is bandlimited within the
    # delay window.  Accuracy is validated in tests/test_simulate.py at hexnum=2
    # where this condition is satisfied.

    return dict(
        hexnum=hexnum, nside=nside, nfreq=nfreq, ntime=ntime,
        nbls=fwd.nbls, npix=npix,
        mean_3d_ms=mean_3d * 1e3, mean_2d_ms=mean_2d * 1e3,
        speedup=speedup,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark 2D vs 3D NUFFT in newnucal")
    parser.add_argument("--hexnum", type=int, default=None)
    parser.add_argument("--nside",  type=int, default=None)
    parser.add_argument("--nfreq",  type=int, default=None)
    parser.add_argument("--ntime",  type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if any(v is not None for v in [args.hexnum, args.nside, args.nfreq, args.ntime]):
        # Single custom run
        run_benchmark(
            hexnum  = args.hexnum  or 3,
            nside   = args.nside   or 16,
            nfreq   = args.nfreq   or 64,
            ntime   = args.ntime   or 5,
            n_repeat= args.repeats,
        )
    else:
        # Sweep over representative configurations
        configs = [
            dict(hexnum=2, nside=8,  nfreq=32,  ntime=3),
            dict(hexnum=3, nside=16, nfreq=64,  ntime=5),
            dict(hexnum=5, nside=32, nfreq=128, ntime=10),
        ]
        results = []
        for cfg in configs:
            r = run_benchmark(**cfg, n_repeat=args.repeats)
            results.append(r)

        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        hdr = f"{'hexnum':>6} {'nfreq':>6} {'nbls':>6} {'3D ms':>8} {'2D ms':>8} {'speedup':>8}"
        print(hdr)
        for r in results:
            print(f"{r['hexnum']:>6} {r['nfreq']:>6} {r['nbls']:>6} "
                  f"{r['mean_3d_ms']:>8.1f} {r['mean_2d_ms']:>8.1f} {r['speedup']:>8.1f}×")


if __name__ == "__main__":
    main()
