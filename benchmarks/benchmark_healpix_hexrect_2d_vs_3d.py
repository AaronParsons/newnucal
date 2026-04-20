
"""
Benchmark HEALPix-sky pipelines for a hexagonal HERA array:

(1) 3D path:
    spectral-basis -> eta per sky pixel
    then one 3D type-3 NUFFT from nonuniform (sky, eta) sources
    to nonuniform (baseline, frequency) targets

(2) 2D path:
    spectral-basis -> frequency per sky pixel
    then one 2D type-1 NUFFT per frequency from nonuniform transformed HEALPix
    sky coordinates to a uniform axial uv grid
    then sample only the occupied unique redundant baseline cells

This script reflects:
- HEALPix sky support in all cases
- axial-coordinate rectangularization of the HERA hexagonal baseline lattice
- one representative per redundant baseline
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import healpy as hp
import finufft

C = 299_792_458.0


def hex_array_positions(hex_num=5, spacing=14.6):
    positions = []
    for q in range(-(hex_num - 1), hex_num):
        rmin = max(-(hex_num - 1), -q - (hex_num - 1))
        rmax = min(hex_num - 1, -q + (hex_num - 1))
        for r in range(rmin, rmax + 1):
            x = spacing * (q + 0.5 * r)
            y = spacing * (np.sqrt(3.0) / 2.0) * r
            positions.append((x, y, 0.0))
    return np.asarray(positions, dtype=np.float32)


def unique_redundant_baselines_and_axial(antpos, spacing=14.6, round_decimals=8):
    bls = []
    for i in range(len(antpos)):
        for j in range(i + 1, len(antpos)):
            bls.append(antpos[j] - antpos[i])
    bls = np.asarray(bls, dtype=np.float32)

    A = np.array([
        [spacing, 0.5 * spacing],
        [0.0, spacing * np.sqrt(3.0) / 2.0],
    ], dtype=np.float32)
    Ainv = np.linalg.inv(A)

    axial = (Ainv @ bls[:, :2].T).T
    axial_round = np.round(axial, decimals=round_decimals).astype(np.int64)

    _, idx = np.unique(axial_round, axis=0, return_index=True)
    idx = np.sort(idx)
    return bls[idx], axial_round[idx], A


def critical_channel_spacing(bmax_m, field_radius=1.0, safety=1.0):
    return C / (2.0 * safety * bmax_m * field_radius)


def make_polynomial_basis(freqs_hz, nmodes):
    x = (freqs_hz / np.mean(freqs_hz)) - 1.0
    raw = [np.ones_like(x)]
    for p in range(1, nmodes):
        raw.append(x ** p)
    raw = np.stack(raw, axis=0)
    Q, _ = np.linalg.qr(raw.T)
    return Q.T.astype(np.complex128)


def make_healpix_topocentric_points(nside):
    npix = hp.nside2npix(nside)
    vec = np.array(hp.pix2vec(nside, np.arange(npix)), dtype=np.float32)  # (3, npix)
    mask = vec[2] > 0.0
    vec = vec[:, mask]
    ell = vec[0]
    emm = vec[1]
    return ell, emm, mask


def make_random_coeffs(npix, nmodes, seed=0):
    rng = np.random.default_rng(seed)
    coeffs = (rng.standard_normal((npix, nmodes)) + 1j * rng.standard_normal((npix, nmodes))).astype(np.complex128)
    return coeffs


def time_call(fn, repeat=3):
    out = None
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hex-num", type=int, default=5)
    ap.add_argument("--spacing-m", type=float, default=14.6)
    ap.add_argument("--nu-min-mhz", type=float, default=50.0)
    ap.add_argument("--nu-max-mhz", type=float, default=225.0)
    ap.add_argument("--field-radius", type=float, default=1.0)
    ap.add_argument("--safety", type=float, default=1.0)
    ap.add_argument("--nside", type=int, default=16)
    ap.add_argument("--nmodes", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--eta-max-ns", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    antpos = hex_array_positions(args.hex_num, args.spacing_m)
    bls, axial, A = unique_redundant_baselines_and_axial(antpos, spacing=args.spacing_m)
    bmax = float(np.max(np.sqrt(np.sum(bls[:, :2] ** 2, axis=1))))
    dnu_crit = critical_channel_spacing(bmax, args.field_radius, args.safety)

    bandwidth_hz = (args.nu_max_mhz - args.nu_min_mhz) * 1e6
    nfreq = int(np.ceil(bandwidth_hz / dnu_crit)) + 1
    freqs_hz = np.linspace(args.nu_min_mhz * 1e6, args.nu_max_mhz * 1e6, nfreq)

    basis_nu = make_polynomial_basis(freqs_hz, args.nmodes).astype(np.complex64)
    eta_grid_full = np.fft.fftfreq(nfreq, d=float(freqs_hz[1] - freqs_hz[0]))  # seconds
    eta_mask = np.abs(eta_grid_full) <= (args.eta_max_ns * 1e-9)
    eta_grid = eta_grid_full[eta_mask]
    basis_eta = np.fft.fft(basis_nu, axis=1, norm="ortho")[:, eta_mask].astype(np.complex64)

    # HEALPix sky coordinates in topocentric (ell,m), above horizon only
    ell, emm, mask = make_healpix_topocentric_points(args.nside)
    npix = len(ell)

    # Map sky coordinates to dual axial coordinates xi = A^T s
    S = np.stack([ell, emm], axis=0)      # (2, npix)
    Xi = (A.T @ S).T                      # (npix, 2)
    xi1 = Xi[:, 0]
    xi2 = Xi[:, 1]

    coeffs = make_random_coeffs(npix, args.nmodes, seed=args.seed).astype(np.complex64)

    # Uniform axial uv grid bounding all occupied unique redundant baselines
    q = axial[:, 0]
    r = axial[:, 1]
    qmin, qmax = int(np.min(q)), int(np.max(q))
    rmin, rmax = int(np.min(r)), int(np.max(r))
    n_q = qmax - qmin + 1
    n_r = rmax - rmin + 1
    q_idx = q - qmin
    r_idx = r - rmin

    # --- 2D path: nonuniform sky -> uniform axial uv plane, per frequency ----
    def basis_to_freq():
        return np.tensordot(coeffs, basis_nu, axes=(1, 0)).astype(np.complex128)  # (npix, nfreq)

    t_freq, spec_freq = time_call(basis_to_freq, repeat=args.repeat)

    def run_2d_only():
        vis_grid = np.empty((nfreq, n_q, n_r), dtype=np.complex64)
        for fi, nu in enumerate(freqs_hz):
            xj = np.ascontiguousarray(-2.0 * np.pi * (nu / C) * xi1, dtype=np.float32)
            yj = np.ascontiguousarray(-2.0 * np.pi * (nu / C) * xi2, dtype=np.float32)
            cj = np.ascontiguousarray(spec_freq[:, fi], dtype=np.complex64)
            out = finufft.nufft2d1(xj, yj, cj, (n_q, n_r), isign=+1, eps=1e-6)
            vis_grid[fi] = np.asarray(out)
        return vis_grid[:, q_idx, r_idx]

    t_2d_only, _ = time_call(run_2d_only, repeat=args.repeat)
    t_2d_total = t_freq + t_2d_only

    # --- 3D path: standard coordinates (ell, m, eta) -> nonuniform baseline-frequency samples
    def basis_to_eta():
        return np.tensordot(coeffs, basis_eta, axes=(1, 0)).astype(np.complex128)  # (npix, nfreq_eta)

    t_eta, spec_eta = time_call(basis_to_eta, repeat=args.repeat)

    # 3D type-3 source points in the standard topocentric coordinates used by simulate.py:
    # source = (2π ell, 2π m, -2π eta), target = (2π ν Bx/c, 2π ν By/c, 2π ν)
    ndelay_eff = len(eta_grid)
    sx = np.repeat(2.0 * np.pi * ell, ndelay_eff).astype(np.float32)
    sy = np.repeat(2.0 * np.pi * emm, ndelay_eff).astype(np.float32)
    sz = np.tile(-2.0 * np.pi * eta_grid, npix).astype(np.float32)

    # Match the scaling used in simulate.py / jax_finufft.nufft3:
    # source sky/eta coords carry the 2π factors, target baseline/frequency coords do not.
    tx = np.repeat((freqs_hz[:, None] / C) * bls[None, :, 0], 1, axis=0).reshape(-1).astype(np.float32)
    ty = np.repeat((freqs_hz[:, None] / C) * bls[None, :, 1], 1, axis=0).reshape(-1).astype(np.float32)
    tz = np.repeat(freqs_hz, len(bls)).astype(np.float32)

    def run_3d_only():
        cj = np.ascontiguousarray(spec_eta.reshape(-1), dtype=np.complex64)
        out = finufft.nufft3d3(sx, sy, sz, cj, tx, ty, tz, isign=+1, eps=1e-6)
        return np.asarray(out).reshape(nfreq, len(bls))

    t_3d_only, _ = time_call(run_3d_only, repeat=args.repeat)
    t_3d_total = t_eta + t_3d_only

    print("=== HEALPix sky benchmark: 2D hex-rect uv vs 3D standard-coordinate eta NUFFT ===")
    print(f"hex_num                 : {args.hex_num}")
    print(f"nant                    : {len(antpos)}")
    print(f"unique redundant bls    : {len(bls)}")
    print(f"axial uv grid           : {n_q} x {n_r}")
    print(f"HEALPix nside           : {args.nside}")
    print(f"HEALPix above horizon   : {npix}")
    print(f"longest baseline [m]    : {bmax:.3f}")
    print(f"critical dnu [Hz]       : {dnu_crit:.3f}")
    print(f"nfreq                   : {nfreq}")
    print(f"nmodes                  : {args.nmodes}")
    print(f"eta_max [ns]            : {args.eta_max_ns}")
    print(f"ndelay_eff              : {len(eta_grid)}")
    print()
    print("timings [seconds]")
    print(f"  basis -> eta          : {t_eta:.6f}")
    print(f"  3D NUFFT only         : {t_3d_only:.6f}")
    print(f"  3D total              : {t_3d_total:.6f}")
    print(f"  basis -> freq         : {t_freq:.6f}")
    print(f"  2D type1 only         : {t_2d_only:.6f}")
    print(f"  2D total              : {t_2d_total:.6f}")
    print()
    print("speed ratios")
    print(f"  2D_only / 3D_only     : {t_2d_only / max(t_3d_only, 1e-12):.3f}")
    print(f"  2D_total / 3D_total   : {t_2d_total / max(t_3d_total, 1e-12):.3f}")
    print()
    print("notes")
    print("  - 2D path: HEALPix sky pixels mapped to dual axial coordinates,")
    print("    then a type-1 NUFFT per frequency to a uniform rectangular uv grid.")
    print("  - 3D path: HEALPix sky pixels stay in standard topocentric (ell,m) coordinates,")
    print("    then one type-3 NUFFT maps (ell,m,eta) to nonuniform physical baseline-frequency targets.")
    print("  - This is a computational pipeline comparison, not a value-equality test.")


if __name__ == "__main__":
    main()
