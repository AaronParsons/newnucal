"""Per-block timing of newnucal solver components on real HERA data.

Mirrors the setup of ``specred/newnucal_zen_LST_mini.ipynb`` (via
``newnucal.realdata``) and times the building blocks that dominate fit
wall-clock: forward simulate, loss+grad, one joint dirty step, the
closed-form gain solve, and one sky-dirty iteration.

Usage:
    python benchmark_real_data_blocks.py [--ntimes 4] [--nchan 128]
        [--ch0 600] [--pol 0] [--data PATH] [--lbfgs]
"""

import argparse
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from newnucal import BeamModel, BeamBasis, SkyModel, SkyBasis, Calibrator
from newnucal.simulate import compute_rotation_matrices
from newnucal.realdata import (
    load_lst_stack_subset, snap_to_hex_lattice, hera_array_from_file_pairs,
    dead_channel_log_weights, estimate_sigma_freqdiff,
)

DEFAULT_DATA = ('/home/aparsons/projects/hera/analysis/specred/'
                'zen.LST.mini.131_nights.FR0filt.uvh5')


def timeit(label, fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.time()
        jax.block_until_ready(fn())
        ts.append(time.time() - t0)
    print(f'{label:38s} {min(ts):8.2f}s (best of {n}, first {ts[0]:.2f}s)')
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--ntimes', type=int, default=4)
    ap.add_argument('--nchan', type=int, default=128)
    ap.add_argument('--ch0', type=int, default=600)
    ap.add_argument('--pol', type=int, default=0)
    ap.add_argument('--nside-sky', type=int, default=32)
    ap.add_argument('--nside-beam', type=int, default=16)
    ap.add_argument('--method', default='2d', choices=['2d', '3d'])
    ap.add_argument('--lbfgs', action='store_true',
                    help='also time one L-BFGS outer (slow)')
    args = ap.parse_args()

    from astropy.time import Time
    print(f'loading {args.ntimes} times x {args.nchan} chans from {args.data}')
    sub = load_lst_stack_subset(
        args.data,
        np.unique(np.round(np.linspace(0, 139, args.ntimes)).astype(int)),
        args.ch0, args.nchan, pol_index=args.pol)
    snapped, info = snap_to_hex_lattice(sub['antpos'])
    print(f"lattice snap max err {info['max_err_m']*1e3:.1f} mm")
    array = hera_array_from_file_pairs(snapped, sub['pairs'])
    rot = compute_rotation_matrices(Time(sub['jd'], format='jd'), sub['location'])

    freqs = sub['freqs']
    vis_w = ((~sub['flags']) & (sub['nsamples'] > 0) & (sub['data'] != 0)).astype(float)
    beam_model = BeamModel(args.nside_beam, freqs,
                           BeamBasis.from_beam_diameters(freqs, np.linspace(12, 18, 13),
                                                         nside=args.nside_beam, n_modes=8))
    sky_model = SkyModel(args.nside_sky, freqs, SkyBasis.from_dpss(freqs, 250e-9))

    cal = Calibrator(array, beam_model, sky_model, freqs, rot,
                     jnp.array(np.conj(sub['data'])),
                     method=args.method,
                     noise_sigma=estimate_sigma_freqdiff(sub['data'], sub['flags']),
                     channel_weights=dead_channel_log_weights(vis_w),
                     t_chunk_size=args.ntimes,
                     project_out_time_mean=True)
    beam_mask = cal.build_beam_mask_altitude(0.0)
    cal.apply_sky_mask(cal.build_sky_mask_from_beam_pixels(beam_mask))
    cal.apply_beam_mask(beam_mask)
    cal.set_visibility_weights(vis_w)

    prms = cal.init_params()
    rng = np.random.default_rng(0)
    prms['sky_coeffs'] = jnp.array(
        rng.exponential(1.0, prms['sky_coeffs'].shape), dtype=prms['sky_coeffs'].dtype)

    print(f'\n--- {args.method} path, {args.ntimes}t x {args.nchan}f x {array.nbls}b, '
          f'{cal._get_active_size()} active sky pix ---')
    timeit('forward simulate (cached beam)', lambda: cal.simulate(
        {k: prms[k] for k in ('sky_coeffs', 'log_amp', 'phase', 'phi')}))
    timeit('forward simulate (variable beam)', lambda: cal.simulate(prms))
    timeit('loss + grad', lambda: cal._jit_val_grad(
        {k: prms[k] for k in ('sky_coeffs', 'log_amp', 'phase', 'phi')},
        cal._effective_weights())[0])
    timeit('gain closed-form solve', lambda: cal.fit_gains_linear_variable_beam(
        prms['sky_coeffs'], prms['beam_coeffs'])[1])
    timeit('sky dirty (1 iter)', lambda: cal.fit_sky_dirty(
        prms['sky_coeffs'], {k: prms[k] for k in ('log_amp', 'phase', 'phi')},
        n_iter=1)[1])
    st = cal.init_joint_sky_beam_dirty_state(prms, solve_every={'gains': 0, 'rfi': 0})
    timeit('joint dirty step', lambda: cal.run_joint_sky_beam_dirty_state(st, 1).loss)
    if args.lbfgs:
        timeit('lbfgs outer (maxiter=30)', lambda: cal.fit_joint_sky_beam_lbfgs(
            prms, n_outer=1, lbfgs_maxiter=30, verbose=False)[1], n=1)


if __name__ == '__main__':
    main()
