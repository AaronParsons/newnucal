"""Profile masked 2D joint-adjoint costs without adding CI timing assertions.

This helper builds a small closed-loop case, applies the same horizon-style
sky/beam masks used in ``notebooks/closed_loop_test.ipynb``, then times the
major pieces of the 2D combined sky+beam adjoint.  It is intended for local
regression checks after changing masked scatter or dirty-map code.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from newnucal.array import HERAArray
from newnucal.basis import BeamBasis, SkyBasis
from newnucal.beam import BeamModel
from newnucal.calibrator import Calibrator
from newnucal.gains import apply_gains, init_gain_params
from newnucal.simulate import compute_rotation_matrices
from newnucal.sky import SkyModel


HERA_LOC = EarthLocation.from_geodetic(
    lat=-30.72152612068926,
    lon=21.42830382686301,
    height=1051.69,
)


def _block(x):
    return jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )


def _time_call(label, fn, repeats=5):
    _block(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block(fn())
        samples.append(time.perf_counter() - t0)
    mean = float(np.mean(samples))
    print(f"{label:28s} mean={mean:.5f}s min={min(samples):.5f}s max={max(samples):.5f}s")
    return mean


def _make_masked_case():
    nfreq = 16
    ntimes = 2
    freqs = np.linspace(100e6, 200e6, nfreq, endpoint=False, dtype=np.float32)
    array = HERAArray.from_hex(hexnum=2)
    sky_basis = SkyBasis.from_dpss(freqs, eta_max=40e-9)
    beam_basis = BeamBasis.from_dpss(freqs, eta_max=20e-9)
    sky_model = SkyModel(nside=8, freqs=freqs, basis=sky_basis)
    beam_model = BeamModel(nside=8, freqs=freqs, basis=beam_basis)

    t0 = Time("2018-08-31T04:02:30", format="isot", scale="utc")
    times = Time(np.linspace(t0.jd, t0.jd + 1 / 24, ntimes), format="jd")
    rot_matrices = compute_rotation_matrices(times, HERA_LOC)

    true_sky = jnp.array(
        sky_basis.project(
            np.ones((sky_model.npix, nfreq), dtype=np.float32) * 0.1
        ),
        dtype=jnp.float32,
    )
    true_vis = Calibrator(
        array, beam_model, sky_model, freqs, rot_matrices,
        data=np.zeros((ntimes, nfreq, array.nbls), dtype=np.complex64),
        method="2d",
    ).fwd.simulate_2d(true_sky, jnp.array(rot_matrices))

    t_norm = np.linspace(0, 1, ntimes)[:, None]
    f_norm = np.linspace(0, 1, nfreq, endpoint=False)[None, :]
    data = apply_gains(
        true_vis,
        jnp.array(0.05 * np.cos(2 * np.pi * f_norm) + 0.02 * np.sin(2 * np.pi * t_norm)),
        jnp.array(0.15 * np.sin(2 * np.pi * f_norm) + 0.05 * np.cos(2 * np.pi * t_norm)),
        jnp.zeros((ntimes, 2, nfreq), dtype=jnp.float32),
        jnp.array(array.bls, dtype=jnp.float32),
    )

    cal = Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
        freqs=freqs,
        rot_matrices=rot_matrices,
        data=np.asarray(data),
        method="2d",
    )
    beam_mask = cal.build_beam_mask_altitude(0)
    sky_mask = cal.build_sky_mask_from_beam_pixels(beam_mask)
    cal.apply_sky_mask(sky_mask)
    cal.apply_beam_mask(beam_mask)

    rng = np.random.default_rng(42)
    params = {
        "sky_coeffs": jnp.array(
            rng.normal(0, 0.1, (sky_model.npix, sky_model.nmodes)),
            dtype=jnp.float32,
        ),
        "beam_coeffs": cal.fwd._beam_coeffs_full,
        **init_gain_params(ntimes, nfreq),
    }
    state = cal.init_joint_sky_beam_dirty_state(
        params,
        joint_anderson_history=2,
        solve_every={"gains": 0, "rfi": 0},
    )
    weights = cal._effective_weights()
    resid, _ = cal._jit_residual_and_loss_variable_beam(state.params, weights)
    _block(resid)
    return cal, state, resid


def main():
    cal, state, resid = _make_masked_case()
    fwd = cal.fwd
    sky_coeffs = state.params["sky_coeffs"]
    beam_reg = state.settings["joint_beam_reg"]
    sky_reg = state.settings["joint_sky_reg"]

    print(
        f"masked sky={fwd.npix_sky}/{fwd.npix_sky_full} "
        f"beam={fwd.npix_beam}/{len(fwd._beam_mask)} "
        f"ntime={cal.ntime} nfreq={cal.nfreq} nbls={cal.nbls}"
    )

    def dirty_all():
        return jnp.stack([
            fwd.dirty_apparent_sky_one_time_2d(resid[tind], tind)
            for tind in range(cal.ntime)
        ])

    dirty_pf_all = _block(dirty_all())

    sky_spec = jnp.array(sky_coeffs) @ fwd.A_sky.T
    sky_weight = jnp.abs(sky_spec) ** 2
    sky_pow = sky_weight.sum(axis=1)
    a_sky = jnp.array(fwd.A_sky, dtype=sky_spec.real.dtype)
    a_beam = jnp.array(fwd.A_beam, dtype=sky_spec.real.dtype)
    beam_spec_h_all = jnp.stack(fwd._beam_spec_horizon_all)
    horizon_all = jnp.stack(fwd._horizon_all)
    px_all = jnp.stack(fwd._interp_px_all)
    wgt_all = jnp.stack(fwd._interp_wgt_all)
    lookup_np = getattr(fwd, "_beam_index_lookup", None)
    if lookup_np is None:
        lookup_np = np.full(fwd._beam_mask.shape, -1, dtype=np.int32)
        lookup_np[fwd._beam_indices] = np.arange(len(fwd._beam_indices), dtype=np.int32)
    lookup = jnp.array(lookup_np)

    @jax.jit
    def sky_accum_from_dirty(dirty):
        weight = jnp.abs(beam_spec_h_all) ** 2
        delta_app = jnp.conj(beam_spec_h_all.astype(dirty.dtype)) * dirty / (
            weight.astype(dirty.dtype) + beam_reg
        )
        sky_num = (weight.astype(dirty.dtype) * delta_app).sum(axis=0)
        sky_den = weight.sum(axis=0)
        delta_eq_pf = sky_num / (sky_den.astype(dirty.dtype) + fwd.eps)
        delta_eq_pf = delta_eq_pf / cal.ntime
        return (jnp.real(delta_eq_pf) @ a_sky).astype(jnp.float32)

    @jax.jit
    def beam_project_from_dirty(dirty):
        delta_bsf = jnp.conj(sky_spec)[None, :, :] * dirty / (sky_weight[None, :, :] + sky_reg)
        return (jnp.real(delta_bsf) * horizon_all[:, :, None]) @ a_beam

    @jax.jit
    def masked_scatter_from_delta(delta_bi_all):
        beam_num = jnp.zeros((fwd.npix_beam, fwd.nmodes_beam), dtype=jnp.float32)
        beam_den = jnp.zeros(fwd.npix_beam, dtype=jnp.float32)

        def one_time(carry, tind):
            b_num, b_den = carry
            px = px_all[tind]
            wgt = wgt_all[tind]
            delta_bi = delta_bi_all[tind]
            px_flat = px.reshape(-1)
            px_masked_flat = lookup[px_flat]
            valid_flat = px_masked_flat >= 0
            px_masked_flat = jnp.where(valid_flat, px_masked_flat, 0)
            w_flat = (wgt * sky_pow[None, :]).reshape(-1)
            w_flat = jnp.where(valid_flat, w_flat, 0.0)
            delta_flat = jnp.repeat(
                delta_bi[None, :, :], 4, axis=0
            ).reshape(-1, fwd.nmodes_beam)
            b_num = b_num.at[px_masked_flat].add(w_flat[:, None] * delta_flat)
            b_den = b_den.at[px_masked_flat].add(w_flat)
            return (b_num, b_den), None

        (beam_num, beam_den), _ = jax.lax.scan(
            one_time, (beam_num, beam_den), jnp.arange(cal.ntime)
        )
        return beam_num / (beam_den[:, None] + fwd.eps) / cal.ntime

    delta_bi_all = _block(beam_project_from_dirty(dirty_pf_all))

    @jax.jit
    def jitted_sky_adjoint(residual_vis):
        return fwd.accumulate_equatorial_sky_update_2d(
            residual_vis, step_size=1.0, beam_reg=beam_reg
        )

    @jax.jit
    def jitted_beam_adjoint(sky, residual_vis):
        return fwd.accumulate_beam_update_2d(
            sky, residual_vis, step_size=1.0, sky_reg=sky_reg
        )

    @jax.jit
    def jitted_combined_adjoint(sky, residual_vis):
        return fwd.accumulate_sky_and_beam_update_2d(
            sky, residual_vis, sky_step_size=1.0, beam_step_size=1.0,
            beam_reg=beam_reg, sky_reg=sky_reg,
        )

    print("\nend-to-end adjoints")
    t_dirty = _time_call("dirty apparent sky", dirty_all)
    t_sky = _time_call(
        "sky-only adjoint",
        lambda: fwd.accumulate_equatorial_sky_update_2d(resid, step_size=1.0, beam_reg=beam_reg),
    )
    t_beam = _time_call(
        "beam-only adjoint",
        lambda: fwd.accumulate_beam_update_2d(sky_coeffs, resid, step_size=1.0, sky_reg=sky_reg),
    )
    t_combined = _time_call(
        "combined adjoint",
        lambda: fwd.accumulate_sky_and_beam_update_2d(
            sky_coeffs, resid, sky_step_size=1.0, beam_step_size=1.0,
            beam_reg=beam_reg, sky_reg=sky_reg,
        ),
    )

    print("\njit-wrapped end-to-end adjoints")
    t_jit_sky = _time_call("jitted sky-only adjoint", lambda: jitted_sky_adjoint(resid))
    t_jit_beam = _time_call("jitted beam-only adjoint", lambda: jitted_beam_adjoint(sky_coeffs, resid))
    t_jit_combined = _time_call("jitted combined adjoint", lambda: jitted_combined_adjoint(sky_coeffs, resid))

    print("\nsubcomponents from precomputed dirty sky")
    t_sky_acc = _time_call("sky accum/projection", lambda: sky_accum_from_dirty(dirty_pf_all))
    t_beam_proj = _time_call("beam delta projection", lambda: beam_project_from_dirty(dirty_pf_all))
    t_scatter = _time_call("masked beam scatter", lambda: masked_scatter_from_delta(delta_bi_all))

    accounted = t_dirty + t_sky_acc + t_beam_proj + t_scatter
    print("\nsummary")
    print(f"combined adjoint mean       {t_combined:.5f}s")
    print(f"dirty+sky+beam+scatter mean {accounted:.5f}s")
    print(f"beam-only / combined        {t_beam / t_combined:.2f}")
    print(f"dirty apparent / combined   {t_dirty / t_combined:.2f}")
    print(f"masked scatter / combined   {t_scatter / t_combined:.2f}")
    print(f"jitted combined / combined  {t_jit_combined / t_combined:.2f}")
    print(f"jitted beam / jitted comb.   {t_jit_beam / t_jit_combined:.2f}")
    print(f"jitted sky / jitted comb.    {t_jit_sky / t_jit_combined:.2f}")


if __name__ == "__main__":
    main()
