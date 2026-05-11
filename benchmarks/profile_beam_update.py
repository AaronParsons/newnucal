import cProfile
import pstats
from io import StringIO

from bench_utils import block_until_ready, setup_benchmark_calibrator, time_call


def setup_calibrator(*, apply_beam_mask=False):
    """Set up a notebook-sized scenario with the shared benchmark helper."""
    cal, params, info = setup_benchmark_calibrator(
        hexnum=5,
        nfreq=32,
        ntime=24,
        sky_nside=32,
        beam_nside=16,
        use_dpss_basis=True,
        apply_masks="standard",
    )
    if apply_beam_mask:
        beam_mask = cal.build_beam_mask_altitude(min_altitude_deg=0.0)
        cal.apply_beam_mask(beam_mask)
    print(
        f"Array: {info['nants']} antennas, {info['nbls']} baselines; "
        f"sky pixels={cal.fwd.npix_sky}, beam pixels={cal.fwd.npix_beam}"
    )
    return cal, params


def profile_sky_update(cal, params):
    """Profile a single sky dirty update."""
    print("\n" + "="*60)
    print("PROFILING SKY UPDATE (1 iteration)")
    print("="*60)

    sky_coeffs = params['sky_coeffs']
    gain_params = {k: params[k] for k in ['log_amp', 'phase', 'phi']}

    pr = cProfile.Profile()
    pr.enable()
    sky_out, loss = cal.fit_sky_dirty(sky_coeffs, gain_params, n_iter=1, verbose=True)
    block_until_ready(sky_out)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_beam_update(cal, params):
    """Profile a single beam dirty update."""
    print("\n" + "="*60)
    print("PROFILING BEAM UPDATE (1 iteration)")
    print("="*60)

    pr = cProfile.Profile()
    pr.enable()
    params_out, loss = cal.fit_beam_dirty(params, n_iter=1, verbose=True)
    block_until_ready(params_out)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    cal, params = setup_calibrator(apply_beam_mask=False)

    # Warm up JIT
    print("\nWarming up JIT...")
    time_call("calc_loss warmup", lambda: cal.calc_loss(params), repeats=1)

    # Profile sky and beam updates WITHOUT beam masking
    print("\n" + "="*60)
    print("WITHOUT BEAM MASKING")
    print("="*60)
    profile_sky_update(cal, params)
    profile_beam_update(cal, params)

    # Now apply beam masking and profile again
    print("\n" + "="*60)
    print("WITH BEAM MASKING (ever-illuminated)")
    print("="*60)
    cal2, params2 = setup_calibrator(apply_beam_mask=True)
    print(f"\nBeam reduced from {cal.fwd.npix_beam} to {cal2.fwd.npix_beam} pixels")

    # Warm up JIT for masked version
    time_call("masked calc_loss warmup", lambda: cal2.calc_loss(params2), repeats=1)

    profile_sky_update(cal2, params2)
    profile_beam_update(cal2, params2)
