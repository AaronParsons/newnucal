"""Tests for newnucal.identifiability (uv–frequency identifiability tool)."""

import numpy as np
import pytest
import hera_sim

from newnucal.identifiability import (
    OMEGA_EARTH,
    cell_snr_from_visibility_snr,
    STATUS_ALIAS,
    STATUS_FR0,
    STATUS_MEASURED,
    ObservationSpec,
    analyze,
    coupling_target_mask,
    disk_aperture_kernel,
    spectral_basis,
    summarize,
    unique_baselines,
)
from newnucal.utils import C


def _hex(hexnum, sep=14.6):
    return hera_sim.antpos.HexArray(sep=sep, split_core=False, outriggers=0)(hexnum)


def _brute_force(bls, w, spec, A, snr, u_e, v_n, kernel=disk_aperture_kernel,
                 prior_var=None):
    """Dense Fisher over every (baseline, freq, fringe bin) measurement."""
    bls = np.vstack([bls, -bls])
    w = np.concatenate([w, w])
    du = spec.fringe_resolution_u
    nE, R, M = u_e.size, v_n.size, A.shape[1]
    UE, VN = np.meshgrid(u_e, v_n, indexing="ij")
    bin_of = np.round(u_e / du).astype(int)
    rows, noise = [], []
    for f, nu in enumerate(spec.freqs):
        for b, wb in zip(bls, w):
            centre = b * nu / C
            K = kernel(np.hypot(UE - centre[0], VN - centre[1]) * C / nu,
                       spec.dish_diameter)
            K = K * spec.time_taper(UE)
            for kb in np.unique(bin_of):
                if spec.bin_status(kb * du) != STATUS_MEASURED:
                    continue
                g = np.where((bin_of == kb)[:, None], K, 0.0)
                if not np.any(g):
                    continue
                rows.append((g[:, :, None] * A[f][None, None, :]).ravel())
                noise.append(wb)
    G = np.array(rows)
    pv = np.ones(M) if prior_var is None else np.asarray(prior_var)
    prior = np.tile(pv, nE * R)
    P = snr ** 2 * (G.T * np.array(noise)) @ G + np.diag(1.0 / prior)
    return (np.diag(np.linalg.inv(P)) / prior).reshape(nE, R, M)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def test_unique_baselines_hex7():
    ants = _hex(2)
    bls, counts = unique_baselines(ants)
    n = len(ants)
    assert counts.sum() == n * (n - 1) // 2
    # Hex-7 has 9 unique non-zero separations (up to sign).
    assert len(bls) == 9
    assert np.all((bls[:, 0] > 0) | ((bls[:, 0] == 0) & (bls[:, 1] > 0)))
    # The 14.6 m E-W group contains 4 pairs in hex-7 (rows of 2, 3, 2).
    ew = np.isclose(bls[:, 0], 14.6) & np.isclose(bls[:, 1], 0.0)
    assert counts[ew][0] == 4


def test_unique_baselines_accepts_array():
    pos = np.array([[0, 0, 0], [14.6, 0, 0], [29.2, 0, 0]])
    bls, counts = unique_baselines(pos)
    np.testing.assert_allclose(bls, [[14.6, 0], [29.2, 0]])
    np.testing.assert_array_equal(counts, [2, 1])


def test_disk_kernel_matches_numerical_overlap():
    D = 14.0
    assert disk_aperture_kernel(0.0, D) == pytest.approx(1.0)
    assert disk_aperture_kernel(D, D) == pytest.approx(0.0, abs=1e-12)
    assert disk_aperture_kernel(1.5 * D, D) == 0.0
    # Overlap area of two disks on a fine grid, normalised by disk area.
    x = np.linspace(-D, 2 * D, 1201)
    X, Y = np.meshgrid(x, x)
    disk = X ** 2 + Y ** 2 <= (D / 2) ** 2
    for s in [0.2, 0.5, 0.8]:
        shifted = (X - s * D) ** 2 + Y ** 2 <= (D / 2) ** 2
        frac = np.sum(disk & shifted) / np.sum(disk)
        assert disk_aperture_kernel(s * D, D) == pytest.approx(frac, abs=5e-3)


@pytest.mark.parametrize("kind", ["dpss", "legendre"])
def test_spectral_basis_orthonormal(kind):
    freqs = np.linspace(140e6, 160e6, 41)
    A = spectral_basis(freqs, 4, kind=kind)
    assert A.shape == (41, 4)
    np.testing.assert_allclose(A.T @ A, np.eye(4), atol=1e-10)
    assert A[:, 0].sum() > 0


def test_spectral_basis_rejects_bad_kind():
    with pytest.raises(ValueError):
        spectral_basis(np.linspace(1, 2, 5), 2, kind="nope")
    with pytest.raises(ValueError):
        spectral_basis(np.linspace(1, 2, 5), 6)


def test_observation_spec_limits():
    spec = ObservationSpec(freqs=[150e6], lst_span_hours=2.0, integration_s=60.0,
                           fr0_halfwidth_hz=1e-4)
    cosd = abs(np.cos(np.deg2rad(spec.latitude_deg)))
    assert spec.fringe_resolution_u == pytest.approx(1 / ((np.pi / 6) * cosd))
    assert spec.u_alias == pytest.approx(1 / (2 * 60.0 * OMEGA_EARTH * cosd))
    assert spec.u_fr0 == pytest.approx(1e-4 / (OMEGA_EARTH * cosd))
    assert spec.bin_status(0.0) == STATUS_FR0
    assert spec.bin_status(10.0) == STATUS_MEASURED
    assert spec.bin_status(1e4) == STATUS_ALIAS
    assert spec.time_taper(0.0) == pytest.approx(1.0)
    # At the Nyquist limit, boxcar averaging gives sinc(1/2) = 2/pi.
    assert spec.time_taper(spec.u_alias) == pytest.approx(2 / np.pi)


# ----------------------------------------------------------------------
# analyze()
# ----------------------------------------------------------------------

@pytest.fixture
def small_case():
    pos = np.array([[0.0, 0.0], [14.6, 0.0], [7.3, 12.64]])
    bls, counts = unique_baselines(pos)
    freqs = np.array([40e6, 44e6, 48e6, 52e6])
    spec = ObservationSpec(freqs=freqs, lst_span_hours=3.0, integration_s=600.0,
                           fr0_halfwidth_hz=0.0)
    A = spectral_basis(freqs, 2, kind="legendre")
    return bls, counts.astype(float), spec, A


@pytest.mark.parametrize("e_oversample", [1, 2])
def test_analyze_matches_brute_force(small_case, e_oversample):
    bls, w, spec, A = small_case
    res = analyze(bls, spec, A, weights=w, snr=3.0, e_oversample=e_oversample)
    ref = _brute_force(bls, w, spec, A, 3.0, res.u_e, res.v_n)
    np.testing.assert_allclose(res.var_ratio, ref, rtol=1e-8, atol=1e-10)


def test_analyze_brute_force_with_fr0_and_taper(small_case):
    bls, w, spec, A = small_case
    spec = ObservationSpec(freqs=spec.freqs, lst_span_hours=3.0,
                           integration_s=1500.0,
                           fr0_halfwidth_hz=spec.fringe_rate_per_u * 1.2)
    res = analyze(bls, spec, A, weights=w, snr=2.0)
    assert np.any(res.status == STATUS_FR0)
    ref = _brute_force(bls, w, spec, A, 2.0, res.u_e, res.v_n)
    np.testing.assert_allclose(res.var_ratio, ref, rtol=1e-8, atol=1e-10)


def test_excluded_bins_keep_prior(small_case):
    bls, w, spec, A = small_case
    spec = ObservationSpec(freqs=spec.freqs, lst_span_hours=3.0,
                           integration_s=3000.0,
                           fr0_halfwidth_hz=spec.fringe_rate_per_u * 1.2)
    res = analyze(bls, spec, A, weights=w, snr=5.0)
    excluded = res.status != STATUS_MEASURED
    assert np.any(res.status == STATUS_FR0)
    assert np.any(res.status == STATUS_ALIAS)
    assert np.all(res.var_ratio[excluded] == 1.0)
    assert np.all(res.n_chan[excluded] == 0)


def test_far_cells_unconstrained(small_case):
    bls, w, spec, A = small_case
    res = analyze(bls, spec, A, weights=w, u_max=40.0)
    UE, VN = np.meshgrid(res.u_e, res.v_n, indexing="ij")
    far = np.hypot(UE, VN) > 30.0
    assert np.all(res.var_ratio[far] == 1.0)
    assert np.all(res.n_eff[far] == 0.0)
    assert np.all(res.n_chan[far] == 0)


def test_n_eff_bounds_and_snr_monotonic(small_case):
    bls, w, spec, A = small_case
    lo = analyze(bls, spec, A, weights=w, snr=1.0)
    hi = analyze(bls, spec, A, weights=w, snr=30.0)
    for r in (lo, hi):
        assert np.all(r.var_ratio > 0) and np.all(r.var_ratio <= 1 + 1e-12)
        assert np.all(r.n_eff >= -1e-12) and np.all(r.n_eff <= A.shape[1] + 1e-12)
    assert np.all(hi.var_ratio <= lo.var_ratio + 1e-12)


def test_more_channels_do_not_lose_information():
    ants = _hex(2)
    bls, counts = unique_baselines(ants)
    f_lo = np.linspace(140e6, 150e6, 6)
    f_hi = np.linspace(140e6, 160e6, 11)       # superset of f_lo
    kw = dict(lst_span_hours=2.0, integration_s=60.0)
    A_lo = spectral_basis(f_lo, 1, kind="legendre")
    A_hi = spectral_basis(f_hi, 1, kind="legendre")
    u_max = 70.0
    r_lo = analyze(bls, ObservationSpec(freqs=f_lo, **kw), A_lo * np.sqrt(6),
                   weights=counts, u_max=u_max, snr=2.0)
    r_hi = analyze(bls, ObservationSpec(freqs=f_hi, **kw), A_hi * np.sqrt(11),
                   weights=counts, u_max=u_max, snr=2.0)
    # A constant spectrum (M = 1, unit amplitude per channel): adding channels
    # only adds measurements of the same unknown.
    assert np.all(r_hi.var_ratio <= r_lo.var_ratio + 1e-12)
    assert np.all(r_hi.n_chan >= r_lo.n_chan)


def test_single_cell_analytic():
    """One baseline, one channel, M = 1: var = 1 / (1 + snr² w K²) at the centre."""
    bls = np.array([[14.6, 0.0]])
    nu = 150e6
    spec = ObservationSpec(freqs=[nu], lst_span_hours=24.0, integration_s=1.0)
    du = spec.fringe_resolution_u
    # A cell far larger than the footprint isolates the footprint centre.
    res = analyze(bls, spec, np.ones((1, 1)), weights=[3.0], snr=2.0,
                  dv=50.0, u_max=12.0, include_conjugates=False)
    centre_e = 14.6 * nu / C
    ie = np.argmin(np.abs(res.u_e - centre_e))
    iv = np.argmin(np.abs(res.v_n))
    dist_m = np.hypot(res.u_e[ie] - centre_e, res.v_n[iv]) * C / nu
    K = disk_aperture_kernel(dist_m) * spec.time_taper(res.u_e[ie])
    assert res.var_ratio[ie, iv, 0] == pytest.approx(1 / (1 + 4 * 3 * K ** 2))
    assert du < 1.0


def test_coverage_counts_and_band_fraction(small_case):
    bls, w, spec, A = small_case
    res = analyze(bls, spec, A, weights=w)
    assert res.n_chan.max() <= spec.freqs.size
    bf = res.band_fraction
    assert np.all((bf >= 0) & (bf <= 1))
    covered = res.n_chan > 0
    assert np.all(np.isfinite(res.nu_lo[covered]))
    assert np.all(np.isnan(res.nu_lo[~covered]))
    assert np.all(res.nu_hi[covered] >= res.nu_lo[covered])


def test_target_mask_and_summary():
    ants = _hex(2)
    bls, counts = unique_baselines(ants)
    freqs = np.linspace(145e6, 155e6, 11)
    spec = ObservationSpec(freqs=freqs, lst_span_hours=2.0, integration_s=60.0,
                           fr0_halfwidth_hz=1e-4)
    A = spectral_basis(freqs, 2)
    res = analyze(bls, spec, A, weights=counts, snr=10.0, coverage_threshold=0.0)
    mask = coupling_target_mask(res, ants, 150e6)
    assert mask.any()
    assert not mask[res.status == STATUS_FR0].any()
    # Without the zero-spacing term, every target cell lies inside some
    # cross-correlation footprint at 150 MHz.
    cross = coupling_target_mask(res, ants, 150e6, include_self=False)
    assert np.all(res.n_chan[cross] > 0)
    assert cross.sum() <= mask.sum()
    s = summarize(res, mask)
    assert s["n_cells"] == mask.sum()
    assert 0 <= s["frac_determined"] <= s["frac_any"] <= 1
    assert 0 <= s["mean_n_eff"] <= 2
    empty = summarize(res, np.zeros_like(mask))
    assert empty["n_cells"] == 0 and np.isnan(empty["mean_n_eff"])


def test_analyze_validates_inputs(small_case):
    bls, w, spec, A = small_case
    with pytest.raises(ValueError):
        analyze(bls, spec, A, weights=w[:1])
    with pytest.raises(ValueError):
        analyze(bls, spec, A, e_oversample=0)


def test_analyze_brute_force_with_prior(small_case):
    bls, w, spec, A = small_case
    pv = np.array([1.0, 1e-3])
    res = analyze(bls, spec, A, weights=w, snr=3.0, prior_var=pv)
    ref = _brute_force(bls, w, spec, A, 3.0, res.u_e, res.v_n, prior_var=pv)
    np.testing.assert_allclose(res.var_ratio, ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(res.prior_var, pv)
    e = res.error_power_fraction
    assert np.all((e > 0) & (e <= 1 + 1e-12))
    np.testing.assert_allclose(e, (res.var_ratio @ pv) / pv.sum())


def test_smoothness_prior_helps(small_case):
    bls, w, spec, A = small_case
    flat = analyze(bls, spec, A, weights=w, snr=3.0)
    smooth = analyze(bls, spec, A, weights=w, snr=3.0, prior_var=[1.0, 1e-4])
    # Tightening the prior on mode 1 can only shrink every posterior variance
    # (Loewner order); mode 0 has the same prior in both, so compare directly.
    assert np.all(smooth.var_ratio[..., 0] <= flat.var_ratio[..., 0] + 1e-12)
    assert np.all(smooth.var_ratio[..., 1] * 1e-4 <= flat.var_ratio[..., 1] + 1e-12)


def test_prior_var_validation(small_case):
    bls, w, spec, A = small_case
    with pytest.raises(ValueError):
        analyze(bls, spec, A, prior_var=[1.0])
    with pytest.raises(ValueError):
        analyze(bls, spec, A, prior_var=[1.0, 0.0])


def test_summary_suppression_fractions(small_case):
    bls, w, spec, A = small_case
    res = analyze(bls, spec, A, weights=w, snr=30.0)
    s = summarize(res, suppressions=(10.0, 100.0))
    assert 0 <= s["frac_supp"][100.0] <= s["frac_supp"][10.0] <= 1


def test_cell_snr_matches_kernel_sum():
    nu, D = 150e6, 14.0
    a = 0.25                      # fine cells so the sum approximates the integral
    rho = D * nu / C
    g = np.arange(-np.ceil(rho / a), np.ceil(rho / a) + 1) * a
    UE, VN = np.meshgrid(g, g, indexing="ij")
    k2sum = np.sum(disk_aperture_kernel(np.hypot(UE, VN) * C / nu, D) ** 2)
    snr_cell = cell_snr_from_visibility_snr(1000.0, nu, a * a, D)
    assert snr_cell == pytest.approx(1000.0 / np.sqrt(k2sum), rel=1e-2)


# ----------------------------------------------------------------------
# recover(), coupling_visibility(), delay_transform()
# ----------------------------------------------------------------------

from newnucal.identifiability import (  # noqa: E402
    coupling_visibility,
    delay_transform,
    recover,
)


@pytest.fixture(scope="module")
def hex7_case():
    ants = _hex(2)
    bls, counts = unique_baselines(ants)
    freqs = np.linspace(145e6, 155e6, 11)
    spec = ObservationSpec(freqs=freqs, lst_span_hours=2.0, integration_s=60.0,
                           fr0_halfwidth_hz=1e-4)
    A = spectral_basis(freqs, 2, kind="legendre")
    return bls, counts.astype(float), spec, A


def test_recover_std_matches_analyze(hex7_case):
    bls, w, spec, A = hex7_case
    pv = np.array([1.0, 1e-2])
    kw = dict(weights=w, snr=5.0, prior_var=pv, e_oversample=2)
    rec = recover(bls, spec, A, **kw)
    res = analyze(bls, spec, A, **kw)
    np.testing.assert_allclose(rec.a_std ** 2 / pv, res.var_ratio, rtol=1e-8)
    np.testing.assert_array_equal(rec.status, res.status)


def test_recover_posterior_is_calibrated(hex7_case):
    """With the truth drawn from the prior, (a_hat - a_true)/a_std has unit rms."""
    bls, w, spec, A = hex7_case
    rec = recover(bls, spec, A, weights=w, snr=5.0, n_samples=8, seed=3)
    constrained = (rec.a_std / np.sqrt(rec.prior_var)) < 0.9
    assert constrained.sum() > 200
    z = (rec.a_hat - rec.a_true)[constrained] / rec.a_std[constrained]
    assert np.mean(np.abs(z) ** 2) == pytest.approx(1.0, abs=0.15)
    zs = (rec.samples - rec.a_hat[None])[:, constrained] / rec.a_std[constrained]
    assert np.mean(np.abs(zs) ** 2) == pytest.approx(1.0, abs=0.1)
    # Unmeasured columns keep the prior: zero mean, prior std.
    out = rec.status != STATUS_MEASURED
    assert np.all(rec.a_hat[out] == 0)
    np.testing.assert_allclose(
        rec.a_std[out], np.broadcast_to(np.sqrt(rec.prior_var), rec.a_std[out].shape))


def test_recover_high_snr_recovers_truth(hex7_case):
    bls, w, spec, A = hex7_case
    rec = recover(bls, spec, A[:, :1], weights=w, snr=1e4, seed=1)
    good = rec.a_std < 0.05
    assert good.sum() > 50
    err = np.abs(rec.a_hat - rec.a_true)[good]
    assert np.all(err <= 5 * rec.a_std[good])
    assert np.max(err) < 0.1          # truth has unit rms


def test_recover_validates_truth_shape(hex7_case):
    bls, w, spec, A = hex7_case
    with pytest.raises(ValueError):
        recover(bls, spec, A, weights=w, a_true=np.zeros((2, 2, 2)))


def test_coupling_visibility_with_iso_kernel_matches_measurement_model(hex7_case):
    """δK = K_iso and point sampling reproduce Σ_c κ T̃ per fringe bin."""
    bls, w, spec, A = hex7_case
    rec = recover(bls, spec, A, weights=w, snr=5.0, e_oversample=2)
    b = bls[0]
    iso = lambda be, bn, nu: disk_aperture_kernel(np.hypot(be, bn), 14.0)  # noqa: E731
    dV, bin_u = coupling_visibility(rec.truth, rec.a_true, b, iso, spec, n_sub=1)
    T = rec.sky(rec.a_true)
    UE, VN = np.meshgrid(rec.u_e, rec.v_n, indexing="ij")
    nsub = rec.params["e_oversample"]
    for f, nu in enumerate(rec.freqs):
        K = disk_aperture_kernel(np.hypot(UE - b[0] * nu / C, VN - b[1] * nu / C)
                                 * C / nu, 14.0)
        terms = (T[:, :, f] * K * spec.time_taper(UE)).sum(axis=1)
        terms[rec.status != STATUS_MEASURED] = 0
        np.testing.assert_allclose(dV[f], terms.reshape(-1, nsub).sum(axis=1),
                                   atol=1e-10)
    np.testing.assert_allclose(bin_u, rec.u_e.reshape(-1, nsub).mean(axis=1))


def test_coupling_visibility_linear_and_filtered(hex7_case):
    bls, w, spec, A = hex7_case
    rec = recover(bls, spec, A, weights=w, snr=5.0)
    dk = lambda be, bn, nu: np.exp(-((np.hypot(be, bn) - 7.0) ** 2))  # noqa: E731
    v1, _ = coupling_visibility(rec, rec.a_true, bls[1], dk, spec)
    v2, _ = coupling_visibility(rec, rec.a_hat, bls[1], dk, spec)
    v12, _ = coupling_visibility(rec, rec.a_true - 2 * rec.a_hat, bls[1], dk, spec)
    np.testing.assert_allclose(v12, v1 - 2 * v2, atol=1e-10)
    zero, _ = coupling_visibility(rec, np.zeros_like(rec.a_true), bls[1], dk, spec)
    assert np.all(zero == 0)
    unmeasured = rec.status[:: rec.params["e_oversample"]] != STATUS_MEASURED
    assert np.all(v1[:, unmeasured] == 0)
    # A support cut that covers everything changes nothing.
    v1s, _ = coupling_visibility(rec, rec.a_true, bls[1], dk, spec, support_m=1e3)
    np.testing.assert_allclose(v1s, v1)


def test_delay_transform_locates_tone():
    freqs = np.linspace(140e6, 160e6, 81)
    tau0 = 400e-9
    delays, xt = delay_transform(np.exp(2j * np.pi * freqs * tau0), freqs)
    assert delays[np.argmax(np.abs(xt))] == pytest.approx(tau0, abs=1 / (20e6))


def test_recover_default_truth_grid_is_fit_grid(hex7_case):
    bls, w, spec, A = hex7_case
    r1 = recover(bls, spec, A, weights=w, snr=5.0, seed=4)
    r2 = recover(bls, spec, A, weights=w, snr=5.0, seed=4, truth_e_oversample=1)
    np.testing.assert_array_equal(r1.truth.u_e, r1.u_e)
    np.testing.assert_allclose(r1.a_hat, r2.a_hat)
    assert r1.truth.params["area_ratio"] == pytest.approx(1.0)


def test_recover_fine_truth_fits_data(hex7_case):
    """A coarse fit to a finer truth still reproduces the measured visibilities."""
    bls, w, spec, A = hex7_case
    rec = recover(bls, spec, A[:, :1], weights=w, snr=1e3, seed=5,
                  truth_e_oversample=4, truth_dv=spec.fringe_resolution_u / 4)
    assert rec.truth.u_e.size == 4 * rec.u_e.size
    assert rec.a_true.shape[1] > rec.a_hat.shape[1]
    assert rec.truth.params["area_ratio"] == pytest.approx(1 / 16)
    iso = lambda be, bn, nu: disk_aperture_kernel(np.hypot(be, bn), 14.0)  # noqa: E731
    for b in bls[:3]:
        vt, _ = coupling_visibility(rec.truth, rec.a_true, b, iso, spec, n_sub=1)
        vh, _ = coupling_visibility(rec, rec.a_hat, b, iso, spec, n_sub=1)
        rel = np.linalg.norm(vt - vh) / np.linalg.norm(vt)
        assert rel < 0.1


def test_sample_interval_sets_alias_not_taper():
    a = ObservationSpec(freqs=[150e6], lst_span_hours=1.5, integration_s=9.66)
    b = ObservationSpec(freqs=[150e6], lst_span_hours=1.5, integration_s=9.66,
                        sample_interval_s=38.66)
    assert a.sample_interval_s == a.integration_s
    assert b.u_alias == pytest.approx(a.u_alias * 9.66 / 38.66)
    assert b.time_taper(50.0) == pytest.approx(a.time_taper(50.0))
    assert b.bin_status(0.9 * a.u_alias) == STATUS_ALIAS
    assert a.bin_status(0.9 * a.u_alias) == STATUS_MEASURED


def test_target_mask_restricted_to_measured_baselines():
    ants = _hex(3)
    bls, counts = unique_baselines(ants)
    freqs = np.linspace(145e6, 155e6, 6)
    spec = ObservationSpec(freqs=freqs, lst_span_hours=2.0, integration_s=60.0)
    res = analyze(bls, spec, spectral_basis(freqs, 1, kind="legendre"),
                  weights=counts, coverage_threshold=0.0)
    full = coupling_target_mask(res, ants, 150e6)
    # All groups and a radius spanning the array: superset of the full set.
    big = coupling_target_mask(res, ants, 150e6, bls=bls, coupling_radius=1e3)
    assert np.all(big[full])
    # Radius 0 with self term: only the measured footprints themselves.
    own = coupling_target_mask(res, ants, 150e6, bls=bls[:3], coupling_radius=0.0,
                               include_self=True)
    UE, VN = np.meshgrid(res.u_e, res.v_n, indexing="ij")
    ref = np.zeros_like(own)
    for b in np.vstack([bls[:3], -bls[:3]]) * 150e6 / C:
        ref |= np.hypot(UE - b[0], VN - b[1]) <= 14.0 * 150e6 / C
    np.testing.assert_array_equal(own, ref & (res.status != STATUS_FR0)[:, None])
    with pytest.raises(ValueError):
        coupling_target_mask(res, ants, 150e6, bls=bls)
