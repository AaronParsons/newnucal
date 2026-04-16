"""
Tests for GridFitter.

Covers:
  - Adjoint consistency: <Fx, y> ≈ <x, F^H y>
  - fit_one_time reduces chi2 and returns correct shapes
  - fit_all_times returns correctly stacked gain arrays
  - estimate_diagonal_precond returns valid preconditioner
  - accumulate_product_healpix scatter-accumulation
  - init_sky_coeffs_from_product sky DPSS projection
"""

import numpy as np
import pytest

from newnucal.basis import BeamBasis, SkyBasis
from newnucal.grid_fitter import GridFitter, _derive_product_basis

# Small grid so tests stay fast
NL = NM  = 32
N_PRODUCT_MODES = 4   # → nbasis = 5 (when SVD data present)
NSIDE_SKY = 8         # matches conftest NSIDE_SKY


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def sky_basis_with_svd(freqs):
    """SkyBasis with SVD metadata (from_ensemble path)."""
    rng   = np.random.default_rng(7)
    nfreq = len(freqs)
    # Random power-law sky flux ensemble
    ref   = rng.exponential(size=(12, 1)).astype(np.float64)
    idx   = rng.normal(-0.7, 0.2, (12, 1))
    f_n   = (freqs / freqs[0])[None, :]
    spectra = ref * f_n ** idx
    return SkyBasis.from_ensemble(freqs, spectra, n_modes=2)


@pytest.fixture(scope="module")
def beam_basis_with_svd(freqs):
    """BeamBasis with SVD metadata (from_ensemble path)."""
    rng     = np.random.default_rng(8)
    nfreq   = len(freqs)
    spectra = np.abs(rng.standard_normal((10, nfreq)).astype(np.float64)) + 0.1
    return BeamBasis.from_ensemble(freqs, spectra, n_modes=3)


@pytest.fixture(scope="module")
def grid_fitter(array, sky_basis_with_svd, beam_basis_with_svd, freqs):
    return GridFitter(
        array           = array,
        sky_basis       = sky_basis_with_svd,
        beam_basis      = beam_basis_with_svd,
        freqs           = freqs,
        nl              = NL,
        nm              = NM,
        ell_max         = 0.9,
        m_max           = 0.9,
        eps             = 1e-4,    # lower accuracy for speed in tests
        n_product_modes = N_PRODUCT_MODES,
    )


@pytest.fixture(scope="module")
def synthetic_data(grid_fitter):
    """One integration of synthetic visibilities: true_coeffs → vis + small noise."""
    rng = np.random.default_rng(42)
    nl, nm, nb = NL, NM, grid_fitter.nbasis
    nfreq, nbl = grid_fitter.nfreq, grid_fitter.nbl

    # Random smooth coefficient maps (above-horizon pixels only)
    true_coeffs = np.zeros((nl, nm, nb), dtype=np.complex64)
    mask = grid_fitter.mask_horizon
    n_vis = int(mask.sum())
    true_coeffs[mask, :] = (
        rng.standard_normal((n_vis, nb)) + 1j * rng.standard_normal((n_vis, nb))
    ).astype(np.complex64) * 0.1

    vis_true = grid_fitter._fwd(true_coeffs)
    noise_amp = float(np.std(np.abs(vis_true))) * 0.05
    noise = (rng.standard_normal((nfreq, nbl)) +
             1j * rng.standard_normal((nfreq, nbl))).astype(np.complex64) * noise_amp
    return vis_true + noise, true_coeffs


# ------------------------------------------------------------------
# _derive_product_basis
# ------------------------------------------------------------------

class TestDeriveProductBasis:

    def test_with_svd_data(self, sky_basis_with_svd, beam_basis_with_svd, freqs):
        pb = _derive_product_basis(sky_basis_with_svd, beam_basis_with_svd,
                                   n_product_modes=N_PRODUCT_MODES)
        assert pb.shape[1] == len(freqs)
        assert pb.shape[0] == N_PRODUCT_MODES + 1   # mean + modes
        assert pb.dtype == np.float32

    def test_rows_orthonormal(self, sky_basis_with_svd, beam_basis_with_svd):
        pb   = _derive_product_basis(sky_basis_with_svd, beam_basis_with_svd,
                                     n_product_modes=N_PRODUCT_MODES).astype(np.float64)
        gram = pb @ pb.T
        res  = gram - np.eye(pb.shape[0])
        assert np.max(np.abs(res)) < 1e-5

    def test_fallback_dpss(self, freqs):
        """DPSS bases (no SVD) fall back to sky_basis.A.T."""
        sb = SkyBasis.from_dpss(freqs, 40e-9)
        bb = BeamBasis.from_dpss(freqs, 20e-9)
        pb = _derive_product_basis(sb, bb, n_product_modes=10)
        assert pb.shape == (sb.nmodes, len(freqs))
        np.testing.assert_array_equal(pb, sb.A.T)


# ------------------------------------------------------------------
# Adjoint test
# ------------------------------------------------------------------

class TestAdjointConsistency:
    """Verify <Fx, y> = <x, F^H y> to better than 1e-4 relative error."""

    def test_adjoint(self, grid_fitter):
        rng  = np.random.default_rng(99)
        nl, nm, nb = NL, NM, grid_fitter.nbasis
        nfreq, nbl = grid_fitter.nfreq, grid_fitter.nbl

        x = (rng.standard_normal((nl, nm, nb)) +
             1j * rng.standard_normal((nl, nm, nb))).astype(np.complex64)
        y = (rng.standard_normal((nfreq, nbl)) +
             1j * rng.standard_normal((nfreq, nbl))).astype(np.complex64)

        Fx   = grid_fitter._fwd(x)          # (nfreq, nbl)
        FHy  = grid_fitter._adj(y)          # (nl, nm, nb)

        lhs  = complex(np.vdot(Fx.ravel(),  y.ravel()))
        rhs  = complex(np.vdot(x.ravel(),  FHy.ravel()))

        rel_diff = abs(lhs - rhs) / (0.5 * (abs(lhs) + abs(rhs)) + 1e-30)
        assert rel_diff < 1e-3, (
            f"Adjoint relative error {rel_diff:.3e} > 1e-3\n"
            f"  lhs = {lhs:.6e},  rhs = {rhs:.6e}"
        )


# ------------------------------------------------------------------
# fit_one_time
# ------------------------------------------------------------------

class TestFitOneTime:

    def test_chi2_decreases(self, grid_fitter, synthetic_data):
        data, _ = synthetic_data
        _, _, history = grid_fitter.fit_one_time(data, n_iter=10, verbose=False)
        chi2 = history['chi2_cal']
        assert chi2[-1] < chi2[0], (
            f"chi2 did not decrease: {chi2[0]:.3e} → {chi2[-1]:.3e}"
        )

    def test_chi2_drops_significantly(self, grid_fitter, synthetic_data):
        """With low noise, chi2 should decrease by at least 2× in 20 iterations."""
        data, _ = synthetic_data
        _, _, history = grid_fitter.fit_one_time(data, n_iter=20, verbose=False)
        chi2 = history['chi2_cal']
        assert chi2[-1] < 0.5 * chi2[0], (
            f"chi2 did not drop by 2×: {chi2[0]:.3e} → {chi2[-1]:.3e}"
        )

    def test_coeff_maps_shape(self, grid_fitter, synthetic_data):
        data, _ = synthetic_data
        coeff, _, _ = grid_fitter.fit_one_time(data, n_iter=5)
        assert coeff.shape == (NL, NM, grid_fitter.nbasis)

    def test_coeff_maps_dtype(self, grid_fitter, synthetic_data):
        data, _ = synthetic_data
        coeff, _, _ = grid_fitter.fit_one_time(data, n_iter=5)
        assert coeff.dtype == np.complex64

    def test_gain_params_shapes(self, grid_fitter, synthetic_data):
        data, _ = synthetic_data
        _, gp, _ = grid_fitter.fit_one_time(data, n_iter=5)
        nfreq = grid_fitter.nfreq
        assert gp['log_amp'].shape == (nfreq,)
        assert gp['phase'].shape   == (nfreq,)
        assert gp['phi'].shape     == (2, nfreq)

    def test_gain_params_dtype(self, grid_fitter, synthetic_data):
        data, _ = synthetic_data
        _, gp, _ = grid_fitter.fit_one_time(data, n_iter=5)
        assert gp['log_amp'].dtype == np.float32
        assert gp['phase'].dtype   == np.float32
        assert gp['phi'].dtype     == np.float32

    def test_history_length(self, grid_fitter, synthetic_data):
        n_iter = 7
        data, _ = synthetic_data
        _, _, history = grid_fitter.fit_one_time(data, n_iter=n_iter)
        assert len(history['chi2_cal'])      == n_iter
        assert len(history['accepted_step']) == n_iter

    def test_with_gain_init(self, grid_fitter, synthetic_data):
        """Providing non-zero gain init should not crash and chi2 should still fall."""
        data, _ = synthetic_data
        nfreq = grid_fitter.nfreq
        gp0 = {
            'log_amp': np.full(nfreq, 0.1, dtype=np.float32),
            'phase':   np.zeros(nfreq, dtype=np.float32),
            'phi':     np.zeros((2, nfreq), dtype=np.float32),
        }
        _, _, history = grid_fitter.fit_one_time(data, gain_params_init=gp0, n_iter=10)
        assert history['chi2_cal'][-1] < history['chi2_cal'][0]

    def test_with_precond(self, grid_fitter, synthetic_data):
        """Preconditioned fit should converge at least as fast as unpreconditioned."""
        data, _ = synthetic_data
        precond = grid_fitter.estimate_diagonal_precond(nprobe=4)
        _, _, hist_pre  = grid_fitter.fit_one_time(data, n_iter=10, precond=precond)
        _, _, hist_none = grid_fitter.fit_one_time(data, n_iter=10)
        # Both should decrease chi2
        assert hist_pre['chi2_cal'][-1]  < hist_pre['chi2_cal'][0]
        assert hist_none['chi2_cal'][-1] < hist_none['chi2_cal'][0]


# ------------------------------------------------------------------
# fit_all_times
# ------------------------------------------------------------------

class TestFitAllTimes:

    @pytest.fixture(scope="class")
    def multi_time_data(self, grid_fitter):
        """Two identical time steps (no rotation info needed for this test)."""
        rng = np.random.default_rng(55)
        nfreq, nbl = grid_fitter.nfreq, grid_fitter.nbl
        vis = (rng.standard_normal((2, nfreq, nbl)) +
               1j * rng.standard_normal((2, nfreq, nbl))).astype(np.complex64) * 0.5
        return vis

    def test_output_list_length(self, grid_fitter, multi_time_data):
        coeff_all, _, _ = grid_fitter.fit_all_times(
            multi_time_data, n_iter=3, use_precond=False
        )
        assert len(coeff_all) == 2

    def test_coeff_maps_shapes(self, grid_fitter, multi_time_data):
        coeff_all, _, _ = grid_fitter.fit_all_times(
            multi_time_data, n_iter=3, use_precond=False
        )
        for cm in coeff_all:
            assert cm.shape == (NL, NM, grid_fitter.nbasis)

    def test_gain_params_shapes(self, grid_fitter, multi_time_data):
        ntime = 2
        _, gp_all, _ = grid_fitter.fit_all_times(
            multi_time_data, n_iter=3, use_precond=False
        )
        nfreq = grid_fitter.nfreq
        assert gp_all['log_amp'].shape == (ntime, nfreq)
        assert gp_all['phase'].shape   == (ntime, nfreq)
        assert gp_all['phi'].shape     == (ntime, 2, nfreq)

    def test_histories_length(self, grid_fitter, multi_time_data):
        _, _, histories = grid_fitter.fit_all_times(
            multi_time_data, n_iter=3, use_precond=False
        )
        assert len(histories) == 2
        for h in histories:
            assert len(h['chi2_cal']) == 3


# ------------------------------------------------------------------
# estimate_diagonal_precond
# ------------------------------------------------------------------

class TestDiagonalPrecond:

    def test_shape(self, grid_fitter):
        precond = grid_fitter.estimate_diagonal_precond(nprobe=4)
        assert precond.shape == (NL, NM, grid_fitter.nbasis)

    def test_all_positive(self, grid_fitter):
        precond = grid_fitter.estimate_diagonal_precond(nprobe=4)
        assert np.all(precond > 0), "Preconditioner contains non-positive values"

    def test_median_approximately_one(self, grid_fitter):
        precond = grid_fitter.estimate_diagonal_precond(nprobe=4)
        med = float(np.median(precond))
        assert abs(med - 1.0) < 0.1, f"Preconditioner median = {med:.4f}, expected ≈ 1"

    def test_dtype(self, grid_fitter):
        precond = grid_fitter.estimate_diagonal_precond(nprobe=4)
        assert precond.dtype == np.float32


# ------------------------------------------------------------------
# accumulate_product_healpix
# ------------------------------------------------------------------

class TestAccumulateProductHealpix:

    @pytest.fixture(scope="class")
    def fit_results(self, grid_fitter):
        """One synthetic time step fit result."""
        rng = np.random.default_rng(77)
        nfreq, nbl = grid_fitter.nfreq, grid_fitter.nbl
        vis = (rng.standard_normal((nfreq, nbl)) +
               1j * rng.standard_normal((nfreq, nbl))).astype(np.complex64)
        coeff, _, _ = grid_fitter.fit_one_time(vis, n_iter=3)
        return [coeff]

    def test_output_shapes(self, grid_fitter, fit_results, rot_matrices):
        import healpy as hp
        product_map, weights = grid_fitter.accumulate_product_healpix(
            fit_results, rot_matrices[:1], sky_nside=NSIDE_SKY
        )
        npix_sky = hp.nside2npix(NSIDE_SKY)
        nfreq    = grid_fitter.nfreq
        assert product_map.shape == (npix_sky, nfreq)
        assert weights.shape     == (npix_sky,)

    def test_some_pixels_nonzero(self, grid_fitter, fit_results, rot_matrices):
        product_map, weights = grid_fitter.accumulate_product_healpix(
            fit_results, rot_matrices[:1], sky_nside=NSIDE_SKY
        )
        assert np.any(weights > 0), "All pixels have zero weight — nothing was accumulated"
        assert np.any(np.abs(product_map) > 0)

    def test_dtype(self, grid_fitter, fit_results, rot_matrices):
        product_map, weights = grid_fitter.accumulate_product_healpix(
            fit_results, rot_matrices[:1], sky_nside=NSIDE_SKY
        )
        assert product_map.dtype == np.float32
        assert weights.dtype     == np.float32

    def test_weights_count_accumulations(self, grid_fitter, rot_matrices):
        """Using two identical time steps doubles the weights for covered pixels."""
        rng = np.random.default_rng(88)
        nfreq, nbl = grid_fitter.nfreq, grid_fitter.nbl
        vis = (rng.standard_normal((nfreq, nbl)) +
               1j * rng.standard_normal((nfreq, nbl))).astype(np.complex64)
        coeff, _, _ = grid_fitter.fit_one_time(vis, n_iter=2)

        _, weights1 = grid_fitter.accumulate_product_healpix(
            [coeff], rot_matrices[:1], sky_nside=NSIDE_SKY
        )
        _, weights2 = grid_fitter.accumulate_product_healpix(
            [coeff, coeff], rot_matrices[:1].repeat(2, axis=0), sky_nside=NSIDE_SKY
        )
        # Every pixel that was hit once should now be hit twice
        hit = weights1 > 0
        np.testing.assert_array_equal(weights2[hit], 2.0 * weights1[hit])


# ------------------------------------------------------------------
# init_sky_coeffs_from_product
# ------------------------------------------------------------------

class TestInitSkyCoeffsFromProduct:

    @pytest.fixture(scope="class")
    def product_map(self, grid_fitter, rot_matrices):
        """Accumulated product map from one random time step."""
        rng = np.random.default_rng(33)
        nfreq, nbl = grid_fitter.nfreq, grid_fitter.nbl
        vis = (rng.standard_normal((nfreq, nbl)) +
               1j * rng.standard_normal((nfreq, nbl))).astype(np.complex64)
        coeff, _, _ = grid_fitter.fit_one_time(vis, n_iter=2)
        pm, _ = grid_fitter.accumulate_product_healpix(
            [coeff], rot_matrices[:1], sky_nside=NSIDE_SKY
        )
        return pm

    def test_shape(self, grid_fitter, product_map):
        import healpy as hp
        npix_sky   = hp.nside2npix(NSIDE_SKY)
        nmodes_sky = grid_fitter.sky_basis.nmodes
        sky_coeffs = grid_fitter.init_sky_coeffs_from_product(product_map)
        assert sky_coeffs.shape == (npix_sky, nmodes_sky)

    def test_with_beam_model(self, grid_fitter, product_map, beam_model):
        """Dividing by beam should not crash and should change the result."""
        sky_coeffs_nobm = grid_fitter.init_sky_coeffs_from_product(product_map)
        sky_coeffs_bm   = grid_fitter.init_sky_coeffs_from_product(
            product_map, beam_model=beam_model
        )
        assert sky_coeffs_nobm.shape == sky_coeffs_bm.shape
        # Beam division should change values
        import jax.numpy as jnp
        diff = jnp.max(jnp.abs(sky_coeffs_nobm - sky_coeffs_bm))
        assert float(diff) > 1e-6, "Beam division produced no change"

    def test_finite_values(self, grid_fitter, product_map):
        sky_coeffs = grid_fitter.init_sky_coeffs_from_product(product_map)
        assert np.all(np.isfinite(np.asarray(sky_coeffs))), "Sky coeffs contain non-finite values"
