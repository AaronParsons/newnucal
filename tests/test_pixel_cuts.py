"""
Tests for pixel-cut functionality.

Covers three independent features:
  1. Static horizon cut  -- build_ever_visible_mask / apply_pixel_mask
  2. Dynamic zenith cut  -- precompute_time_geometry(zenith_cut_scale=...)
  3. Calibrator helpers  -- apply_horizon_cut / apply_zenith_cut /
                            disable_zenith_cut / zenith_cut_scale in fit methods

All tests that mutate a ForwardModel use function-scoped fixtures so they
cannot corrupt the shared session-scoped fixtures in conftest.py.
"""

import numpy as np
import jax
import jax.numpy as jnp
import healpy
import pytest

from newnucal.dpss import dpss_matrix, dpss_project
from newnucal.simulate import ForwardModel
from newnucal.array import HERAArray
from newnucal.beam import BeamModel


# ── conftest constants ────────────────────────────────────────────────────────
NSIDE_SKY   = 8
NSIDE_BEAM  = 8
ETA_SKY     = 40e-9
ETA_BEAM    = 20e-9


# ── local function-scoped fixtures ───────────────────────────────────────────

@pytest.fixture
def fwd(array, beam_model, freqs):
    """Fresh ForwardModel (function-scoped; safe to mutate)."""
    A = dpss_matrix(freqs, ETA_SKY)
    fwd = ForwardModel(array, NSIDE_SKY, beam_model, freqs)
    fwd.set_sky_dpss(A)
    return fwd, A


@pytest.fixture
def rot1():
    """Single identity rotation matrix."""
    return np.eye(3, dtype=np.float32)[None]


@pytest.fixture
def rot2():
    """Two identity rotation matrices."""
    return np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))


@pytest.fixture
def calibrator(array, beam_model, freqs):
    """Minimal Calibrator with synthetic zero data (function-scoped)."""
    from newnucal.calibrator import Calibrator
    ntime, nbls = 2, array.nbls
    nfreq = len(freqs)
    data = np.zeros((ntime, nfreq, nbls), dtype=np.complex64)
    rot = np.tile(np.eye(3, dtype=np.float32), (ntime, 1, 1))
    return Calibrator(
        array=array,
        beam_model=beam_model,
        sky_nside=NSIDE_SKY,
        sky_eta_max=ETA_SKY,
        freqs=freqs,
        rot_matrices=rot,
        data=data,
    )


# ── helper ────────────────────────────────────────────────────────────────────

def _zero_sky(fwd, A_sky):
    return jnp.zeros((fwd.npix_sky, A_sky.shape[1]), dtype=jnp.float32)


def _zenith_sky(fwd, A_sky, px=0):
    """Sky with one bright pixel (default: pixel 0 ≈ zenith in HEALPix RING)."""
    nfreq = fwd.nfreq
    flux = np.zeros((fwd.npix_sky, nfreq), dtype=np.float32)
    flux[px] = 1.0
    return jnp.array(dpss_project(flux, A_sky), dtype=jnp.float32)


# =============================================================================
# 1.  Static horizon cut
# =============================================================================

class TestEverVisibleMask:

    def test_shape_and_dtype(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        assert mask.shape == (healpy.nside2npix(NSIDE_SKY),)
        assert mask.dtype == bool

    def test_some_visible(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        assert mask.sum() > 0

    def test_not_all_visible(self, fwd, rot2):
        """Southern hemisphere pixels should never rise above horizon."""
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        assert mask.sum() < healpy.nside2npix(NSIDE_SKY)

    def test_consistent_with_topo_z(self, fwd, rot2):
        """Each True pixel must be above the horizon in at least one frame."""
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        eq = np.asarray(f.eq_coords)   # (3, npix)
        for px in np.where(mask)[0][:20]:   # spot-check
            any_above = any(
                float((rot2[t] @ eq[:, px])[2]) > 0
                for t in range(rot2.shape[0])
            )
            assert any_above, f"Pixel {px} in mask but never above horizon"

    def test_false_pixels_always_below(self, fwd, rot2):
        """Each False pixel must be below the horizon at every time step."""
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        eq = np.asarray(f.eq_coords)
        never_visible = np.where(~mask)[0]
        for px in never_visible[:20]:
            all_below = all(
                float((rot2[t] @ eq[:, px])[2]) <= 0
                for t in range(rot2.shape[0])
            )
            assert all_below, f"Pixel {px} excluded from mask but is above horizon"

    def test_pixel_0_visible_for_identity(self, fwd, rot1):
        """Pixel 0 (north pole) should always be above the horizon for identity rotation."""
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot1)
        assert mask[0], "Pixel 0 (zenith) should be visible under identity rotation"


class TestApplyPixelMask:

    def test_npix_reduced(self, fwd, rot2):
        f, _ = fwd
        npix_before = f.npix_sky
        mask = f.build_ever_visible_mask(rot2)
        f.apply_pixel_mask(mask)
        assert f.npix_sky == int(mask.sum())
        assert f.npix_sky < npix_before

    def test_eq_coords_compressed(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        f.apply_pixel_mask(mask)
        assert f.eq_coords.shape == (3, int(mask.sum()))

    def test_pixel_indices_stored(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_ever_visible_mask(rot2)
        f.apply_pixel_mask(mask)
        assert hasattr(f, '_pixel_indices')
        assert len(f._pixel_indices) == int(mask.sum())
        assert f._pixel_indices[0] == np.where(mask)[0][0]

    def test_geometry_invalidated(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2)
        assert f._geom_ready
        mask = f.build_ever_visible_mask(rot2)
        f.apply_pixel_mask(mask)
        assert not f._geom_ready

    def test_simulate_after_mask(self, fwd, rot2):
        """simulate() should return correct shape and finite values after masking."""
        f, A = fwd
        mask = f.build_ever_visible_mask(rot2)
        f.apply_pixel_mask(mask)
        f.precompute_time_geometry(rot2)
        vis = f.simulate(_zero_sky(f, A), rot2)
        assert vis.shape == (2, f.nfreq, f.nbls)
        assert jnp.issubdtype(vis.dtype, jnp.complexfloating)
        assert jnp.allclose(vis, 0.0, atol=1e-5)

    def test_zenith_source_nonzero_after_mask(self, fwd, rot1):
        """A pixel above the horizon, retained by the mask, should give nonzero vis."""
        f, A = fwd
        mask = f.build_ever_visible_mask(rot1)
        f.apply_pixel_mask(mask)
        f.precompute_time_geometry(rot1)
        # Pixel 0 in the compressed sky corresponds to the lowest original index
        sky = _zenith_sky(f, A, px=0)
        vis = f.simulate(sky, rot1)
        assert float(jnp.abs(vis).mean()) > 1e-5

    def test_result_matches_full_model(self, fwd, rot1):
        """Masked model must give the same vis as full model for a retained source."""
        f, A = fwd
        npix_full = f.npix_sky

        # Full model: pixel 0 (zenith)
        flux_full = np.zeros((npix_full, f.nfreq), dtype=np.float32)
        flux_full[0] = 1.0
        sky_full = jnp.array(dpss_project(flux_full, A), dtype=jnp.float32)
        f.precompute_time_geometry(rot1)
        vis_full = f.simulate(sky_full, rot1)

        # Masked model: apply horizon cut, find where pixel 0 landed
        mask = f.build_ever_visible_mask(rot1)
        assert mask[0], "pixel 0 must survive horizon mask for this test"
        f.apply_pixel_mask(mask)
        f.precompute_time_geometry(rot1)

        px0_compressed = int(np.searchsorted(f._pixel_indices, 0))
        flux_masked = np.zeros((f.npix_sky, f.nfreq), dtype=np.float32)
        flux_masked[px0_compressed] = 1.0
        sky_masked = jnp.array(dpss_project(flux_masked, A), dtype=jnp.float32)
        vis_masked = f.simulate(sky_masked, rot1)

        rms_err = float(jnp.sqrt(jnp.mean(jnp.abs(vis_full - vis_masked) ** 2)))
        rms_sig = float(jnp.sqrt(jnp.mean(jnp.abs(vis_full) ** 2))) + 1e-30
        assert rms_err / rms_sig < 1e-2, (
            f"Masked model relative error {rms_err/rms_sig:.3e} > 1e-2"
        )


# =============================================================================
# 2.  Dynamic zenith cut
# =============================================================================

class TestZenithCutSetup:

    def test_inactive_by_default(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2)
        assert not f._zenith_cut_active

    def test_active_after_cut(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        assert f._zenith_cut_active

    def test_pixel_lists_populated(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        assert f._zenith_px_idx_list is not None
        assert len(f._zenith_px_idx_list) == 2
        assert len(f._zenith_beam_spec_list) == 2
        assert len(f._zenith_src_x_list) == 2

    def test_fewer_pixels_than_full(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        for t in range(2):
            n_active = len(f._zenith_px_idx_list[t])
            assert n_active < f.npix_sky, (
                f"Time {t}: zenith cut kept all {f.npix_sky} pixels"
            )

    def test_larger_scale_more_pixels(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=0.5)
        n_tight = [len(px) for px in f._zenith_px_idx_list]
        f.precompute_time_geometry(rot2, zenith_cut_scale=2.0)
        n_wide = [len(px) for px in f._zenith_px_idx_list]
        for t in range(2):
            assert n_wide[t] >= n_tight[t], (
                f"Time {t}: wider scale gave fewer pixels ({n_wide[t]} < {n_tight[t]})"
            )

    def test_pixel_0_included_at_scale_1(self, fwd, rot2):
        """Pixel 0 (zenith) must always be inside a scale=1.0 cut."""
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        for t in range(2):
            assert 0 in f._zenith_px_idx_list[t], (
                f"Time {t}: pixel 0 missing from zenith cut at scale=1.0"
            )

    def test_beam_spec_shape(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        for t in range(2):
            npix_t = len(f._zenith_px_idx_list[t])
            assert f._zenith_beam_spec_list[t].shape == (npix_t, f.nfreq)

    def test_disable_clears_flag(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        assert f._zenith_cut_active
        f.precompute_time_geometry(rot2, zenith_cut_scale=None)
        assert not f._zenith_cut_active
        assert f._zenith_px_idx_list is None

    def test_requires_beam_diameter(self, array, freqs):
        """zenith_cut_scale should raise if the beam has no diameter."""
        from unittest.mock import MagicMock
        beam = MagicMock()
        beam.nside = NSIDE_BEAM
        beam.A_beam = dpss_matrix(freqs, ETA_BEAM)
        beam.coeffs = np.zeros((healpy.nside2npix(NSIDE_BEAM), beam.A_beam.shape[1]))
        beam.beam = MagicMock(spec=[])   # no 'diameter' attribute
        rot = np.eye(3, dtype=np.float32)[None]
        f = ForwardModel(array, NSIDE_SKY, beam, freqs)
        f.set_sky_dpss(dpss_matrix(freqs, ETA_SKY))
        with pytest.raises(ValueError, match="diameter"):
            f.precompute_time_geometry(rot, zenith_cut_scale=1.0)


class TestZenithCutResults:

    def test_output_shape(self, fwd, rot2):
        f, A = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        vis = f.simulate(_zero_sky(f, A), rot2)
        assert vis.shape == (2, f.nfreq, f.nbls)

    def test_zero_sky_gives_zero_vis(self, fwd, rot2):
        f, A = fwd
        f.precompute_time_geometry(rot2, zenith_cut_scale=1.0)
        vis = f.simulate(_zero_sky(f, A), rot2)
        assert jnp.allclose(vis, 0.0, atol=1e-5)

    def test_zenith_source_nonzero(self, fwd, rot1):
        f, A = fwd
        f.precompute_time_geometry(rot1, zenith_cut_scale=1.0)
        sky = _zenith_sky(f, A, px=0)
        vis = f.simulate(sky, rot1)
        assert float(jnp.abs(vis).mean()) > 1e-5

    def test_zenith_source_matches_full_model(self, fwd, rot1):
        """
        For a source at pixel 0 (zenith), the zenith-cut model should match
        the full model to within float32 precision (~1e-3 relative RMS).
        """
        f, A = fwd
        sky = _zenith_sky(f, A, px=0)

        f.precompute_time_geometry(rot1)
        vis_full = f.simulate(sky, rot1)

        f.precompute_time_geometry(rot1, zenith_cut_scale=1.0)
        assert f._zenith_cut_active
        vis_cut = f.simulate(sky, rot1)

        rms_err = float(jnp.sqrt(jnp.mean(jnp.abs(vis_full - vis_cut) ** 2)))
        rms_sig = float(jnp.sqrt(jnp.mean(jnp.abs(vis_full) ** 2))) + 1e-30
        assert rms_err / rms_sig < 1e-2, (
            f"Zenith-cut model relative error {rms_err/rms_sig:.3e} > 1e-2"
        )

    def test_excluded_source_near_zero(self, fwd, rot1):
        """
        A source well below the zenith cut should produce near-zero visibilities.
        Uses a very tight scale (0.05) so only a handful of pixels near zenith
        are included; any pixel with zenith angle > ~5° is excluded.
        """
        f, A = fwd
        f.precompute_time_geometry(rot1, zenith_cut_scale=0.05)
        active_0 = set(f._zenith_px_idx_list[0].tolist())

        # Find a pixel above the horizon (eq_z > 0) that is NOT in the cut
        eq = np.asarray(f.eq_coords)
        above_horizon = np.where(eq[2] > 0)[0]
        outside_cut = [px for px in above_horizon if px not in active_0]

        if len(outside_cut) == 0:
            pytest.skip("No above-horizon pixels outside tight zenith cut at nside=8")

        # Bright source at an excluded pixel
        px_out = outside_cut[0]
        flux = np.zeros((f.npix_sky, f.nfreq), dtype=np.float32)
        flux[px_out] = 100.0
        sky = jnp.array(dpss_project(flux, A), dtype=jnp.float32)

        vis = f.simulate(sky, rot1)
        assert float(jnp.abs(vis).max()) < 1e-3, (
            "Source outside zenith cut leaked into visibilities"
        )

    def test_output_dtype_complex(self, fwd, rot1):
        f, A = fwd
        f.precompute_time_geometry(rot1, zenith_cut_scale=1.0)
        vis = f.simulate(_zenith_sky(f, A), rot1)
        assert jnp.issubdtype(vis.dtype, jnp.complexfloating)

    def test_gradient_flows_through_zenith_cut(self, fwd, rot1):
        """jax.grad should work through _simulate_loop (zenith cut path)."""
        f, A = fwd
        f.precompute_time_geometry(rot1, zenith_cut_scale=1.0)
        sky = _zenith_sky(f, A)

        def loss(sc):
            return jnp.sum(jnp.abs(f.simulate(sc, rot1)) ** 2).real

        grad = jax.grad(loss)(sky)
        assert grad.shape == sky.shape
        assert jnp.isfinite(grad).all(), "Gradient contains non-finite values"

    def test_jit_compiles_with_zenith_cut(self, fwd, rot1):
        """simulate() should be JIT-compilable under the zenith cut path."""
        f, A = fwd
        f.precompute_time_geometry(rot1, zenith_cut_scale=1.0)
        sky = _zenith_sky(f, A)

        sim_jit = jax.jit(lambda sc: f.simulate(sc, rot1))
        vis = sim_jit(sky)
        assert vis.shape == (1, f.nfreq, f.nbls)
        assert jnp.isfinite(vis).all()


# =============================================================================
# 3.  Calibrator convenience methods
# =============================================================================

class TestCalibratorPixelCuts:

    def test_apply_horizon_cut_reduces_npix(self, calibrator):
        npix_before = calibrator.fwd.npix_sky
        n_keep = calibrator.apply_horizon_cut()
        assert calibrator.fwd.npix_sky < npix_before
        assert calibrator.fwd.npix_sky == n_keep

    def test_apply_horizon_cut_recomputes_geometry(self, calibrator):
        calibrator.apply_horizon_cut()
        assert calibrator.fwd._geom_ready

    def test_apply_zenith_cut_sets_active(self, calibrator):
        calibrator.apply_zenith_cut(scale=1.0)
        assert calibrator.fwd._zenith_cut_active

    def test_disable_zenith_cut_clears_active(self, calibrator):
        calibrator.apply_zenith_cut(scale=1.0)
        calibrator.disable_zenith_cut()
        assert not calibrator.fwd._zenith_cut_active

    def test_horizon_then_zenith_cut(self, calibrator):
        """Horizon cut followed by zenith cut should both take effect."""
        calibrator.apply_horizon_cut()
        npix_after_horizon = calibrator.fwd.npix_sky
        calibrator.apply_zenith_cut(scale=0.5)
        assert calibrator.fwd._zenith_cut_active
        for t in range(calibrator.ntime):
            n_zenith = len(calibrator.fwd._zenith_px_idx_list[t])
            assert n_zenith <= npix_after_horizon

    def test_loss_finite_after_horizon_cut(self, calibrator):
        params = calibrator.init_params()
        calibrator.apply_horizon_cut()
        # Rebuild sky_coeffs for new npix
        params['sky_coeffs'] = jnp.zeros(
            (calibrator.fwd.npix_sky, calibrator.A_sky.shape[1])
        )
        loss = calibrator.calc_loss(params)
        assert np.isfinite(loss)


# =============================================================================
# 4.  zenith_cut_scale in fit methods
# =============================================================================

class TestFitMethodZenithCut:
    """Verify that zenith_cut_scale is applied during the fit and always restored."""

    @pytest.fixture
    def cal_and_params(self, calibrator):
        params = calibrator.init_params()
        return calibrator, params

    def test_fit_alternating_dirty_restores_cut(self, cal_and_params):
        cal, params = cal_and_params
        assert not cal.fwd._zenith_cut_active
        sky, gp, _ = cal.fit_alternating_dirty(
            params['sky_coeffs'], {k: params[k] for k in ('log_amp', 'phase', 'phi')},
            n_outer=1, zenith_cut_scale=1.0,
        )
        assert not cal.fwd._zenith_cut_active, (
            "fit_alternating_dirty did not restore zenith cut after returning"
        )

    def test_fit_alternating_dirty_anderson_restores_cut(self, cal_and_params):
        cal, params = cal_and_params
        sky, gp, _ = cal.fit_alternating_dirty_anderson(
            params['sky_coeffs'], {k: params[k] for k in ('log_amp', 'phase', 'phi')},
            n_outer=1, zenith_cut_scale=1.0,
        )
        assert not cal.fwd._zenith_cut_active

    def test_fit_alternating_restores_cut(self, cal_and_params):
        cal, params = cal_and_params
        sky, gp, _ = cal.fit_alternating(
            params['sky_coeffs'], {k: params[k] for k in ('log_amp', 'phase', 'phi')},
            n_outer=1, zenith_cut_scale=1.0,
        )
        assert not cal.fwd._zenith_cut_active

    def test_cut_is_active_during_fit(self, cal_and_params):
        """By checking the pixel count, confirm the cut was active while fitting."""
        cal, params = cal_and_params
        # Without zenith cut, simulate uses vmap path (_zenith_cut_active=False)
        # With it, simulate uses loop path.  We capture the state mid-fit via a side-effect.
        active_flags = []

        original_fit_gains = cal.fit_gains_linear.__func__

        def patched(self, sky_coeffs):
            active_flags.append(self.fwd._zenith_cut_active)
            return original_fit_gains(self, sky_coeffs)

        import types
        cal.fit_gains_linear = types.MethodType(patched, cal)

        cal.fit_alternating_dirty(
            params['sky_coeffs'], {k: params[k] for k in ('log_amp', 'phase', 'phi')},
            n_outer=1, zenith_cut_scale=1.0,
        )
        assert any(active_flags), "Zenith cut was never active during fit"
        assert not cal.fwd._zenith_cut_active, "Zenith cut not restored after fit"

    def test_no_scale_leaves_state_unchanged(self, cal_and_params):
        """When zenith_cut_scale=None, the fit must not change zenith cut state."""
        cal, params = cal_and_params
        cal.apply_zenith_cut(scale=1.0)
        assert cal.fwd._zenith_cut_active

        cal.fit_alternating_dirty(
            params['sky_coeffs'], {k: params[k] for k in ('log_amp', 'phase', 'phi')},
            n_outer=1,          # no zenith_cut_scale → default None
        )
        # State should be unchanged (still active)
        assert cal.fwd._zenith_cut_active, (
            "fit_alternating_dirty cleared zenith cut when scale=None"
        )


# =============================================================================
# 5.  Anderson acceleration (unit tests for _anderson_coeffs)
# =============================================================================

class TestAndersonCoeffs:
    from newnucal.calibrator import Calibrator as _Cal

    def test_shape(self):
        from newnucal.calibrator import Calibrator
        n, d = 4, 10
        F = np.random.default_rng(0).standard_normal((n, d))
        beta = Calibrator._anderson_coeffs(F)
        assert beta.shape == (n,)

    def test_sums_to_one(self):
        """The constraint sum_i beta_i = 1 must be satisfied."""
        from newnucal.calibrator import Calibrator
        F = np.random.default_rng(1).standard_normal((5, 20))
        beta = Calibrator._anderson_coeffs(F)
        assert abs(beta.sum() - 1.0) < 1e-6, f"beta sums to {beta.sum():.6f}, not 1"

    def test_trivial_case(self):
        """With n=1 the only solution is beta=[1]."""
        from newnucal.calibrator import Calibrator
        F = np.array([[1.0, 2.0, 3.0]])
        beta = Calibrator._anderson_coeffs(F)
        assert abs(beta[0] - 1.0) < 1e-6

    def test_finite_values(self):
        from newnucal.calibrator import Calibrator
        F = np.random.default_rng(42).standard_normal((6, 50))
        beta = Calibrator._anderson_coeffs(F)
        assert np.isfinite(beta).all()
