"""
Tests for pixel-cut functionality.

Covers two independent features:
  1. Static horizon cut  -- build_ever_visible_mask / apply_sky_mask
  2. Calibrator helpers  -- apply_horizon_cut

All tests that mutate a ForwardModel use function-scoped fixtures so they
cannot corrupt the shared session-scoped fixtures in conftest.py.
"""

import numpy as np
import jax
import jax.numpy as jnp
import healpy
import pytest

from newnucal.basis import dpss_matrix, basis_project
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
    from newnucal.sky import SkyModel
    from newnucal.basis import SkyBasis
    sky_model = SkyModel(nside=NSIDE_SKY, freqs=freqs,
                         basis=SkyBasis.from_dpss(freqs, ETA_SKY))
    fwd = ForwardModel(array, sky_model, beam_model, freqs)
    return fwd, np.asarray(sky_model.A)


@pytest.fixture
def rot1():
    """Single identity rotation matrix."""
    return np.eye(3, dtype=np.float32)[None]


@pytest.fixture
def rot2():
    """Two identity rotation matrices."""
    return np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))


@pytest.fixture
def calibrator(array, beam_model, freqs, sky_model):
    """Minimal Calibrator with synthetic zero data (function-scoped)."""
    from newnucal.calibrator import Calibrator
    ntime, nbls = 2, array.nbls
    nfreq = len(freqs)
    data = np.zeros((ntime, nfreq, nbls), dtype=np.complex64)
    rot = np.tile(np.eye(3, dtype=np.float32), (ntime, 1, 1))
    return Calibrator(
        array=array,
        beam_model=beam_model,
        sky_model=sky_model,
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
    return jnp.array(basis_project(flux, A_sky), dtype=jnp.float32)


# =============================================================================
# 1.  Static horizon cut
# =============================================================================

class TestEverVisibleMask:

    def test_shape_and_dtype(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        assert mask.shape == (healpy.nside2npix(NSIDE_SKY),)
        assert mask.dtype == bool

    def test_some_visible(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        assert mask.sum() > 0

    def test_not_all_visible(self, fwd, rot2):
        """Southern hemisphere pixels should never rise above horizon."""
        f, _ = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        assert mask.sum() < healpy.nside2npix(NSIDE_SKY)

    def test_consistent_with_topo_z(self, fwd, rot2):
        """Each True pixel must be above the horizon in at least one frame."""
        f, _ = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
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
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
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
        mask = f.build_sky_mask_altitude(rot1, min_altitude_deg=0.0)
        assert mask[0], "Pixel 0 (zenith) should be visible under identity rotation"


class TestApplyPixelMask:

    def test_npix_reduced(self, fwd, rot2):
        f, _ = fwd
        npix_before = f.npix_sky
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        f.apply_sky_mask(mask)
        assert f.npix_sky == int(mask.sum())
        assert f.npix_sky < npix_before

    def test_eq_coords_compressed(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        f.apply_sky_mask(mask)
        assert f.eq_coords.shape == (3, int(mask.sum()))

    def test_pixel_indices_stored(self, fwd, rot2):
        f, _ = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        f.apply_sky_mask(mask)
        assert hasattr(f, '_pixel_indices')
        assert len(f._pixel_indices) == int(mask.sum())
        assert f._pixel_indices[0] == np.where(mask)[0][0]

    def test_geometry_invalidated(self, fwd, rot2):
        f, _ = fwd
        f.precompute_time_geometry(rot2)
        assert f._geom_ready
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        f.apply_sky_mask(mask)
        assert not f._geom_ready

    def test_simulate_after_mask(self, fwd, rot2):
        """simulate() should return correct shape and finite values after masking."""
        f, A = fwd
        mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        f.apply_sky_mask(mask)
        f.precompute_time_geometry(rot2)
        vis = f.simulate(_zero_sky(f, A), rot2)
        assert vis.shape == (2, f.nfreq, f.nbls)
        assert jnp.issubdtype(vis.dtype, jnp.complexfloating)
        assert jnp.allclose(vis, 0.0, atol=1e-5)

    def test_zenith_source_nonzero_after_mask(self, fwd, rot1):
        """A pixel above the horizon, retained by the mask, should give nonzero vis."""
        f, A = fwd
        mask = f.build_sky_mask_altitude(rot1, min_altitude_deg=0.0)
        f.apply_sky_mask(mask)
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
        sky_full = jnp.array(basis_project(flux_full, A), dtype=jnp.float32)
        f.precompute_time_geometry(rot1)
        vis_full = f.simulate(sky_full, rot1)

        # Masked model: apply horizon cut, find where pixel 0 landed
        mask = f.build_sky_mask_altitude(rot1, min_altitude_deg=0.0)
        assert mask[0], "pixel 0 must survive horizon mask for this test"
        f.apply_sky_mask(mask)
        f.precompute_time_geometry(rot1)

        px0_compressed = int(np.searchsorted(f._pixel_indices, 0))
        flux_masked = np.zeros((f.npix_sky, f.nfreq), dtype=np.float32)
        flux_masked[px0_compressed] = 1.0
        sky_masked = jnp.array(basis_project(flux_masked, A), dtype=jnp.float32)
        vis_masked = f.simulate(sky_masked, rot1)

        rms_err = float(jnp.sqrt(jnp.mean(jnp.abs(vis_full - vis_masked) ** 2)))
        rms_sig = float(jnp.sqrt(jnp.mean(jnp.abs(vis_full) ** 2))) + 1e-30
        assert rms_err / rms_sig < 1e-2, (
            f"Masked model relative error {rms_err/rms_sig:.3e} > 1e-2"
        )

    def test_mask_supersession_empty_then_valid(self, fwd, rot2):
        """Applying a new mask should supersede a previous empty mask."""
        f, A = fwd
        npix_full = f.npix_sky

        # First: apply an empty mask (e.g., no pixels above altitude threshold)
        empty_mask = np.zeros(npix_full, dtype=bool)
        f.apply_sky_mask(empty_mask)
        f.precompute_time_geometry(rot2)
        assert f.npix_sky == 0, "Empty mask should result in 0 active pixels"

        # Second: apply a valid mask (e.g., some pixels above horizon)
        valid_mask = f.build_sky_mask_altitude(rot2, min_altitude_deg=0.0)
        assert valid_mask.sum() > 0, "Valid mask should have > 0 pixels"
        f.apply_sky_mask(valid_mask)  # Should not error
        f.precompute_time_geometry(rot2)
        assert f.npix_sky == int(valid_mask.sum()), "Active pixel count mismatch after supersession"
        assert f.eq_coords.shape == (3, int(valid_mask.sum())), "eq_coords shape incorrect"

    def test_build_sky_mask_from_beam_returns_full_sky_size(self, fwd, rot2):
        """build_sky_mask_from_beam_pixels should always return full-sky-size mask."""
        f, _ = fwd
        f.precompute_time_geometry(rot2)

        # Create a beam mask (some beam pixels)
        beam_mask = np.zeros(healpy.nside2npix(NSIDE_BEAM), dtype=bool)
        beam_mask[:healpy.nside2npix(NSIDE_BEAM) // 4] = True

        # Get sky mask: should be full-sky size
        sky_mask = f.build_sky_mask_from_beam_pixels(beam_mask)
        assert sky_mask.shape[0] == f.npix_sky_full, (
            f"Expected full-sky mask size {f.npix_sky_full}, got {sky_mask.shape[0]}"
        )


# =============================================================================
# 2.  Calibrator convenience methods
# =============================================================================

class TestCalibratorPixelCuts:

    def test_apply_horizon_cut_reduces_npix(self, calibrator):
        npix_before = calibrator.fwd.npix_sky
        mask = calibrator.build_sky_mask_altitude(min_altitude_deg=0.0)
        calibrator.apply_sky_mask(mask)
        assert calibrator.fwd.npix_sky < npix_before
        assert calibrator.fwd.npix_sky == int(mask.sum())

    def test_apply_horizon_cut_recomputes_geometry(self, calibrator):
        mask = calibrator.build_sky_mask_altitude(min_altitude_deg=0.0)
        calibrator.apply_sky_mask(mask)
        assert calibrator.fwd._geom_ready

    def test_loss_finite_after_horizon_cut(self, calibrator):
        params = calibrator.init_params()
        mask = calibrator.build_sky_mask_altitude(min_altitude_deg=0.0)
        calibrator.apply_sky_mask(mask)
        # Rebuild sky_coeffs for new npix
        params['sky_coeffs'] = jnp.zeros(
            (calibrator.fwd.npix_sky, calibrator.fwd.A_sky.shape[1])
        )
        loss = calibrator.calc_loss(params)
        assert np.isfinite(loss)


# =============================================================================
# 3.  Anderson acceleration (unit tests for AndersonAccelerator)
# =============================================================================

class TestAndersonAccelerator:

    def test_solve_coeffs_shape(self):
        from newnucal.calibrator import AndersonAccelerator
        n, d = 4, 10
        F = np.random.default_rng(0).standard_normal((n, d))
        beta = AndersonAccelerator._solve_coeffs(F)
        assert beta.shape == (n,)

    def test_solve_coeffs_sums_to_one(self):
        """The constraint sum_i beta_i = 1 must be satisfied."""
        from newnucal.calibrator import AndersonAccelerator
        F = np.random.default_rng(1).standard_normal((5, 20))
        beta = AndersonAccelerator._solve_coeffs(F)
        assert abs(beta.sum() - 1.0) < 1e-6, f"beta sums to {beta.sum():.6f}, not 1"

    def test_solve_coeffs_trivial_case(self):
        """With n=2 and orthogonal rows, sum to 1."""
        from newnucal.calibrator import AndersonAccelerator
        F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        beta = AndersonAccelerator._solve_coeffs(F)
        assert beta is not None
        assert abs(beta.sum() - 1.0) < 1e-6

    def test_solve_coeffs_finite_values(self):
        from newnucal.calibrator import AndersonAccelerator
        F = np.random.default_rng(42).standard_normal((6, 50))
        beta = AndersonAccelerator._solve_coeffs(F)
        assert np.isfinite(beta).all()

    def test_push_returns_none_when_disabled(self):
        from newnucal.calibrator import AndersonAccelerator
        acc = AndersonAccelerator(history=0)
        x = np.zeros(10)
        g = np.ones(10)
        assert acc.push(x, g) is None

    def test_push_returns_none_before_start(self):
        from newnucal.calibrator import AndersonAccelerator
        acc = AndersonAccelerator(history=5, start=2)
        x = np.zeros(10)
        for i in range(2):
            result = acc.push(x, x + 0.1 * i)
            assert result is None, f"Expected None on push {i} (before start=2)"

    def test_push_returns_array_after_start(self):
        from newnucal.calibrator import AndersonAccelerator
        rng = np.random.default_rng(7)
        acc = AndersonAccelerator(history=5, start=2, damping=0.5)
        x = rng.standard_normal(20)
        g = rng.standard_normal(20)
        acc.push(x, g)            # step 0 — before start
        acc.push(x + 0.1, g - 0.1)  # step 1 — before start
        result = acc.push(x + 0.2, g + 0.2)  # step 2 — should return candidate
        assert result is not None
        assert result.shape == (20,)
        assert np.isfinite(result).all()

    def test_push_candidate_is_damped_mix(self):
        """With damping=1 the output is exactly g_aa; with damping=0 it is g_plain."""
        from newnucal.calibrator import AndersonAccelerator
        rng = np.random.default_rng(99)
        n = 30

        acc0 = AndersonAccelerator(history=5, start=0, damping=0.0)
        acc1 = AndersonAccelerator(history=5, start=0, damping=1.0)

        x = rng.standard_normal(n)
        g1 = rng.standard_normal(n)
        g2 = rng.standard_normal(n)

        r0 = acc0.push(x, g1); r1 = acc1.push(x, g1)  # step 0: only 1 in history
        r0 = acc0.push(x + 0.1, g2)
        r1 = acc1.push(x + 0.1, g2)

        # damping=0: candidate == g_plain
        assert r0 is not None and np.allclose(r0, g2, atol=1e-12)
        # damping=1: candidate == g_aa (not necessarily g_plain)
        assert r1 is not None

    def test_clear_resets_state(self):
        from newnucal.calibrator import AndersonAccelerator
        rng = np.random.default_rng(3)
        acc = AndersonAccelerator(history=5, start=0)
        for _ in range(4):
            acc.push(rng.standard_normal(10), rng.standard_normal(10))
        acc.clear()
        assert acc._step == 0
        assert len(acc._hist_g) == 0
        assert len(acc._hist_f) == 0

    def test_history_trimmed_to_max(self):
        from newnucal.calibrator import AndersonAccelerator
        rng = np.random.default_rng(5)
        acc = AndersonAccelerator(history=3, start=0)
        for _ in range(10):
            acc.push(rng.standard_normal(8), rng.standard_normal(8))
        assert len(acc._hist_g) <= 3
        assert len(acc._hist_f) <= 3
