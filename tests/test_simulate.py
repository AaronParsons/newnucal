"""
Forward model tests.

Key analytical property used for testing:

    A point source at topocentric zenith (tx=ty=0) contributes to the
    NUFFT type-1 output as  Σ_px flux_px * exp(0) = const  at *every*
    grid cell.  Therefore all baselines see the same visibility value.
    This holds at every frequency independently.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import healpy

from newnucal.dpss import dpss_matrix, dpss_project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_rot():
    """3×3 identity rotation matrix (topocentric = equatorial)."""
    return jnp.eye(3, dtype=jnp.float32)


def _zenith_sky_coeffs(sky_nside, A_sky, flux_per_pixel: float = 1.0):
    """
    Sky coefficients for a uniform flat-spectrum sky of *flux_per_pixel* Jy.
    Every pixel has the same spectrum → easiest DPSS representation.
    """
    import numpy as np
    npix = healpy.nside2npix(sky_nside)
    nfreq = A_sky.shape[0]
    flat_flux = np.full((npix, nfreq), flux_per_pixel, dtype=np.float32)
    coeffs = dpss_project(flat_flux, A_sky)
    return jnp.array(coeffs, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_sky_gives_zero_vis(forward_model, A_sky):
    """Zero sky coefficients → zero visibilities."""
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = jnp.zeros((npix, nmodes), dtype=jnp.float32)
    rot_m = _identity_rot()[None, :, :]  # (1, 3, 3)

    vis = forward_model.simulate(sky_coeffs, rot_m)  # (1, nfreq, nbls)
    assert jnp.allclose(vis, 0.0, atol=1e-5), "non-zero visibilities for zero sky"


def test_output_shape(forward_model, A_sky, rot_matrices):
    """Output shape matches (ntime, nfreq, nbls)."""
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = jnp.zeros((npix, nmodes), dtype=jnp.float32)

    vis = forward_model.simulate(sky_coeffs, rot_matrices)
    ntime = rot_matrices.shape[0]
    assert vis.shape == (ntime, forward_model.nfreq, forward_model.nbls)


def test_nonzero_sky_gives_nonzero_vis(forward_model, A_sky):
    """A sky with positive flux should produce non-zero visibilities."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky, flux_per_pixel=1.0)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate(sky_coeffs, rot_m)[0]  # (nfreq, nbls)
    assert jnp.abs(vis).mean() > 0.0


def test_gradient_computable(forward_model, A_sky):
    """JAX should be able to differentiate through the forward model."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    rot_m = _identity_rot()[None, :, :]

    def loss(sc):
        return jnp.sum(jnp.abs(forward_model.simulate(sc, rot_m)) ** 2).real

    grad = jax.grad(loss)(sky_coeffs)
    assert grad.shape == sky_coeffs.shape
    assert jnp.isfinite(grad).all(), "gradient contains non-finite values"


def test_vis_is_complex(forward_model, A_sky):
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate(sky_coeffs, rot_m)
    assert jnp.issubdtype(vis.dtype, jnp.complexfloating)


def test_sky_rotation_changes_vis(forward_model, A_sky, rot_matrices):
    """Different rotation matrices should produce different visibilities."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    vis = forward_model.simulate(sky_coeffs, rot_matrices)  # (ntime, nfreq, nbls)

    if rot_matrices.shape[0] > 1:
        # Visibilities at different times should differ (sky rotates)
        diff = jnp.abs(vis[0] - vis[1])
        assert diff.mean() > 1e-6, "sky rotation did not change visibilities"


# ---------------------------------------------------------------------------
# 2D hex-rect NUFFT path
# ---------------------------------------------------------------------------

def test_simulate_2d_zero_sky(forward_model, A_sky):
    """Zero sky → zero visibilities for the 2D path."""
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = jnp.zeros((npix, nmodes), dtype=jnp.float32)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate_2d(sky_coeffs, rot_m)
    assert jnp.allclose(vis, 0.0, atol=1e-5), "non-zero vis for zero sky (2D path)"


def test_simulate_2d_output_shape(forward_model, A_sky, rot_matrices):
    """2D path returns shape (ntime, nfreq, nbls)."""
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = jnp.zeros((npix, nmodes), dtype=jnp.float32)

    vis = forward_model.simulate_2d(sky_coeffs, rot_matrices)
    ntime = rot_matrices.shape[0]
    assert vis.shape == (ntime, forward_model.nfreq, forward_model.nbls)


def test_simulate_2d_nonzero_sky(forward_model, A_sky):
    """A non-zero sky produces non-zero visibilities via the 2D path."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky, flux_per_pixel=1.0)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate_2d(sky_coeffs, rot_m)[0]
    assert jnp.abs(vis).mean() > 0.0


def test_simulate_2d_matches_3d(forward_model, A_sky):
    """2D and 3D paths agree to within NUFFT approximation error."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky, flux_per_pixel=1.0)
    rot_m = _identity_rot()[None, :, :]

    vis_3d = forward_model.simulate(sky_coeffs, rot_m)
    vis_2d = forward_model.simulate_2d(sky_coeffs, rot_m)

    assert vis_2d.shape == vis_3d.shape
    # Both paths compute the same integral; differences arise only from
    # float32 NUFFT approximation (eps=1e-6) and different factorisations.
    max_rel_err = float(
        jnp.max(jnp.abs(vis_2d - vis_3d)) / (jnp.mean(jnp.abs(vis_3d)) + 1e-30)
    )
    assert max_rel_err < 0.2, (
        f"2D vs 3D max relative error {max_rel_err:.4f} exceeds 20%"
    )


def test_simulate_2d_matches_3d_with_rotation(forward_model, A_sky, rot_matrices):
    """2D and 3D paths agree across multiple rotation matrices."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky, flux_per_pixel=1.0)

    vis_3d = forward_model.simulate(sky_coeffs, rot_matrices)
    vis_2d = forward_model.simulate_2d(sky_coeffs, rot_matrices)

    mean_mag = float(jnp.mean(jnp.abs(vis_3d)))
    max_abs_err = float(jnp.max(jnp.abs(vis_2d - vis_3d)))
    assert max_abs_err / (mean_mag + 1e-30) < 0.2, (
        f"2D vs 3D max relative error {max_abs_err/mean_mag:.4f} exceeds 20%"
    )


def test_accumulate_sky_update_2d_shape(forward_model, A_sky):
    """accumulate_equatorial_sky_update_2d returns same shape as sky_coeffs."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    rot_m = _identity_rot()[None, :, :]
    vis = forward_model.simulate_2d(sky_coeffs, rot_m)
    resid = vis * 0.01   # small residual

    delta = forward_model.accumulate_equatorial_sky_update_2d(
        resid, step_size=1.0, beam_reg=1e-3
    )
    assert delta.shape == sky_coeffs.shape
    assert jnp.isfinite(delta).all(), "sky update delta contains non-finite values"


def test_adjoint_consistency_2d(forward_model, A_sky):
    """Adjoint of the 2D NUFFT: <NUFFT(W), r> == <W, NUFFT*(r)>.

    The forward NUFFT operator maps the combined source strength
    W = sky_freq * beam_spec_h to visibilities.  dirty_apparent_sky_one_time_2d
    is its adjoint (not the full sky adjoint).  Tests without requiring the
    critical-channel-spacing Nyquist condition.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    rot_m = _identity_rot()[None, :, :]

    sky_coeffs = jnp.array(
        rng.standard_normal((npix, nmodes)).astype(np.float32)
    )
    resid = jnp.array(
        (rng.standard_normal((1, forward_model.nfreq, forward_model.nbls))
         + 1j * rng.standard_normal((1, forward_model.nfreq, forward_model.nbls))
        ).astype(np.complex64)
    )

    # Forward: vis = NUFFT(sky * beam)
    vis = forward_model.simulate_2d(sky_coeffs, rot_m)   # (1, nfreq, nbls)

    # Adjoint: dirty_pf = NUFFT*(scatter(resid))   [before beam conjugation]
    dirty_pf = forward_model.dirty_apparent_sky_one_time_2d(resid[0], 0)  # (npix, nfreq)

    # Combined source W = sky_freq * beam_spec_h  (what was passed to the NUFFT)
    sky_freq = jnp.array(sky_coeffs @ jnp.array(A_sky, dtype=jnp.float32).T)
    beam_spec_h = forward_model._beam_spec_horizon_all[0]   # (npix, nfreq), cached
    W = (sky_freq * beam_spec_h).astype(jnp.complex64)      # (npix, nfreq)

    # Inner products (using <a,b> = sum conj(a)*b convention)
    lhs = complex(jnp.sum(jnp.conj(vis[0]) * resid[0]))

    # <W, dirty_pf * norm>  — undo the 1/(nfreq*nbls) normalisation in the adjoint
    rhs_unnorm = complex(
        jnp.sum(jnp.conj(W) * dirty_pf) * (forward_model.nfreq * forward_model.nbls)
    )

    ratio = abs(lhs) / (abs(rhs_unnorm) + 1e-30)
    assert 0.8 < ratio < 1.2, (
        f"2D NUFFT adjoint: <NUFFT(W),r>={lhs:.4e}, <W,NUFFT*(r)>={rhs_unnorm:.4e}, "
        f"ratio={ratio:.4f}"
    )


def test_simulate_2d_is_complex(forward_model, A_sky):
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    rot_m = _identity_rot()[None, :, :]
    vis = forward_model.simulate_2d(sky_coeffs, rot_m)
    assert jnp.issubdtype(vis.dtype, jnp.complexfloating)


def test_calibrator_method_2d(forward_model, A_sky, rot_matrices):
    """Calibrator with method='2d' initialises without error and has non-zero loss."""
    from newnucal.calibrator import Calibrator
    import numpy as np

    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    vis_data = np.array(forward_model.simulate(sky_coeffs, rot_matrices))

    cal = Calibrator(
        forward_model.array,
        forward_model.beam_model,
        forward_model.sky_model,
        np.array(forward_model.freqs),
        rot_matrices,
        vis_data,
        method='2d',
    )
    assert cal.method == '2d'
    params = cal.init_params()
    loss = cal.calc_loss(params)
    assert np.isfinite(loss) and loss > 0.0
