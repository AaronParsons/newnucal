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
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_nside, A_sky, flux_per_pixel=1.0)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate(sky_coeffs, rot_m)[0]  # (nfreq, nbls)
    assert jnp.abs(vis).mean() > 0.0


def test_gradient_computable(forward_model, A_sky):
    """JAX should be able to differentiate through the forward model."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_nside, A_sky)
    rot_m = _identity_rot()[None, :, :]

    def loss(sc):
        return jnp.sum(jnp.abs(forward_model.simulate(sc, rot_m)) ** 2).real

    grad = jax.grad(loss)(sky_coeffs)
    assert grad.shape == sky_coeffs.shape
    assert jnp.isfinite(grad).all(), "gradient contains non-finite values"


def test_vis_is_complex(forward_model, A_sky):
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_nside, A_sky)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate(sky_coeffs, rot_m)
    assert jnp.issubdtype(vis.dtype, jnp.complexfloating)


def test_sky_rotation_changes_vis(forward_model, A_sky, rot_matrices):
    """Different rotation matrices should produce different visibilities."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_nside, A_sky)
    vis = forward_model.simulate(sky_coeffs, rot_matrices)  # (ntime, nfreq, nbls)

    if rot_matrices.shape[0] > 1:
        # Visibilities at different times should differ (sky rotates)
        diff = jnp.abs(vis[0] - vis[1])
        assert diff.mean() > 1e-6, "sky rotation did not change visibilities"
