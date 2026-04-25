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

from newnucal.basis import dpss_matrix, basis_project


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
    coeffs = basis_project(flat_flux, A_sky)
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

    vis = forward_model.simulate_3d(sky_coeffs, rot_m)  # (1, nfreq, nbls)
    assert jnp.allclose(vis, 0.0, atol=1e-5), "non-zero visibilities for zero sky"


def test_output_shape(forward_model, A_sky, rot_matrices):
    """Output shape matches (ntime, nfreq, nbls)."""
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = jnp.zeros((npix, nmodes), dtype=jnp.float32)

    vis = forward_model.simulate_3d(sky_coeffs, rot_matrices)
    ntime = rot_matrices.shape[0]
    assert vis.shape == (ntime, forward_model.nfreq, forward_model.nbls)


def test_nonzero_sky_gives_nonzero_vis(forward_model, A_sky):
    """A sky with positive flux should produce non-zero visibilities."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky, flux_per_pixel=1.0)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate_3d(sky_coeffs, rot_m)[0]  # (nfreq, nbls)
    assert jnp.abs(vis).mean() > 0.0


def test_gradient_computable(forward_model, A_sky):
    """JAX should be able to differentiate through the forward model."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    rot_m = _identity_rot()[None, :, :]

    def loss(sc):
        return jnp.sum(jnp.abs(forward_model.simulate_3d(sc, rot_m)) ** 2).real

    grad = jax.grad(loss)(sky_coeffs)
    assert grad.shape == sky_coeffs.shape
    assert jnp.isfinite(grad).all(), "gradient contains non-finite values"


def test_vis_is_complex(forward_model, A_sky):
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    rot_m = _identity_rot()[None, :, :]

    vis = forward_model.simulate_3d(sky_coeffs, rot_m)
    assert jnp.issubdtype(vis.dtype, jnp.complexfloating)


def test_sky_rotation_changes_vis(forward_model, A_sky, rot_matrices):
    """Different rotation matrices should produce different visibilities."""
    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    vis = forward_model.simulate_3d(sky_coeffs, rot_matrices)  # (ntime, nfreq, nbls)

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

    vis_3d = forward_model.simulate_3d(sky_coeffs, rot_m)
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

    vis_3d = forward_model.simulate_3d(sky_coeffs, rot_matrices)
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

    # <W, dirty_pf * norm>  — undo the 1/nbls normalisation in the adjoint
    rhs_unnorm = complex(
        jnp.sum(jnp.conj(W) * dirty_pf) * forward_model.nbls
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


def test_adjoint_normalization_2d_matches_3d(forward_model, A_sky):
    """2D and 3D dirty-sky amplitudes agree for the same residual.

    Both paths implement the same adjoint (F^H of the same forward), so
    dirty_apparent_sky_one_time_2d and dirty_apparent_sky_one_time should
    produce maps of the same magnitude at each pixel and frequency.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]
    rot_m = _identity_rot()[None, :, :]

    sky_coeffs = jnp.array(rng.standard_normal((npix, nmodes)).astype(np.float32))
    resid = jnp.array(
        (rng.standard_normal((forward_model.nfreq, forward_model.nbls))
         + 1j * rng.standard_normal((forward_model.nfreq, forward_model.nbls))
        ).astype(np.complex64)
    )

    dirty_2d = forward_model.dirty_apparent_sky_one_time_2d(resid, 0)   # (npix, nfreq)
    dirty_3d = forward_model.dirty_apparent_sky_one_time(resid, 0)       # (npix, nfreq)

    rms_2d = float(jnp.sqrt(jnp.mean(jnp.abs(dirty_2d) ** 2)))
    rms_3d = float(jnp.sqrt(jnp.mean(jnp.abs(dirty_3d) ** 2)))
    ratio = rms_2d / (rms_3d + 1e-30)
    assert 0.5 < ratio < 2.0, (
        f"2D/3D dirty-sky RMS ratio={ratio:.3f}; expected ~1. "
        f"rms_2d={rms_2d:.3e}, rms_3d={rms_3d:.3e}. "
        "Possible adjoint normalization mismatch between the two paths."
    )


def test_sky_update_2d_reduces_loss(forward_model, A_sky, rot_matrices):
    """One dirty-map sky update step should reduce the reconstruction loss.

    Constructs a perturbed sky, computes the residual, applies one 2D dirty
    step, and verifies the loss decreases.
    """
    import numpy as np

    rng = np.random.default_rng(11)
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]

    sky_true = jnp.array(
        (rng.standard_normal((npix, nmodes)) * 0.5).astype(np.float32)
    )
    vis_true = forward_model.simulate_2d(sky_true, rot_matrices)

    sky_pert = sky_true + jnp.array(
        (rng.standard_normal((npix, nmodes)) * 0.3 * float(jnp.std(sky_true))).astype(np.float32)
    )
    vis_pert = forward_model.simulate_2d(sky_pert, rot_matrices)

    resid = vis_true - vis_pert   # (ntime, nfreq, nbls)
    loss_before = float(jnp.sum(jnp.abs(resid) ** 2))

    delta = forward_model.accumulate_equatorial_sky_update_2d(
        resid, step_size=1.0, beam_reg=1e-3
    )
    sky_updated = (sky_pert + delta).astype(jnp.float32)
    vis_updated = forward_model.simulate_2d(sky_updated, rot_matrices)
    loss_after = float(jnp.sum(jnp.abs(vis_true - vis_updated) ** 2))

    assert loss_after < loss_before, (
        f"2D sky update increased loss: {loss_before:.4e} → {loss_after:.4e}. "
        "Adjoint or sign error in the 2D update path."
    )


def test_sky_update_2d_direction_matches_3d(forward_model, A_sky, rot_matrices):
    """2D and 3D accumulated sky updates point in the same direction.

    For the same residual, the sky update from the 2D adjoint and the 3D
    adjoint should have positive cosine similarity (same direction).
    """
    import numpy as np

    rng = np.random.default_rng(17)
    npix = forward_model.npix_sky
    nmodes = A_sky.shape[1]

    resid = jnp.array(
        (rng.standard_normal((rot_matrices.shape[0], forward_model.nfreq, forward_model.nbls))
         + 1j * rng.standard_normal((rot_matrices.shape[0], forward_model.nfreq, forward_model.nbls))
        ).astype(np.complex64)
    )

    delta_2d = forward_model.accumulate_equatorial_sky_update_2d(
        resid, step_size=1.0, beam_reg=1e-3
    )
    delta_3d = forward_model.accumulate_equatorial_sky_update_3d(
        resid, step_size=1.0, beam_reg=1e-3
    )

    cosine = float(
        jnp.sum(delta_2d * delta_3d) /
        (jnp.linalg.norm(delta_2d) * jnp.linalg.norm(delta_3d) + 1e-30)
    )
    assert cosine > 0.5, (
        f"2D and 3D sky updates point in different directions: cosine={cosine:.3f}. "
        "Normalization or sign mismatch in the 2D adjoint path."
    )


def test_combined_sky_and_beam_update_equivalence(forward_model, A_sky, A_beam, rot_matrices):
    """Verify combined update gives same results as separate sky+beam updates."""
    import numpy as np

    # Precompute time geometry
    forward_model.precompute_time_geometry(rot_matrices)

    rng = np.random.default_rng(42)
    npix_sky = forward_model.npix_sky
    nmodes_sky = A_sky.shape[1]
    nmodes_beam = A_beam.shape[1]

    # Generate test residuals
    resid = jnp.array(
        (rng.standard_normal((rot_matrices.shape[0], forward_model.nfreq, forward_model.nbls))
         + 1j * rng.standard_normal((rot_matrices.shape[0], forward_model.nfreq, forward_model.nbls))
        ).astype(np.complex64)
    )

    # Generate random sky coefficients
    sky_coeffs = jnp.array(
        rng.standard_normal((npix_sky, nmodes_sky)).astype(np.float32)
    )

    # Get separate updates
    sky_delta_sep, beam_delta_sep = (
        forward_model.accumulate_equatorial_sky_update_3d(resid, step_size=0.5, beam_reg=1e-3),
        forward_model.accumulate_beam_update_3d(sky_coeffs, resid, step_size=0.5, sky_reg=1e-3),
    )

    # Get combined updates
    sky_delta_comb, beam_delta_comb = forward_model.accumulate_sky_and_beam_update_3d(
        sky_coeffs, resid, sky_step_size=0.5, beam_step_size=0.5, beam_reg=1e-3, sky_reg=1e-3
    )

    # Check that results are close (within float32 tolerance)
    np.testing.assert_allclose(sky_delta_sep, sky_delta_comb, rtol=1e-4, atol=1e-6,
                               err_msg="Sky update differs between combined and separate paths")
    np.testing.assert_allclose(beam_delta_sep, beam_delta_comb, rtol=1e-4, atol=1e-6,
                               err_msg="Beam update differs between combined and separate paths")


def test_combined_sky_and_beam_update_2d_equivalence(forward_model, A_sky, A_beam, rot_matrices):
    """Verify 2D combined update gives same results as separate 2D sky+beam updates."""
    import numpy as np

    # Precompute time geometry
    forward_model.precompute_time_geometry(rot_matrices)

    rng = np.random.default_rng(43)
    npix_sky = forward_model.npix_sky
    nmodes_sky = A_sky.shape[1]
    nmodes_beam = A_beam.shape[1]

    # Generate test residuals
    resid = jnp.array(
        (rng.standard_normal((rot_matrices.shape[0], forward_model.nfreq, forward_model.nbls))
         + 1j * rng.standard_normal((rot_matrices.shape[0], forward_model.nfreq, forward_model.nbls))
        ).astype(np.complex64)
    )

    # Generate random sky coefficients
    sky_coeffs = jnp.array(
        rng.standard_normal((npix_sky, nmodes_sky)).astype(np.float32)
    )

    # Get separate updates (2D path)
    sky_delta_sep, beam_delta_sep = (
        forward_model.accumulate_equatorial_sky_update_2d(resid, step_size=0.5, beam_reg=1e-3),
        forward_model.accumulate_beam_update_2d(sky_coeffs, resid, step_size=0.5, sky_reg=1e-3),
    )

    # Get combined updates (2D path)
    sky_delta_comb, beam_delta_comb = forward_model.accumulate_sky_and_beam_update_2d(
        sky_coeffs, resid, sky_step_size=0.5, beam_step_size=0.5, beam_reg=1e-3, sky_reg=1e-3
    )

    # Check that results are close (within float32 tolerance)
    np.testing.assert_allclose(sky_delta_sep, sky_delta_comb, rtol=1e-4, atol=1e-6,
                               err_msg="2D Sky update differs between combined and separate paths")
    np.testing.assert_allclose(beam_delta_sep, beam_delta_comb, rtol=1e-4, atol=1e-6,
                               err_msg="2D Beam update differs between combined and separate paths")


def test_simulate_chunked_matches_unchunked(forward_model, rot_matrices, A_sky):
    """Chunked simulation should produce identical results to non-chunked."""
    fwd = forward_model
    sky_coeffs = _zenith_sky_coeffs(fwd.sky_model.nside, A_sky)

    # Simulate without chunking
    vis_no_chunk = fwd.simulate_3d(sky_coeffs, rot_matrices, t_chunk_size=rot_matrices.shape[0])

    # Simulate with chunking (smaller chunks)
    vis_chunked = fwd.simulate_3d(sky_coeffs, rot_matrices, t_chunk_size=1)

    # Results should match within float32 tolerance
    assert jnp.allclose(vis_no_chunk, vis_chunked, rtol=1e-5, atol=1e-8), (
        f"Chunked vs non-chunked mismatch: max diff = {float(jnp.max(jnp.abs(vis_no_chunk - vis_chunked)))}"
    )


def test_simulate_2d_chunked_matches_unchunked(forward_model, rot_matrices):
    """Chunked 2D simulation should produce identical results to non-chunked."""
    fwd = forward_model
    sky_coeffs = _zenith_sky_coeffs(fwd.sky_model.nside, fwd.A_sky)

    # Simulate without chunking
    vis_no_chunk = fwd.simulate_2d(sky_coeffs, rot_matrices, t_chunk_size=rot_matrices.shape[0])

    # Simulate with chunking (smaller chunks)
    vis_chunked = fwd.simulate_2d(sky_coeffs, rot_matrices, t_chunk_size=1)

    # Results should match within float32 tolerance
    assert jnp.allclose(vis_no_chunk, vis_chunked, rtol=1e-5, atol=1e-8), (
        f"Chunked vs non-chunked mismatch: max diff = {float(jnp.max(jnp.abs(vis_no_chunk - vis_chunked)))}"
    )


def test_calibrator_method_2d(forward_model, A_sky, rot_matrices):
    """Calibrator with method='2d' initialises without error and has non-zero loss."""
    from newnucal.calibrator import Calibrator
    import numpy as np

    sky_coeffs = _zenith_sky_coeffs(forward_model.sky_model.nside, A_sky)
    vis_data = np.array(forward_model.simulate_3d(sky_coeffs, rot_matrices))

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
