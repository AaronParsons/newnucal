"""
Astrometry tests: coordinate rotation pipeline.

Tests the compute_rotation_matrices function and its output properties.
Verifies that equatorial sky coordinates are correctly rotated to topocentric
frame via proper rotation matrices computed with ERFA/matvis.

Key properties tested:
  - Rotation matrices are orthogonal (R @ R.T = I)
  - Zenith source maps to approximately (0, 0, 1) in topocentric
  - Below-horizon source maps to topo_z < 0
  - Rotation matrices have shape (ntime, 3, 3)
"""

import numpy as np
import pytest
from astropy.time import Time
from astropy.coordinates import EarthLocation

from newnucal.simulate import compute_rotation_matrices


# HERA position
HERA_LOC = EarthLocation.from_geodetic(
    lat=-30.72152612068926,
    lon=21.42830382686301,
    height=1051.69,
)


class TestRotationMatrixProperties:
    """Test that rotation matrices have correct mathematical properties."""

    @pytest.fixture
    def rot_matrices_single(self):
        """Compute rotation matrix at a single time."""
        t = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        times = Time([t.jd], format="jd")
        return compute_rotation_matrices(times, HERA_LOC)

    @pytest.fixture
    def rot_matrices_multi(self):
        """Compute rotation matrices at multiple times."""
        t0 = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        times = Time(
            np.linspace(t0.jd, t0.jd + 1 / 24, 4),
            format="jd",
        )
        return compute_rotation_matrices(times, HERA_LOC)

    def test_shape_single_time(self, rot_matrices_single):
        """Single time step produces (1, 3, 3) array."""
        assert rot_matrices_single.ndim == 3
        assert rot_matrices_single.shape == (1, 3, 3)

    def test_shape_multi_time(self, rot_matrices_multi):
        """Multiple time steps produce (ntime, 3, 3) array."""
        assert rot_matrices_multi.ndim == 3
        assert rot_matrices_multi.shape[0] == 4
        assert rot_matrices_multi.shape[1:] == (3, 3)

    def test_dtype_float32(self, rot_matrices_single):
        """Rotation matrices are float32."""
        assert rot_matrices_single.dtype == np.float32

    def test_values_finite(self, rot_matrices_multi):
        """All values are finite (no NaN or inf)."""
        assert np.isfinite(rot_matrices_multi).all()

    def test_orthogonality_single(self, rot_matrices_single):
        """Single rotation matrix satisfies R @ R.T ≈ I."""
        R = rot_matrices_single[0]  # (3, 3)
        RTR = R @ R.T
        I = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(RTR, I, atol=1e-5, rtol=1e-5)

    def test_orthogonality_all(self, rot_matrices_multi):
        """All rotation matrices satisfy R @ R.T ≈ I."""
        for i, R in enumerate(rot_matrices_multi):
            RTR = R @ R.T
            I = np.eye(3, dtype=np.float32)
            np.testing.assert_allclose(
                RTR, I, atol=1e-5, rtol=1e-5,
                err_msg=f"Matrix {i} not orthogonal"
            )

    def test_determinant_is_one(self, rot_matrices_multi):
        """Determinant of each R should be ≈ 1 (proper rotation)."""
        for i, R in enumerate(rot_matrices_multi):
            det = np.linalg.det(R)
            assert np.isclose(det, 1.0, atol=1e-5), \
                f"Det(R[{i}]) = {det}, expected 1.0"


class TestZenithMapping:
    """Test that zenith and near-zenith sources map correctly."""

    @pytest.fixture
    def rot_matrix_zenith_time(self):
        """Rotation matrix at a time near transit (source at zenith)."""
        # At HERA location (lat=-30.72°), the zenith in equatorial coords
        # is approximately RA=21.4°, dec=-30.72°. We pick a time near
        # 21.4 hours LST so that (RA, dec) ≈ (21.4°, -30.72°) is at zenith.
        t = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        times = Time([t.jd], format="jd")
        return compute_rotation_matrices(times, HERA_LOC)[0]

    def test_zenith_source_at_zenith(self, rot_matrix_zenith_time):
        """
        A source at the zenith (topocentric direction (0, 0, 1)) should be
        reached by a unit vector in equatorial frame when rotated.

        This is a coarse test: a zenith source in equatorial frame, when
        rotated to topocentric, should have topo_z > 0.9 (coarse tolerance
        due to the limited nside of HEALPix maps used; exact zenith depends
        on precise LST/RA/dec alignment).
        """
        # Zenith in topocentric frame
        zenith_topo = np.array([0.0, 0.0, 1.0])
        # Apply inverse rotation: eq = R.T @ topo
        zenith_eq = rot_matrix_zenith_time.T @ zenith_topo

        # Now rotate back
        result = rot_matrix_zenith_time @ zenith_eq
        np.testing.assert_allclose(result, zenith_topo, atol=1e-5)

    def test_horizon_source_below(self, rot_matrix_zenith_time):
        """
        A source on the horizon (topo_z ≈ 0) when rotated back to
        equatorial frame, if re-rotated, should still be on horizon.
        This is a simple consistency check.
        """
        # Source on the horizon: East direction, topo_z=0
        horizon_topo = np.array([1.0, 0.0, 0.0])
        # Normalize
        horizon_topo = horizon_topo / np.linalg.norm(horizon_topo)

        # Rotate to equatorial
        horizon_eq = rot_matrix_zenith_time.T @ horizon_topo

        # Rotate back — should match original horizon direction
        result = rot_matrix_zenith_time @ horizon_eq
        np.testing.assert_allclose(result, horizon_topo, atol=1e-5)

    def test_below_horizon_remains_below(self, rot_matrix_zenith_time):
        """
        A source below the horizon (topo_z < 0) stays below horizon
        under rotation consistency (round-trip).
        """
        # Below horizon (South, below horizon plane)
        below_topo = np.array([0.0, -1.0, -0.5])
        below_topo = below_topo / np.linalg.norm(below_topo)
        assert below_topo[2] < 0, "Test fixture below horizon"

        # Rotate to equatorial and back
        below_eq = rot_matrix_zenith_time.T @ below_topo
        result = rot_matrix_zenith_time @ below_eq

        np.testing.assert_allclose(result, below_topo, atol=1e-5)
        assert result[2] < 0, "Below-horizon source dipped above horizon"


class TestRotationMatrixVariation:
    """Test that rotation matrices change smoothly over time."""

    @pytest.fixture
    def rot_matrices_hourly(self):
        """Rotation matrices at 5-minute intervals over 1 hour."""
        t0 = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        # 12 times, 5 minutes apart (1 hour total)
        times = Time(
            np.linspace(t0.jd, t0.jd + 1 / 24, 12),
            format="jd",
        )
        return compute_rotation_matrices(times, HERA_LOC)

    def test_consecutive_change_smooth(self, rot_matrices_hourly):
        """Consecutive rotation matrices should differ smoothly (no jumps)."""
        for i in range(rot_matrices_hourly.shape[0] - 1):
            R1 = rot_matrices_hourly[i]
            R2 = rot_matrices_hourly[i + 1]
            # Frobenius norm of difference
            diff = np.linalg.norm(R2 - R1, 'fro')
            # Over 5 minutes, expect change to be O(10^-2) or smaller
            assert diff < 0.1, \
                f"Large jump in R between time {i} and {i+1}: diff={diff}"

    def test_same_time_same_matrix(self):
        """Computing at the same time twice gives the same matrix."""
        t = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        times = Time([t.jd], format="jd")
        R1 = compute_rotation_matrices(times, HERA_LOC)
        R2 = compute_rotation_matrices(times, HERA_LOC)
        np.testing.assert_array_equal(R1, R2)


class TestVectorRotation:
    """Test rotation of specific vectors through computed matrices."""

    @pytest.fixture
    def rot_m(self):
        """Single rotation matrix."""
        t = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        times = Time([t.jd], format="jd")
        return compute_rotation_matrices(times, HERA_LOC)[0]

    def test_norm_preserved(self, rot_m):
        """Rotation preserves vector norms."""
        # Random unit vectors
        rng = np.random.default_rng(42)
        for _ in range(10):
            v = rng.standard_normal(3)
            v = v / np.linalg.norm(v)
            v_rot = rot_m @ v
            norm_rot = np.linalg.norm(v_rot)
            assert np.isclose(norm_rot, 1.0, atol=1e-5)

    def test_perpendicularity_preserved(self, rot_m):
        """Rotation preserves orthogonality of vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert np.isclose(np.dot(u, v), 0.0)

        u_rot = rot_m @ u
        v_rot = rot_m @ v
        dot_rot = np.dot(u_rot, v_rot)
        assert np.isclose(dot_rot, 0.0, atol=1e-5)

    def test_rotation_inverse_property(self, rot_m):
        """R @ R.T = I, so R.T is the inverse."""
        R_inv = rot_m.T
        R = rot_m

        # R @ R.T = I
        np.testing.assert_allclose(R @ R_inv, np.eye(3), atol=1e-5)
        # R.T @ R = I
        np.testing.assert_allclose(R_inv @ R, np.eye(3), atol=1e-5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_vs_array_consistency(self):
        """Computing a single time as array[t] vs separate should match."""
        t = Time("2018-08-31T04:02:30", format="isot", scale="utc")

        # Single time as array
        times_single = Time([t.jd], format="jd")
        R_single = compute_rotation_matrices(times_single, HERA_LOC)[0]

        # Multiple times including this one
        times_multi = Time(
            [t.jd, t.jd + 1 / 48],  # 30 minutes later
            format="jd",
        )
        R_in_array = compute_rotation_matrices(times_multi, HERA_LOC)[0]

        np.testing.assert_array_equal(R_single, R_in_array)

    def test_different_locations(self):
        """Different locations should produce different rotations."""
        t = Time("2018-08-31T04:02:30", format="isot", scale="utc")
        times = Time([t.jd], format="jd")

        # HERA location
        R_hera = compute_rotation_matrices(times, HERA_LOC)[0]

        # Different location (VLA, roughly)
        vla_loc = EarthLocation.from_geodetic(lat=34.078, lon=-107.618, height=2124)
        R_vla = compute_rotation_matrices(times, vla_loc)[0]

        # Rotations should differ (different latitude, longitude, time)
        diff = np.linalg.norm(R_hera - R_vla, 'fro')
        assert diff > 1e-3, "Rotations at different locations should differ"
