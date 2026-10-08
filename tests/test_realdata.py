"""Tests for newnucal.realdata helpers (no data file required)."""

import numpy as np
import pytest

from newnucal.realdata import (
    HEX_SEP_M,
    snap_to_hex_lattice,
    hera_array_from_file_pairs,
    dead_channel_mask,
    dead_channel_log_weights,
    estimate_sigma_freqdiff,
    interp_gains_over_dead_channels,
    project_positive,
)
from newnucal.basis import dpss_matrix


@pytest.fixture
def synthetic_lattice():
    """Antenna positions on a rotated sep/3 hex lattice with mm-level noise."""
    rng = np.random.default_rng(7)
    sep = HEX_SEP_M
    M = np.array([[sep, sep / 2.0], [0.0, sep * np.sqrt(3) / 2.0]]) / 3.0
    th = 0.01  # small rotation; snap assumes near-design orientation
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = R @ M
    # core antennas on multiples of 3 plus a few "outriggers" on the /3 grid
    n = [(3 * i, 3 * j) for i in range(5) for j in range(4)]
    n += [(17, 5), (-13, 22), (25, -8)]
    pos = {k: np.append(M @ np.array(v, float)
                        + rng.normal(0, 0.003, 2), 0.0)
           for k, v in enumerate(n)}
    return pos


class TestSnapToHexLattice:
    def test_recovers_lattice_to_mm(self, synthetic_lattice):
        snapped, info = snap_to_hex_lattice(synthetic_lattice)
        assert info['max_err_m'] < 0.02
        for a, p in snapped.items():
            assert np.linalg.norm(p[:2] - synthetic_lattice[a][:2]) < 0.02
            assert p[2] == 0.0

    def test_raises_on_off_lattice_antenna(self, synthetic_lattice):
        bad = dict(synthetic_lattice)
        bad[999] = np.array([1.234, 5.678, 0.0])  # nowhere near the lattice
        with pytest.raises(ValueError, match="lattice snap failed"):
            snap_to_hex_lattice(bad)


class TestHeraArrayFromFilePairs:
    def test_file_pair_ordering_and_consistency(self, synthetic_lattice):
        snapped, _ = snap_to_hex_lattice(synthetic_lattice)
        ants = sorted(snapped)
        pairs = np.array([(ants[i], ants[j])
                          for i in range(len(ants)) for j in range(i + 1, len(ants))])
        # drop redundant duplicates: keep first pair per separation
        seen, keep = set(), []
        for k, (i, j) in enumerate(pairs):
            key = tuple(np.round(snapped[j][:2] - snapped[i][:2], 3))
            if key not in seen:
                seen.add(key)
                keep.append(k)
        pairs = pairs[keep]
        arr = hera_array_from_file_pairs(snapped, pairs)
        assert arr.nbls == len(pairs)
        np.testing.assert_array_equal(arr.antpairs, pairs)
        # HERA/matvis convention: bls = x_j − x_i (see tests/test_convention.py)
        expect = np.array([snapped[j] - snapped[i] for i, j in pairs])
        np.testing.assert_allclose(arr.bls, expect, atol=1e-9)


class TestChannelHelpers:
    def test_dead_channel_mask_and_weights(self):
        w = np.ones((2, 5, 3))
        w[:, 2, :] = 0.0
        w[0, 4, :] = 0.0   # partially dead is not dead
        dead = dead_channel_mask(w)
        np.testing.assert_array_equal(dead, [False, False, True, False, False])
        lw = dead_channel_log_weights(w)
        assert lw[2] == np.log(1e-6)
        assert np.all(lw[[0, 1, 3, 4]] == 0.0)

    def test_sigma_freqdiff_recovers_noise(self):
        rng = np.random.default_rng(3)
        nt, nf, nb, s = 4, 64, 50, 0.7
        smooth = np.linspace(10, 11, nf)[None, :, None]
        noise = s * (rng.standard_normal((nt, nf, nb))
                     + 1j * rng.standard_normal((nt, nf, nb)))
        sig = estimate_sigma_freqdiff(smooth + noise)
        assert sig.shape == (nf,)
        np.testing.assert_allclose(sig, s, rtol=0.2)

    def test_interp_gains_over_dead_channels(self):
        nt, nf = 2, 8
        gp = dict(
            log_amp=np.tile(np.linspace(-1, 1, nf), (nt, 1)),
            phase=np.tile(np.linspace(-0.5, 0.5, nf), (nt, 1)),
            phi=np.tile(np.linspace(0, 1e-3, nf), (nt, 2, 1)),
        )
        dead = np.zeros(nf, bool)
        dead[[3, 4]] = True
        gp_bad = {k: np.array(v) for k, v in gp.items()}
        gp_bad['log_amp'][:, dead] = 0.0
        gp_bad['phase'][:, dead] = 0.0
        out = interp_gains_over_dead_channels(gp_bad, dead)
        # complex-factor interpolation differs slightly from linear-in-log
        np.testing.assert_allclose(out['log_amp'], gp['log_amp'], atol=0.1)
        np.testing.assert_allclose(out['phase'], gp['phase'], atol=0.1)
        np.testing.assert_allclose(out['phi'], gp['phi'], atol=1e-5)
        # live channels untouched
        np.testing.assert_array_equal(out['phase'][:, ~dead], gp_bad['phase'][:, ~dead])


class TestProjectPositive:
    def test_clips_negative_spectra(self):
        freqs = np.linspace(100e6, 120e6, 32)
        A = dpss_matrix(freqs, 200e-9)
        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal((20, A.shape[1]))
        out = project_positive(coeffs, A)
        spec = out @ A.T
        assert spec.min() > -0.01 * np.abs(spec).max()  # POCS leaves <1% leakage

    def test_positive_sky_unchanged(self):
        freqs = np.linspace(100e6, 120e6, 32)
        A = dpss_matrix(freqs, 200e-9)
        flux = np.outer(np.ones(10), np.linspace(5, 6, 32))
        coeffs = flux @ A
        out = project_positive(coeffs, A)
        np.testing.assert_allclose(out, coeffs, rtol=1e-4, atol=1e-6)
