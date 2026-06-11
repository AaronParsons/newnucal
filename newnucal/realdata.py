"""Helpers for applying newnucal to real HERA uvh5 data products.

These utilities cover the plumbing between a (redundantly averaged,
LST-stacked) uvh5 file and a :class:`~newnucal.calibrator.Calibrator`:

- partial loading via ``hera_cal.io.HERAData`` (lazy import),
- snapping measured antenna positions to the exact 14.6/3 m hex lattice that
  fftvis gridding requires,
- building a :class:`~newnucal.array.HERAArray` whose baseline ordering
  matches the file's pair ordering,
- dead-channel bookkeeping, noise estimation, gain interpolation across
  fully-flagged channels, and a positivity projection for sky coefficients.
"""

from __future__ import annotations

import numpy as np

from .array import HERAArray
from .utils import DTYPE_R_NPY as DTYPE_R


HEX_SEP_M = 14.6   # HERA element separation; outriggers sit on the /3 lattice


def load_lst_stack_subset(path, time_idx, channels, nchan=None, pol_index=0):
    """Load a (times × channels × cross-baselines) subset of an LST stack.

    Parameters
    ----------
    path : str
        uvh5 file path.
    time_idx : array_like of int
        Indices into the unique-time axis.
    channels : int or array_like of int
        Channel indices to load.  An array selects arbitrary (e.g. uniformly
        decimated wide-band) channels — spectral-redundancy calibration wants
        bandwidth lever arm, not a contiguous sliver.  For backward
        compatibility, an int is treated as ``ch0`` with ``nchan`` channels.
    nchan : int, optional
        Only used when ``channels`` is an int (contiguous selection).
    pol_index : int
        Index into the file's polarization list.

    Returns
    -------
    dict with keys:
        data, flags, nsamples : (ntime, nfreq, nbls) arrays (crosses only)
        pairs : (nbls, 2) int array of (ant1, ant2)
        freqs : (nfreq,) Hz
        jd : (ntime,) selected julian dates
        pol : str
        autos : dict mapping (ant, ant) -> (ntime, nfreq) real autocorrelation
        antpos : dict ant -> ENU position (m) for antennas present in pairs
        location : astropy EarthLocation
        hd : the HERAData object (metadata access)
    """
    from hera_cal.io import HERAData
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    hd = HERAData(path)
    jd_all = np.unique(hd.times)
    time_idx = np.asarray(time_idx, dtype=int)
    jd = jd_all[time_idx]
    pol = hd.pols[pol_index]
    if np.isscalar(channels) or np.ndim(channels) == 0:
        channels = np.arange(int(channels), int(channels) + int(nchan))
    channels = np.asarray(channels, dtype=int)
    freqs = np.asarray(hd.freqs[channels], dtype=np.float64)

    data_c, flags_c, nsamples_c = hd.read(
        freq_chans=channels, times=jd, polarizations=[pol])

    pairs = np.array(sorted({(i, j) for i, j, p in data_c if i != j}))
    data = np.stack([data_c[(i, j, pol)] for i, j in pairs], axis=-1)
    flags = np.stack([flags_c[(i, j, pol)] for i, j in pairs], axis=-1)
    nsamples = np.stack([nsamples_c[(i, j, pol)] for i, j in pairs], axis=-1)
    autos = {(i, j): data_c[(i, j, pol)].real
             for i, j, p in data_c if i == j}

    try:
        location = hd.telescope.location
    except AttributeError:
        lat, lon, alt = hd.telescope_location_lat_lon_alt_degrees
        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

    used = np.unique(pairs)
    antpos = {int(a): np.asarray(hd.antpos[a], dtype=np.float64) for a in used}

    return dict(data=data, flags=flags, nsamples=nsamples, pairs=pairs,
                freqs=freqs, jd=jd, pol=pol, autos=autos, antpos=antpos,
                location=location, hd=hd)


def snap_to_hex_lattice(antpos, sep=HEX_SEP_M, max_err_m=0.02):
    """Snap measured ENU antenna positions onto the exact sep/3 hex lattice.

    HERA antennas (including outriggers) lie on a hex lattice of spacing
    ``sep/3`` to within a few mm.  This fits the lattice matrix by least
    squares and returns exactly-on-lattice positions, which is what fftvis
    strict griddability (and hence the newnucal NUFFT paths) requires.

    Assumes the array is close to the design orientation (East-aligned hex
    rows, as HERA is): the initial integer assignment uses the design lattice
    and tolerates only small rotations/scalings before rounding fails.

    Do NOT use ``redcal.reds_to_antpos`` for this: its integer embedding is
    not a linear image of the physical layout for outriggers.

    Parameters
    ----------
    antpos : dict ant -> ENU position (length >= 2)
    sep : float
        Core element separation in metres.
    max_err_m : float
        Raise if any snapped position is further than this from the
        measured position.

    Returns
    -------
    snapped : dict ant -> np.array([E, N, 0.0]) exactly on the lattice
    info : dict with 'lattice_matrix' (2, 2), 'max_err_m', 'rms_err_m'
    """
    ants = sorted(antpos)
    E = np.array([np.asarray(antpos[a])[:2] for a in ants], dtype=np.float64)

    M0 = np.array([[sep, sep / 2.0], [0.0, sep * np.sqrt(3) / 2.0]]) / 3.0
    n_int = np.round(np.linalg.solve(M0, (E - E[0]).T).T)
    A_ls = np.hstack([n_int, np.ones((len(E), 1))])
    sol, *_ = np.linalg.lstsq(A_ls, E, rcond=None)
    M_lat, t_off = sol[:2].T, sol[2]

    # one refinement pass with the fitted lattice
    n_int = np.round(np.linalg.solve(M_lat, (E - t_off).T).T)
    A_ls = np.hstack([n_int, np.ones((len(E), 1))])
    sol, *_ = np.linalg.lstsq(A_ls, E, rcond=None)
    M_lat, t_off = sol[:2].T, sol[2]

    fit = n_int @ M_lat.T + t_off
    err = np.linalg.norm(fit - E, axis=1)
    if err.max() > max_err_m:
        raise ValueError(
            f"lattice snap failed: max position error {err.max():.3f} m "
            f"exceeds {max_err_m} m — check antenna selection / sep")

    snapped = {a: np.array([*(n_int[i] @ M_lat.T), 0.0], dtype=DTYPE_R)
               for i, a in enumerate(ants)}
    info = dict(lattice_matrix=M_lat, max_err_m=float(err.max()),
                rms_err_m=float(np.sqrt((err ** 2).mean())))
    return snapped, info


def hera_array_from_file_pairs(snapped_antpos, pairs):
    """Build a HERAArray whose baseline ordering matches the file's pairs.

    HERAArray normally selects one representative per redundant group; for
    redundantly-averaged files the pair list IS the group list, so we
    override the baseline attributes with the file ordering, replicating
    HERAArray's own sign conventions
    (``bls = ants[i] - ants[j]``, ``bl_grid = round(gridded[j] - gridded[i])``).
    """
    import fftvis.core.antenna_gridding as _fg

    array = HERAArray(snapped_antpos)
    ok, gridded, basis_matrix = _fg.check_antpos_griddability(snapped_antpos)
    if not ok:
        raise ValueError("snapped antenna positions are not griddable")

    pairs = np.asarray(pairs, dtype=int)
    array.antpairs = pairs.copy()
    array.bls = np.array(
        [snapped_antpos[i] - snapped_antpos[j] for i, j in pairs], dtype=DTYPE_R)
    array.bl_grid = np.array(
        [np.round(gridded[j] - gridded[i]).astype(int)[:2] for i, j in pairs])
    array.n_modes = int(2 * np.max(np.abs(array.bl_grid)) + 1)

    if np.unique(array.bl_grid, axis=0).shape[0] != len(pairs):
        raise ValueError("file pairs do not map to unique grid separations")
    from .hexrect import hex_lattice_matrix
    A_lat = hex_lattice_matrix(array)
    if not np.allclose(A_lat @ array.bl_grid.T, array.bls[:, :2].T, atol=1e-6):
        raise ValueError("lattice matrix / bl_grid inconsistency")
    return array


def dead_channel_mask(vis_weights):
    """Channels where every (time, baseline) sample has zero weight.

    Parameters
    ----------
    vis_weights : (ntime, nfreq, nbls)

    Returns
    -------
    dead : (nfreq,) bool
    """
    return np.asarray(vis_weights).sum(axis=(0, 2)) == 0


def dead_channel_log_weights(vis_weights, floor=1e-6):
    """Log-space channel weights excluding dead channels from the objective."""
    dead = dead_channel_mask(vis_weights)
    w = np.where(dead, floor, 1.0)
    return np.log(w).astype(DTYPE_R)


def estimate_sigma_freqdiff(data, flags=None):
    """Per-channel noise sigma from frequency differencing.

    Adjacent fine channels see nearly the same sky, so the difference is
    sqrt(2) x noise.  Robust median over times and baselines; for complex z
    with per-component sigma s, median|z| = 1.1774 s.

    Returns (nfreq,) sigma per complex visibility component.
    """
    data = np.asarray(data)
    dvis = np.diff(data, axis=1)
    good = (data[:, 1:] != 0) & (data[:, :-1] != 0)
    if flags is not None:
        flags = np.asarray(flags, dtype=bool)
        good &= ~flags[:, 1:] & ~flags[:, :-1]
    absd = np.where(good, np.abs(dvis), np.nan)
    with np.errstate(all='ignore'):
        sigma = np.nanmedian(absd, axis=(0, 2)) / (1.1774 * np.sqrt(2.0))
    nfreq = data.shape[1]
    sigma = np.interp(np.arange(nfreq), np.arange(nfreq - 1) + 0.5, sigma)
    # fill any all-flagged channels with the band median
    bad = ~np.isfinite(sigma)
    if bad.any():
        sigma[bad] = np.nanmedian(sigma[~bad])
    return sigma.astype(DTYPE_R)


def interp_gains_over_dead_channels(gain_params, dead):
    """Interpolate gain parameters across dead (zero-weight) channels.

    At fully-flagged channels the closed-form gain solve is unconstrained and
    leaves gains at zero, which makes the gain-calibrated model jump to the
    raw forward-model amplitude there — poison for delay-spectrum
    diagnostics.  This interpolates the complex gain factor (and the phase
    gradient) linearly in frequency across dead channels.

    Parameters
    ----------
    gain_params : dict with 'log_amp' (nt, nf), 'phase' (nt, nf),
        'phi' (nt, 2, nf) — numpy or jax arrays.
    dead : (nfreq,) bool

    Returns
    -------
    dict with the same keys, numpy arrays, dead channels filled.
    """
    dead = np.asarray(dead, dtype=bool)
    log_amp = np.array(gain_params['log_amp'], dtype=np.float64)
    phase = np.array(gain_params['phase'], dtype=np.float64)
    phi = np.array(gain_params['phi'], dtype=np.float64)
    if not dead.any():
        return dict(log_amp=log_amp, phase=phase, phi=phi)
    nf = log_amp.shape[1]
    x, xg = np.arange(nf), np.arange(nf)[~dead]
    for t in range(log_amp.shape[0]):
        # interpolate the complex gain factor to avoid phase-wrap artifacts
        g = np.exp(log_amp[t] + 1j * phase[t])
        g_i = (np.interp(x, xg, g.real[~dead])
               + 1j * np.interp(x, xg, g.imag[~dead]))
        log_amp[t, dead] = np.log(np.maximum(np.abs(g_i[dead]), 1e-12))
        phase[t, dead] = np.angle(g_i[dead])
        for k in range(phi.shape[1]):
            phi[t, k, dead] = np.interp(x, xg, phi[t, k][~dead])[dead]
    return dict(log_amp=log_amp, phase=phase, phi=phi)


def project_positive(sky_coeffs, A_sky, floor=0.0, n_iter=10):
    """Project sky coefficients onto (approximately) non-negative spectra.

    Alternating projections (POCS) between the basis subspace and the
    non-negative orthant: reconstruct per-pixel spectra, clip at ``floor``,
    re-project onto the (orthonormal) basis, repeat.  Ten iterations leave
    negative excursions below ~0.3% of the spectrum peak; use as a cheap
    proximal step between fit stages to keep iterates physical.
    """
    coeffs = np.asarray(sky_coeffs)
    A = np.asarray(A_sky)
    x = coeffs
    for _ in range(int(n_iter)):
        spec = np.maximum(x @ A.T, floor)
        x = spec @ A
    return x.astype(coeffs.dtype)
