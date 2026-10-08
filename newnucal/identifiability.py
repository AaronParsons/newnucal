"""
uv–frequency identifiability of the sky at sub-element resolution.

Mutual coupling perturbs each baseline's visibility kernel out to the array
scale, so predicting the coupled visibilities needs the sky Fourier transform
T̃(u, ν) at uv cells finer than the antenna lattice.  This module asks, before
any fitting, how well the *uncoupled* main-lobe measurements pin T̃ down at
those cells.  The handles are Earth rotation, which resolves u_E through
fringe rate, chromatic dilation (u = bν/c), and spectral smoothness of T̃ at
fixed u.  The derivation is in ``memos/memo-001-coupling-identifiability/memo-001.tex``.

Measurement model (flat sky, zenith drift scan, isolated-dish kernel)
---------------------------------------------------------------------
For a redundant group b, a frequency ν and a fringe-rate bin k::

    y(b, ν, k) = Σ_c T̃(u_c, ν) K_ν(bν/c − u_c) τ(u_c,E) 1[u_c,E ∈ bin k] + n

    T̃(u_c, ν)  = Σ_m a_m(u_c) A[ν, m]          (smooth at fixed u)

* ``K_ν`` is the uncoupled visibility kernel (aperture autocorrelation).  Its
  support has radius D in metres, i.e. D ν / c wavelengths.
* The sky drifts at ω cos δ in direction cosine, so a sky mode at u_E has
  fringe rate f = u_E ω cos δ.  An LST span ΔH resolves fringe rate into bins
  of width δu_E = 1 / (ΔH cos δ) wavelengths.
* ``τ(u_E) = sinc(u_E ω cos δ Δt)`` is the boxcar taper of an integration Δt.
  Bins beyond the Nyquist fringe rate of the sample spacing Δt_s,
  |u_E| > 1 / (2 ω cos δ Δt_s), alias and are counted as unmeasured.
* A zero-fringe-rate (FR0) filter removes the bins with |u_E| < f_c / (ω cos δ)
  for every baseline.  Those sky modes are absent from both data and
  (filtered) model, so they are *not needed*.  They are flagged, not counted
  as failures.

Measurements in different fringe bins are independent, so the Fisher matrix
is block diagonal over bins.  Each block holds the uv cells of one bin
column (optionally oversampled in E) times the spectral modes::

    P_k = I + snr² Σ_ν H_{k,ν} ⊗ A[ν]A[ν]ᵀ,   H_{k,ν} = Σ_b w_b κ_bν κ_bνᵀ

with a Gaussian prior a_m(u_c) ~ N(0, π_m) whose variance π_m depends only
on the spectral mode, normalised by π (the matrix above is shown for π = 1).
A decaying π_m encodes foreground spectral smoothness.  The reported
``var_ratio`` is the posterior/prior variance of each coefficient, and
``n_eff = Σ_m (1 − var_ratio_m)`` is the number of spectral degrees of
freedom at a cell that the data determine (0 … M).
``error_power_fraction = Σ_m post_m / Σ_m π_m`` is the fraction of the cell's
sky power left uncertain.  Coupling predicted from that cell can be
subtracted to a residual power of about this fraction, so its inverse is the
coupling suppression the sky knowledge supports.

Including both orientations ±b, with cells u and −u kept independent, is
equivalent to tying T̃(−u) = T̃(u)* for a real sky.  Each physical measurement
then informs each cell exactly once.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .utils import C

OMEGA_EARTH = 7.2921150e-5          # sidereal rotation rate, rad / s
HERA_LAT_DEG = -30.72152612068926
HERA_DISH_DIAMETER = 14.0           # m

STATUS_MEASURED = 0
STATUS_FR0 = 1                      # removed by the zero-fringe-rate filter
STATUS_ALIAS = 2                    # beyond the time-sampling Nyquist limit


# ----------------------------------------------------------------------
# Array, kernel and spectral-basis helpers
# ----------------------------------------------------------------------

def unique_baselines(antpos, decimals: int = 1):
    """Redundant baseline groups, one orientation per group.

    Parameters
    ----------
    antpos : dict[int, array_like] or array_like, (nant, >=2)
        Antenna positions (m), East and North first.
    decimals : int
        Rounding (in metres) used to identify redundant separations.

    Returns
    -------
    bls : ndarray, (nbls, 2)
        (East, North) group separations (m), oriented with E > 0, or E = 0 and
        N > 0.
    counts : ndarray of int, (nbls,)
        Number of antenna pairs in each group.
    """
    pos = antpos.values() if isinstance(antpos, dict) else antpos
    pos = np.asarray(list(pos), dtype=float)[:, :2]
    i, j = np.triu_indices(len(pos), k=1)
    d = np.round(pos[j] - pos[i], decimals) + 0.0
    flip = (d[:, 0] < 0) | ((d[:, 0] == 0) & (d[:, 1] < 0))
    d[flip] *= -1
    bls, counts = np.unique(d + 0.0, axis=0, return_counts=True)
    return bls, counts


def disk_aperture_kernel(beta, diameter: float = HERA_DISH_DIAMETER):
    """Autocorrelation of a uniformly illuminated disk, peak-normalised to 1.

    Parameters
    ----------
    beta : array_like
        Separation magnitude in metres.
    diameter : float
        Aperture diameter D (m).  The kernel vanishes for |beta| >= D.
    """
    s = np.clip(np.abs(np.asarray(beta, dtype=float)) / diameter, 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(s) - s * np.sqrt(1.0 - s * s))


def spectral_basis(freqs, n_modes: int, kind: str = "dpss"):
    """Orthonormal smooth spectral basis A (nfreq, n_modes) at fixed u.

    ``kind='dpss'`` uses the first ``n_modes`` Slepian sequences with
    time-half-bandwidth NW = n_modes / 2.  ``kind='legendre'`` uses
    orthonormalised Legendre polynomials in frequency.
    """
    freqs = np.asarray(freqs, dtype=float)
    nf = freqs.size
    if not 1 <= n_modes <= nf:
        raise ValueError(f"n_modes must be in [1, {nf}], got {n_modes}")
    if kind == "dpss":
        from scipy.signal.windows import dpss
        if n_modes == 1 and nf == 1:
            return np.ones((1, 1))
        A = dpss(nf, max(n_modes / 2.0, 0.5), Kmax=n_modes).T
        A = A.reshape(nf, n_modes)
    elif kind == "legendre":
        x = np.linspace(-1.0, 1.0, nf) if nf > 1 else np.zeros(1)
        V = np.polynomial.legendre.legvander(x, n_modes - 1)
        A, _ = np.linalg.qr(V)
    else:
        raise ValueError(f"kind must be 'dpss' or 'legendre', got {kind!r}")
    # Fix signs so the leading mode is positive (cosmetic, deterministic).
    A = A * np.sign(A.sum(axis=0, keepdims=True) + 1e-300)
    return A


# ----------------------------------------------------------------------
# Observation description
# ----------------------------------------------------------------------

@dataclass
class ObservationSpec:
    """Observing parameters that set uv–ν sampling.

    Parameters
    ----------
    freqs : array_like, (nfreq,)
        Channel centres (Hz).
    lst_span_hours : float
        Contiguous LST range of the data (sidereal hours).
    integration_s : float
        Integration (LST-bin) length in seconds.  Sets the boxcar taper.
    sample_interval_s : float, optional
        Spacing between LST samples.  Sets the fringe-rate Nyquist limit.
        The default is ``integration_s`` (contiguous, non-overlapping bins).
        Decimated data have sample_interval_s > integration_s.
    fr0_halfwidth_hz : float
        Half-width of the zero-fringe-rate notch (Hz); 0 means no filter.
    latitude_deg : float
        Array latitude.  The drift-scan field centre is at δ = latitude.
    dish_diameter : float
        Aperture diameter D (m) of the isolated element.
    """

    freqs: np.ndarray
    lst_span_hours: float
    integration_s: float
    fr0_halfwidth_hz: float = 0.0
    latitude_deg: float = HERA_LAT_DEG
    dish_diameter: float = HERA_DISH_DIAMETER
    sample_interval_s: float | None = None

    def __post_init__(self):
        self.freqs = np.atleast_1d(np.asarray(self.freqs, dtype=float))
        if self.sample_interval_s is None:
            self.sample_interval_s = self.integration_s

    @property
    def cos_dec(self) -> float:
        return float(np.abs(np.cos(np.deg2rad(self.latitude_deg))))

    @property
    def hour_angle_span(self) -> float:
        """LST span in radians."""
        return self.lst_span_hours * np.pi / 12.0

    @property
    def fringe_rate_per_u(self) -> float:
        """Fringe rate (Hz) per wavelength of East–West sky wavenumber."""
        return OMEGA_EARTH * self.cos_dec

    @property
    def fringe_resolution_u(self) -> float:
        """E–W sky-wavenumber resolution δu_E = 1 / (ΔH cos δ) (wavelengths)."""
        return 1.0 / (self.hour_angle_span * self.cos_dec)

    @property
    def u_alias(self) -> float:
        """|u_E| above which fringe rates exceed the time-sampling Nyquist."""
        return 1.0 / (2.0 * self.sample_interval_s * self.fringe_rate_per_u)

    @property
    def u_fr0(self) -> float:
        """|u_E| below which the zero-fringe-rate filter removes the data."""
        return self.fr0_halfwidth_hz / self.fringe_rate_per_u

    def time_taper(self, u_e):
        """Amplitude response of boxcar averaging over one integration."""
        return np.sinc(np.asarray(u_e) * self.fringe_rate_per_u * self.integration_s)

    def bin_status(self, u_e: float) -> int:
        if abs(u_e) < self.u_fr0:
            return STATUS_FR0
        if abs(u_e) > self.u_alias:
            return STATUS_ALIAS
        return STATUS_MEASURED


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class IdentifiabilityResult:
    """Per-cell identifiability on the sky uv grid (wavelength units).

    Arrays are indexed ``[iE, iN]`` (plus a trailing spectral-mode axis for
    ``var_ratio``).
    """

    u_e: np.ndarray              # (nE,) cell centres, East (wavelengths)
    v_n: np.ndarray              # (nN,) cell centres, North (wavelengths)
    freqs: np.ndarray            # (nfreq,) Hz
    var_ratio: np.ndarray        # (nE, nN, M) posterior / prior variance
    n_chan: np.ndarray           # (nE, nN) channels with kernel coverage
    nu_lo: np.ndarray            # (nE, nN) lowest covering frequency (nan if none)
    nu_hi: np.ndarray            # (nE, nN) highest covering frequency
    status: np.ndarray           # (nE,) STATUS_* of the column's fringe bin
    prior_var: np.ndarray = None  # (M,) prior variance per spectral mode
    params: dict = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return self.var_ratio.shape[-1]

    @property
    def n_eff(self) -> np.ndarray:
        """Spectral degrees of freedom determined per cell (0 … M)."""
        return np.sum(1.0 - self.var_ratio, axis=-1)

    @property
    def error_power_fraction(self) -> np.ndarray:
        """Posterior sky-error power over prior sky power, per cell (0 … 1)."""
        pv = np.ones(self.n_modes) if self.prior_var is None else self.prior_var
        return (self.var_ratio @ pv) / pv.sum()

    @property
    def band_fraction(self) -> np.ndarray:
        """Fraction of the band spanned by the covering channels."""
        span = self.freqs.max() - self.freqs.min()
        if span <= 0:
            return np.where(self.n_chan > 0, 1.0, 0.0)
        return np.nan_to_num((self.nu_hi - self.nu_lo) / span, nan=0.0)

    @property
    def cell_size(self):
        """(δu_E, δv_N) cell size in wavelengths."""
        du = self.u_e[1] - self.u_e[0] if self.u_e.size > 1 else np.nan
        dv = self.v_n[1] - self.v_n[0] if self.v_n.size > 1 else np.nan
        return float(du), float(dv)


# ----------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------

class _Grid:
    """uv grid, fringe-bin columns and per-frequency footprint lookup."""

    def __init__(self, bls, weights, spec, e_oversample, dv, u_max,
                 include_conjugates, kernel):
        self.spec = spec
        self.kernel = disk_aperture_kernel if kernel is None else kernel
        self.freqs = spec.freqs
        bls = np.asarray(bls, dtype=float)[:, :2]
        w = np.ones(len(bls)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(bls),):
            raise ValueError("weights must have shape (nbls,)")
        self.n_groups = len(bls)
        if include_conjugates:
            bls = np.vstack([bls, -bls])
            w = np.concatenate([w, w])
        self.bls, self.w = bls, w

        self.D = spec.dish_diameter
        self.du_bin = spec.fringe_resolution_u
        self.nsub = int(e_oversample)
        if self.nsub < 1:
            raise ValueError("e_oversample must be >= 1")
        dE = self.du_bin / self.nsub
        self.dv = dE if dv is None else float(dv)
        if u_max is None:
            bmax = float(np.max(np.linalg.norm(bls, axis=1))) if len(bls) else 0.0
            u_max = (bmax + self.D) * self.freqs.max() / C
        self.u_max = u_max

        nk = int(np.ceil(u_max / self.du_bin))
        self.bin_centres = np.arange(-nk, nk + 1) * self.du_bin
        self.sub_off = (np.arange(self.nsub) - (self.nsub - 1) / 2.0) * dE
        self.u_e = (self.bin_centres[:, None] + self.sub_off[None, :]).ravel()
        self.nr = int(np.ceil(u_max / self.dv))
        self.v_n = np.arange(-self.nr, self.nr + 1) * self.dv
        self.R = self.v_n.size
        self.nc = self.nsub * self.R                 # cells per bin column
        self.status = np.repeat(
            [spec.bin_status(uk) for uk in self.bin_centres], self.nsub)

        # Per-frequency footprint centres, sorted by E for fast strip selection.
        self._centres, self._order, self._sorted_e = [], [], []
        for nu in self.freqs:
            w0 = bls * nu / C
            o = np.argsort(w0[:, 0], kind="stable")
            self._centres.append(w0)
            self._order.append(o)
            self._sorted_e.append(w0[o, 0])

    def cols(self, k):
        return slice(k * self.nsub, (k + 1) * self.nsub)

    def footprints(self, k):
        """Yield (f, sel, ww, cell, Kf) for every channel touching bin k.

        ``cell`` (nsel, ntouch) indexes the bin's cells as s * R + r, and ``Kf``
        holds κ = K τ at those cells (zero where a footprint leaves the grid).
        """
        uk = self.bin_centres[k]
        u_sub = uk + self.sub_off
        tau = self.spec.time_taper(u_sub)
        sub_idx = np.arange(self.nsub)
        for f, nu in enumerate(self.freqs):
            rho = self.D * nu / C
            lo = np.searchsorted(self._sorted_e[f], uk - self.du_bin / 2 - rho, "left")
            hi = np.searchsorted(self._sorted_e[f], uk + self.du_bin / 2 + rho, "right")
            if hi <= lo:
                continue
            sel = self._order[f][lo:hi]
            wE, wN = self._centres[f][sel, 0], self._centres[f][sel, 1]
            nspan = int(np.ceil(rho / self.dv)) + 1
            offs = np.arange(-nspan, nspan + 1)
            rows = np.round(wN / self.dv).astype(int)[:, None] + self.nr + offs[None, :]
            valid = (rows >= 0) & (rows < self.R)
            rows = np.clip(rows, 0, self.R - 1)
            dN = self.v_n[rows] - wN[:, None]                         # (nsel, nO)
            dEm = u_sub[None, :] - wE[:, None]                        # (nsel, nsub)
            rad_m = np.hypot(dEm[:, :, None], dN[:, None, :]) * (C / nu)
            Kv = self.kernel(rad_m, self.D) * tau[None, :, None] * valid[:, None, :]
            cell = sub_idx[None, :, None] * self.R + rows[:, None, :]
            cell = np.broadcast_to(cell, Kv.shape).reshape(len(sel), -1)
            yield f, sel, self.w[sel], cell, Kv.reshape(len(sel), -1)

    def params(self, spec, **extra):
        return dict(
            e_oversample=self.nsub, dv=self.dv, u_max=self.u_max,
            fringe_resolution_u=self.du_bin, u_alias=spec.u_alias,
            u_fr0=spec.u_fr0, lst_span_hours=spec.lst_span_hours,
            integration_s=spec.integration_s,
            sample_interval_s=spec.sample_interval_s, dish_diameter=self.D,
            n_groups=self.n_groups, **extra)


def _check_basis(spec, A_sky, prior_var):
    A = np.asarray(A_sky, dtype=float).reshape(spec.freqs.size, -1)
    M = A.shape[1]
    pv = np.ones(M) if prior_var is None else np.asarray(prior_var, dtype=float)
    if pv.shape != (M,) or np.any(pv <= 0):
        raise ValueError("prior_var must be positive with shape (M,)")
    return A, M, pv


def _whitened_precision(H, A, pv, snr):
    """Posterior precision over the active cells, with the prior whitened to I."""
    act = np.nonzero(np.einsum("fii->i", H) > 0)[0]
    if act.size == 0:
        return act, None
    na, M = act.size, A.shape[1]
    Ha = H[:, act][:, :, act]
    P = np.einsum("fij,fa,fb->iajb", Ha, A, A).reshape(na * M, na * M)
    sq = np.tile(np.sqrt(pv), na)
    P *= snr ** 2 * sq[:, None] * sq[None, :]
    P[np.diag_indices_from(P)] += 1.0
    return act, cholesky(P, lower=True, check_finite=False)


def analyze(
    bls,
    spec: ObservationSpec,
    A_sky,
    weights=None,
    snr: float = 10.0,
    e_oversample: int = 1,
    dv: float | None = None,
    u_max: float | None = None,
    coverage_threshold: float = 0.1,
    include_conjugates: bool = True,
    kernel=None,
    prior_var=None,
) -> IdentifiabilityResult:
    """Posterior identifiability of T̃(u, ν) on a sub-element uv grid.

    Parameters
    ----------
    bls : array_like, (nbls, >=2)
        Redundant-group separations (m), one orientation per group.
    spec : ObservationSpec
    A_sky : array_like, (nfreq, M)
        Spectral basis for T̃ at fixed u (orthonormal columns recommended).
    weights : array_like, (nbls,), optional
        Inverse noise variance per group relative to a single baseline.
        This is the redundancy count for redundantly averaged data.  The
        default is 1.
    snr : float
        Per-measurement SNR of a unit-amplitude sky cell at kernel peak, for
        weight 1.
    e_oversample : int
        Number of E sub-columns per fringe-rate bin.  Values > 1 probe E
        structure finer than Earth rotation resolves, using chromatic
        dilation alone.
    dv : float, optional
        North cell size in wavelengths.  The default is the E cell size
        (square cells).
    u_max : float, optional
        Grid half-extent in wavelengths.  The default is
        (|b|_max + D) ν_max / c.
    coverage_threshold : float
        Kernel power Σ_b K² above which a channel counts as covering a cell
        in ``n_chan``.
    include_conjugates : bool
        Add −b for every group (see the module docstring).
    kernel : callable, optional
        ``kernel(beta_m, diameter) -> K``.  The default is
        :func:`disk_aperture_kernel`.
    prior_var : array_like, (M,), optional
        Prior variance of each spectral mode, relative to a unit cell.  The
        default is 1 for every mode (spectrally agnostic).

    Returns
    -------
    IdentifiabilityResult
    """
    A, M, pv = _check_basis(spec, A_sky, prior_var)
    g = _Grid(bls, weights, spec, e_oversample, dv, u_max, include_conjugates, kernel)
    freqs, nfreq, nc, R, nsub = g.freqs, g.freqs.size, g.nc, g.R, g.nsub

    nE = g.u_e.size
    var_ratio = np.ones((nE, R, M))
    n_chan = np.zeros((nE, R), dtype=int)
    nu_lo = np.full((nE, R), np.nan)
    nu_hi = np.full((nE, R), np.nan)

    for k in range(g.bin_centres.size):
        cols = g.cols(k)
        if g.status[k * nsub] != STATUS_MEASURED:
            continue
        H = np.zeros((nfreq, nc, nc))
        cov = np.zeros((nfreq, nc))
        for f, _, ww, cell, Kf in g.footprints(k):
            pair = (cell[:, :, None] * nc + cell[:, None, :]).ravel()
            val = (ww[:, None, None] * Kf[:, :, None] * Kf[:, None, :]).ravel()
            H[f] = np.bincount(pair, weights=val, minlength=nc * nc).reshape(nc, nc)
            cov[f] = np.bincount(cell.ravel(), weights=(Kf ** 2).ravel(), minlength=nc)

        # Coverage diagnostics (no inversion needed).
        covered = cov > coverage_threshold                            # (nfreq, nc)
        nch = covered.sum(axis=0)
        any_cov = nch > 0
        lo_nu = np.where(any_cov, freqs[np.argmax(covered, axis=0)], np.nan)
        hi_nu = np.where(
            any_cov, freqs[nfreq - 1 - np.argmax(covered[::-1], axis=0)], np.nan
        )
        n_chan[cols] = nch.reshape(nsub, R)
        nu_lo[cols] = lo_nu.reshape(nsub, R)
        nu_hi[cols] = hi_nu.reshape(nsub, R)

        # Posterior over the cells this bin's data touch.
        act, L = _whitened_precision(H, A, pv, snr)
        if L is None:
            continue
        Linv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)
        block = np.ones((nc, M))
        block[act] = np.sum(Linv ** 2, axis=0).reshape(act.size, M)  # post / prior
        var_ratio[cols] = block.reshape(nsub, R, M)

    params = g.params(spec, snr=snr, coverage_threshold=coverage_threshold,
                      include_conjugates=include_conjugates)
    return IdentifiabilityResult(
        u_e=g.u_e, v_n=g.v_n, freqs=freqs.copy(), var_ratio=var_ratio,
        n_chan=n_chan, nu_lo=nu_lo, nu_hi=nu_hi, status=g.status.copy(),
        prior_var=pv, params=params,
    )


# ----------------------------------------------------------------------
# Simulated recovery (posterior mean and samples)
# ----------------------------------------------------------------------

@dataclass
class GridSky:
    """A uv cell grid with a spectral basis; it evaluates gridded skies."""

    u_e: np.ndarray
    v_n: np.ndarray
    freqs: np.ndarray
    A: np.ndarray                # (nfreq, M) spectral basis
    status: np.ndarray           # (nE,) STATUS_* per column
    params: dict = field(default_factory=dict)

    def sky(self, coeffs):
        """T̃(u_c, ν) = Σ_m a_cm A_νm, shape (nE, nN, nfreq)."""
        return np.einsum("ijm,fm->ijf", coeffs, self.A)

    @property
    def cell_size(self):
        du = self.u_e[1] - self.u_e[0] if self.u_e.size > 1 else np.nan
        dv = self.v_n[1] - self.v_n[0] if self.v_n.size > 1 else np.nan
        return float(du), float(dv)


@dataclass
class RecoveryResult(GridSky):
    """A simulated sky, its noisy measurements' posterior, and samples.

    The posterior lives on this (fit) grid.  The true sky ``a_true`` lives on
    ``truth``, which is the fit grid itself unless a finer truth grid was
    requested.  Coefficient arrays are complex with shape (nE, nN, M);
    ``samples`` has a leading sample axis.  Cells outside measured fringe
    bins keep the prior (posterior mean 0, std sqrt(π_m)).
    """

    a_true: np.ndarray = None
    a_hat: np.ndarray = None
    a_std: np.ndarray = None     # posterior rms per coefficient (real)
    samples: np.ndarray | None = None
    prior_var: np.ndarray = None
    truth: GridSky = None


def _complex_normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def recover(
    bls,
    spec: ObservationSpec,
    A_sky,
    weights=None,
    snr: float = 10.0,
    e_oversample: int = 1,
    dv: float | None = None,
    u_max: float | None = None,
    include_conjugates: bool = True,
    kernel=None,
    prior_var=None,
    a_true=None,
    n_samples: int = 0,
    seed: int | None = 0,
    truth_e_oversample: int | None = None,
    truth_dv: float | None = None,
) -> RecoveryResult:
    """Simulate measurements of a sky realisation and compute its posterior.

    This uses the same measurement model, prior and fringe-bin block structure
    as :func:`analyze`.  The posterior variances therefore agree with
    ``analyze(...).var_ratio``, but here the posterior mean is computed from
    noisy simulated data, so recovered and true skies can be compared
    directly.

    Parameters
    ----------
    a_true : complex array, optional
        True coefficients on the truth grid.  The default draws
        a_cm ~ CN(0, π_m · area_truth / area_fit), i.e. a white sky whose
        coarse-cell sums have the fit grid's prior variance.
    truth_e_oversample, truth_dv : optional
        Draw the truth on a finer grid than the fit (same fringe bins, more E
        sub-columns and/or finer rows).  This tests how well a model at the
        fit resolution handles sky structure it cannot represent.  The
        default is the fit grid.
    n_samples : int
        Number of posterior samples to draw.
    seed : int, optional
        RNG seed for the sky, the noise and the samples.

    Other parameters are as in :func:`analyze`.
    """
    A, M, pv = _check_basis(spec, A_sky, prior_var)
    g = _Grid(bls, weights, spec, e_oversample, dv, u_max, include_conjugates, kernel)
    separate = truth_e_oversample is not None or truth_dv is not None
    if separate:
        gt = _Grid(bls, weights, spec,
                   g.nsub if truth_e_oversample is None else truth_e_oversample,
                   truth_dv, g.u_max, include_conjugates, kernel)
    else:
        gt = g
    rng = np.random.default_rng(seed)
    nfreq, nc, R, nsub = g.freqs.size, g.nc, g.R, g.nsub
    nE = g.u_e.size
    shape_t = (gt.u_e.size, gt.R, M)
    area_ratio = (gt.du_bin / gt.nsub * gt.dv) / (g.du_bin / g.nsub * g.dv)
    if a_true is None:
        a_true = _complex_normal(rng, shape_t) * np.sqrt(pv * area_ratio)
    a_true = np.asarray(a_true, dtype=complex)
    if a_true.shape != shape_t:
        raise ValueError(f"a_true must have shape {shape_t}")

    a_hat = np.zeros((nE, R, M), dtype=complex)
    a_std = np.broadcast_to(np.sqrt(pv), (nE, R, M)).copy()
    samples = None
    if n_samples:
        samples = _complex_normal(rng, (n_samples, nE, R, M)) * np.sqrt(pv)

    for k in range(g.bin_centres.size):
        cols = g.cols(k)
        if g.status[k * nsub] != STATUS_MEASURED:
            continue
        at_bin = a_true[gt.cols(k)].reshape(gt.nc, M)
        H = np.zeros((nfreq, nc, nc))
        rhs = np.zeros((nc, M), dtype=complex)
        for (f, sel, ww, cell, Kf), (ft, selt, _, cellt, Kft) in zip(
                g.footprints(k), gt.footprints(k)):
            assert f == ft and np.array_equal(sel, selt)
            y = np.sum(Kft * (at_bin[cellt] @ A[f]), axis=1)
            y = y + _complex_normal(rng, y.shape) / (snr * np.sqrt(ww))
            pair = (cell[:, :, None] * nc + cell[:, None, :]).ravel()
            val = (ww[:, None, None] * Kf[:, :, None] * Kf[:, None, :]).ravel()
            H[f] = np.bincount(pair, weights=val, minlength=nc * nc).reshape(nc, nc)
            z = (ww[:, None] * Kf * y[:, None]).ravel()
            back = (np.bincount(cell.ravel(), weights=z.real, minlength=nc)
                    + 1j * np.bincount(cell.ravel(), weights=z.imag, minlength=nc))
            rhs += back[:, None] * A[f][None, :]

        act, L = _whitened_precision(H, A, pv, snr)
        if L is None:
            continue
        na = act.size
        sq = np.tile(np.sqrt(pv), na)
        b = snr ** 2 * sq * rhs[act].ravel()
        x = solve_triangular(L, solve_triangular(L, b, lower=True, check_finite=False),
                             lower=True, trans="T", check_finite=False)
        Linv = solve_triangular(L, np.eye(na * M), lower=True, check_finite=False)
        post = np.sum(Linv ** 2, axis=0)

        blk_hat = np.zeros((nc, M), dtype=complex)
        blk_hat[act] = (x * sq).reshape(na, M)
        a_hat[cols] = blk_hat.reshape(nsub, R, M)
        blk_std = np.broadcast_to(np.sqrt(pv), (nc, M)).copy()
        blk_std[act] = np.sqrt(post * sq ** 2).reshape(na, M)
        a_std[cols] = blk_std.reshape(nsub, R, M)
        if n_samples:
            zs = _complex_normal(rng, (na * M, n_samples))
            xs = x[:, None] + Linv.T @ zs                    # ~ N(x, P^-1)
            blk = samples[:, cols].reshape(n_samples, nc, M)
            blk[:, act] = (xs * sq[:, None]).T.reshape(n_samples, na, M)
            samples[:, cols] = blk.reshape(n_samples, nsub, R, M)

    params = g.params(spec, snr=snr, include_conjugates=include_conjugates,
                      seed=seed)
    truth = GridSky(u_e=gt.u_e, v_n=gt.v_n, freqs=gt.freqs.copy(), A=A,
                    status=gt.status.copy(),
                    params=gt.params(spec, area_ratio=area_ratio))
    return RecoveryResult(
        u_e=g.u_e, v_n=g.v_n, freqs=g.freqs.copy(), A=A, status=g.status.copy(),
        params=params, a_true=a_true, a_hat=a_hat, a_std=a_std, samples=samples,
        prior_var=pv, truth=truth,
    )


def coupling_visibility(rec: GridSky, coeffs, b, delta_kernel,
                        spec: ObservationSpec, n_sub: int = 5,
                        support_m: float | None = None):
    """Coupled-visibility term of baseline ``b`` predicted from a gridded sky.

    For each fringe bin k (one column of cells) and channel ν::

        δV(b, ν, k) = Σ_{c ∈ k} T̃_c(ν) τ(u_c,E) ⟨δK_ν(bν/c − u)⟩_cell

    ⟨·⟩_cell averages δK over an ``n_sub`` × ``n_sub`` grid inside each cell,
    because coupling kernels carry sub-cell structure while the sky is one
    amplitude per cell.  Bins that are not measured (FR0, aliased) are zero,
    matching filtered data.

    Parameters
    ----------
    rec : GridSky
        Supplies the grid, basis and bin status: a :class:`RecoveryResult`
        for posterior coefficients, or ``rec.truth`` for ``rec.a_true``.
    coeffs : complex array (nE, nN, M)
        Sky coefficients, e.g. ``rec.a_true``, ``rec.a_hat`` or a sample.
    b : array_like (2,)
        Baseline (m).
    delta_kernel : callable
        ``delta_kernel(beta_e_m, beta_n_m, nu) -> complex`` kernel in metres.
    support_m : float, optional
        Only cells with |b − u c/ν| <= support_m + cell diagonal are summed.

    Returns
    -------
    dV : complex array (nfreq, nbins)
    fringe_u : (nbins,) bin-centre E wavenumbers (wavelengths)
    """
    nsub_e = rec.params["e_oversample"]
    du, dv = rec.cell_size
    nbins = rec.u_e.size // nsub_e
    T = rec.sky(coeffs)                                       # (nE, nN, nfreq)
    tau = spec.time_taper(rec.u_e)
    measured = rec.status == STATUS_MEASURED
    off = (np.arange(n_sub) - (n_sub - 1) / 2.0) / n_sub
    oe, on = np.meshgrid(off * du, off * dv, indexing="ij")
    UE, VN = np.meshgrid(rec.u_e, rec.v_n, indexing="ij")
    b = np.asarray(b, dtype=float)[:2]
    dV = np.zeros((rec.freqs.size, nbins), dtype=complex)
    for f, nu in enumerate(rec.freqs):
        lam = C / nu
        w0 = b / lam
        near = measured[:, None] & np.ones_like(UE, dtype=bool)
        if support_m is not None:
            reach = support_m / lam + np.hypot(du, dv)
            near &= np.hypot(UE - w0[0], VN - w0[1]) <= reach
        ie, iv = np.nonzero(near)
        if ie.size == 0:
            continue
        be = (w0[0] - (UE[ie, iv][:, None, None] + oe[None])) * lam
        bn = (w0[1] - (VN[ie, iv][:, None, None] + on[None])) * lam
        dk = np.mean(delta_kernel(be, bn, nu), axis=(1, 2))
        contrib = T[ie, iv, f] * dk * tau[ie]
        np.add.at(dV[f], ie // nsub_e, contrib)
    bin_u = rec.u_e.reshape(nbins, nsub_e).mean(axis=1)
    return dV, bin_u


def delay_transform(x, freqs, axis=0, window="blackmanharris"):
    """Windowed FFT along frequency; returns (delays_s, transformed)."""
    from scipy.signal import get_window
    n = x.shape[axis]
    win = get_window(window, n, fftbins=False)
    shape = [1] * x.ndim
    shape[axis] = n
    xt = np.fft.fftshift(np.fft.fft(x * win.reshape(shape), axis=axis), axes=axis)
    dnu = freqs[1] - freqs[0]
    return np.fft.fftshift(np.fft.fftfreq(n, dnu)), xt


# ----------------------------------------------------------------------
# Targets and summaries
# ----------------------------------------------------------------------

def coupling_target_mask(result: IdentifiabilityResult, antpos, nu: float,
                         dish_diameter: float | None = None,
                         include_self: bool = True, decimals: int = 1,
                         bls=None, coupling_radius: float | None = None):
    """Cells where first-order coupling needs T̃ at frequency ``nu``.

    Coupling features sit at antenna separations (antenna mode) and within a
    dish diameter of them (structural mode).  A coupled visibility on b_ij
    therefore draws on sky modes at (p_i − p_k − δ) ν / c, with |δ| <= D.
    That set is the antenna difference set dilated by D: the union of
    uncoupled footprints, at sub-element resolution and in both orientations.
    The FR0 strip is excluded because those modes are not in the filtered
    data.  ``include_self`` adds the zero-spacing footprint (k = i).  That
    region is measured directly only by autocorrelations.  ``decimals``
    rounds antenna separations (m) when forming the difference set; use 0
    for measured, non-lattice positions.

    If ``bls`` (measured group separations, m) and ``coupling_radius`` (m)
    are given, the targets are restricted to what those visibilities need:
    {b − p − δ}, with b over ±``bls``, p over antenna separations with
    |p| <= coupling_radius (and p = 0 if ``include_self``), and |δ| <= D.
    Use this when the data hold only some of the array's baselines.
    """
    D = result.params["dish_diameter"] if dish_diameter is None else dish_diameter
    seps, _ = unique_baselines(antpos, decimals=decimals)
    seps = np.vstack([seps, -seps] + ([np.zeros((1, 2))] if include_self else []))
    if bls is not None:
        if coupling_radius is None:
            raise ValueError("coupling_radius is required with bls")
        meas = np.asarray(bls, dtype=float)[:, :2]
        meas = np.vstack([meas, -meas])
        near = seps[np.linalg.norm(seps, axis=1) <= coupling_radius]
        cen = (meas[:, None, :] - near[None, :, :]).reshape(-1, 2)
        seps = np.unique(np.round(cen, max(decimals, 0)) + 0.0, axis=0)
    w0 = seps * nu / C
    rho = D * nu / C
    UE, VN = np.meshgrid(result.u_e, result.v_n, indexing="ij")
    mask = np.zeros(UE.shape, dtype=bool)
    for e, n in w0:
        mask |= (UE - e) ** 2 + (VN - n) ** 2 <= rho ** 2
    mask &= (result.status != STATUS_FR0)[:, None]
    return mask


def cell_snr_from_visibility_snr(snr_vis, nu: float, cell_area: float,
                                 diameter: float = HERA_DISH_DIAMETER):
    """Per-unit-cell SNR implied by a measured visibility SNR.

    For a white sky (unit variance per cell), a single-baseline visibility has
    signal variance Σ_c K_c² ≈ (1/cell_area) ∫ K² d²w.  So
    snr_cell = snr_vis / sqrt(Σ_c K_c²).

    Parameters
    ----------
    snr_vis : float or array
        Foreground SNR of one baseline (weight 1) in one channel and one
        fringe-rate bin.
    nu : float
        Frequency (Hz).
    cell_area : float
        Cell area δu_E δv_N in wavelengths².
    """
    rho = diameter * nu / C
    s = np.linspace(0.0, 1.0, 4001)
    k2 = np.trapezoid(2 * np.pi * s * disk_aperture_kernel(s, 1.0) ** 2, s)
    return np.asarray(snr_vis) / np.sqrt(k2 * rho ** 2 / cell_area)


def summarize(result: IdentifiabilityResult, mask=None, determined: float = 0.5,
              suppressions=(10.0, 100.0)):
    """Scalar summaries of identifiability over a set of cells.

    Parameters
    ----------
    mask : bool array (nE, nN), optional
        Cells to summarise.  The default is every cell outside the FR0 strip.
    determined : float
        A cell counts as fully determined when n_eff >= M − ``determined``.

    suppressions : sequence of float
        Coupling-suppression factors S.  ``frac_supp[S]`` is the fraction of
        cells whose error power fraction is at most 1/S.

    Returns
    -------
    dict with ``frac_determined``, ``mean_n_eff``, ``frac_any`` (n_eff >= 0.5),
    ``frac_alias`` (target cells lost to fringe-rate aliasing),
    ``frac_supp`` and ``n_cells``.
    """
    if mask is None:
        mask = np.broadcast_to((result.status != STATUS_FR0)[:, None],
                               result.n_chan.shape)
    neff = result.n_eff[mask]
    alias = np.broadcast_to((result.status == STATUS_ALIAS)[:, None],
                            result.n_chan.shape)[mask]
    M = result.n_modes
    n = int(mask.sum())
    if n == 0:
        return dict(frac_determined=np.nan, mean_n_eff=np.nan, frac_any=np.nan,
                    frac_alias=np.nan, frac_supp={S: np.nan for S in suppressions},
                    n_cells=0)
    err = result.error_power_fraction[mask]
    return dict(
        frac_determined=float(np.mean(neff >= M - determined)),
        mean_n_eff=float(np.mean(neff)),
        frac_any=float(np.mean(neff >= 0.5)),
        frac_alias=float(np.mean(alias)),
        frac_supp={S: float(np.mean(err <= 1.0 / S)) for S in suppressions},
        n_cells=n,
    )
