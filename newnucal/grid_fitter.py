"""
GridFitter — fast per-time-sample fitting on a regular (ℓ,m) image grid.

Workflow
--------
1. :meth:`fit_one_time` — for a single integration, jointly solve for:
   - per-channel antenna gains (4-DOF parametric model)
   - sky × beam product coefficients on a regular (ℓ,m,ν) grid
   using a type-2 NUFFT (uniform image → non-uniform baselines) and an
   alternating gradient / linear-gain-solve loop.

2. :meth:`fit_all_times` — run :meth:`fit_one_time` independently on
   every integration, returning per-time product maps and gain parameters.

3. :meth:`accumulate_product_healpix` — scatter all per-time topocentric
   product maps to an equatorial HEALPix grid using the per-time rotation
   matrices (the "type-3 accumulation" step).

4. :meth:`init_sky_coeffs_from_product` — divide the accumulated product
   by the beam model and project onto the DPSS basis to initialise
   Calibrator sky coefficients.

Coordinate conventions
----------------------
The image grid is centred at the zenith in topocentric (ℓ,m) = (sin θ
sin φ, sin θ cos φ) coordinates, with ℓ ∈ [−ℓ_max, ℓ_max) and m
similarly.  Baselines are given in wavelengths (u = B_E/λ, v = B_N/λ).
The type-2 NUFFT coordinate convention follows ``jax_finufft``: source
grid implicit, target nonuniform points are (u, v, f) with f = ν/ν_max·π.
"""

import numpy as np
import jax.numpy as jnp
from jax_finufft import nufft1, nufft2

from .array import HERAArray
from .beam import BeamModel

try:
    from fftvis.utils import speed_of_light as _C
    C = float(_C)
except ImportError:
    C = 299_792_458.0

DTYPE_R  = np.float32
DTYPE_C  = np.complex64


def _derive_product_basis(sky_basis, beam_basis, n_product_modes: int = 20):
    """Derive an orthonormal sky×beam product spectral basis.

    When both *sky_basis* and *beam_basis* carry SVD metadata (i.e.
    ``has_svd`` is True), the outer-product SVD of the weighted mode
    combinations is computed and QR-orthonormalized to produce a compact
    product basis of shape ``(n_product_modes+1, nfreq)``.

    When either basis lacks SVD metadata (e.g. built from DPSS), the sky
    basis rows are used directly as a fallback: ``sky_basis.A.T``.

    Parameters
    ----------
    sky_basis  : SkyBasis
    beam_basis : BeamBasis
    n_product_modes : int

    Returns
    -------
    product_basis : ndarray, (nbasis, nfreq)
        Rows form an orthonormal frequency basis.
    """
    if not (sky_basis.has_svd and beam_basis.has_svd):
        # Fallback: use biggest basis rows as the product basis
        A = sky_basis.A if sky_basis.nmodes > beam_basis.nmodes else beam_basis.A
        return A.T.copy().astype(DTYPE_R)  # (nmodes, nfreq)

    nfreq = sky_basis.nfreq
    # Extended beam vectors: [mean, s1*v1, s2*v2, ...]
    Vt_bm = np.vstack([
        beam_basis.svd_mean[None, :].astype(np.float64),
        beam_basis.svd_modes.astype(np.float64),
    ])   # (1+n_bm, nfreq)
    S_bm = np.concatenate([
        [float(beam_basis.n_samples)],
        beam_basis.svd_svals.astype(np.float64),
    ])   # (1+n_bm,)

    # Extended sky vectors
    Vt_sk = np.vstack([
        sky_basis.svd_mean[None, :].astype(np.float64),
        sky_basis.svd_modes.astype(np.float64),
    ])   # (1+n_sk, nfreq)
    S_sk = np.concatenate([
        [float(sky_basis.n_samples)],
        sky_basis.svd_svals.astype(np.float64),
    ])   # (1+n_sk,)

    # Form all pairwise products weighted by singular values
    # product[i,j,f] = S_bm[i] * Vt_bm[i,f] * S_sk[j] * Vt_sk[j,f]
    product = (
        S_bm[:, None, None] * Vt_bm[:, None, :]
        * S_sk[None, :, None] * Vt_sk[None, :, :]
    )   # (1+n_bm, 1+n_sk, nfreq)
    product = product.reshape(-1, nfreq)                # (prod_rows, nfreq)
    product_mean = product.mean(axis=0)                 # (nfreq,)
    _, _, Vt_pr = np.linalg.svd(product - product_mean[None, :])

    n_product_modes = min(n_product_modes, Vt_pr.shape[0])
    raw = np.vstack([product_mean[None, :], Vt_pr[:n_product_modes]])
    Q, _ = np.linalg.qr(raw.T)   # (nfreq, n_product_modes+1)
    return Q.T.astype(DTYPE_R)    # (n_product_modes+1, nfreq)


class GridFitter:
    """Fast per-time grid-based sky×beam product solver.

    Parameters
    ----------
    array : HERAArray
    sky_basis : SkyBasis
        Spectral basis for the sky.  SVD metadata (if present) is used to
        derive the joint product basis.
    beam_basis : BeamBasis
        Spectral basis for the beam.
    freqs : array_like, (nfreq,)
    nl, nm : int
        Image grid dimensions (pixels along ℓ and m).
    ell_max, m_max : float
        Half-extents of the image grid in direction-cosine units.
    eps : float
        NUFFT accuracy.
    n_product_modes : int
        Number of SVD product modes (plus mean) in the product basis.
        Ignored when either basis lacks SVD metadata.
    """

    def __init__(
        self,
        array: HERAArray,
        sky_basis,
        beam_basis,
        freqs,
        nl: int = 128,
        nm: int = 128,
        ell_max: float = 1.0,
        m_max: float = 1.0,
        eps: float = 1e-6,
        n_product_modes: int = 20,
    ):
        self.array      = array
        self.sky_basis  = sky_basis
        self.beam_basis = beam_basis
        self.freqs      = np.asarray(freqs, dtype=np.float64)
        self.nl         = nl
        self.nm         = nm
        self.ell_max    = ell_max
        self.m_max      = m_max
        self.eps        = eps

        # Image grid
        ell = np.linspace(-ell_max, ell_max, nl, endpoint=False)
        emm = np.linspace(-m_max,   m_max,   nm, endpoint=False)
        self.ELL, self.MMM = np.meshgrid(ell, emm, indexing="ij")
        self.mask_horizon = (self.ELL**2 + self.MMM**2) < 0.999**2

        # Derive orthonormal product basis: (nbasis, nfreq)
        self.product_basis = _derive_product_basis(sky_basis, beam_basis, n_product_modes)
        self.nbasis        = self.product_basis.shape[0]

        # Precompute NUFFT sampling coordinates
        self._precompute_uv()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _precompute_uv(self):
        bls = np.asarray(self.array.bls, dtype=np.float64)[:, :2]  # (nbl, 2) E/N metres
        nbl   = bls.shape[0]
        nfreq = len(self.freqs)
        # u[freq, bl] = B_E / λ,  v[freq, bl] = B_N / λ
        lam = C / self.freqs          # (nfreq,) wavelength in metres
        u   = bls[:, 0][None, :] / lam[:, None]   # (nfreq, nbl)
        v   = bls[:, 1][None, :] / lam[:, None]

        self._u_flat = u.ravel().astype(DTYPE_R)   # (nfreq*nbl,)
        self._v_flat = v.ravel().astype(DTYPE_R)
        # Frequency axis mapped to [0, π] for the NUFFT third dimension
        self._f_flat = (self.freqs / self.freqs.max()).repeat(nbl).astype(DTYPE_R)

        self._bls_en = bls.astype(DTYPE_R)         # (nbl, 2) for gain solve
        self._ant1   = self.array.antpairs[:, 0].astype(np.int32)
        self._ant2   = self.array.antpairs[:, 1].astype(np.int32)
        self.nbl     = nbl
        self.nfreq   = nfreq

    # ------------------------------------------------------------------
    # Basis helpers
    # ------------------------------------------------------------------

    def _coeff_to_product(self, coeff_maps):
        """(nl, nm, nbasis) → (nl, nm, nfreq)."""
        # product_basis is real so no conjugation needed
        return np.tensordot(coeff_maps, self.product_basis, axes=([2], [0]))

    def _product_to_coeff(self, product_nu):
        """(nl, nm, nfreq) → (nl, nm, nbasis)  (adjoint projection)."""
        return np.tensordot(product_nu, self.product_basis, axes=([2], [1]))

    def _project_to_basis(self, coeff_maps):
        """Round-trip through basis with horizon mask applied."""
        prod = self._coeff_to_product(coeff_maps) * self.mask_horizon[..., None]
        return self._product_to_coeff(prod).astype(DTYPE_C)

    # ------------------------------------------------------------------
    # NUFFT operators
    # ------------------------------------------------------------------

    def _fwd(self, coeff_maps):
        """Type-2 NUFFT: product grid → model visibilities (nfreq, nbl)."""
        prod  = self._coeff_to_product(coeff_maps) * self.mask_horizon[..., None]
        f_jnp = jnp.array(prod.astype(DTYPE_C))
        x     = jnp.array(self._u_flat)
        y     = jnp.array(self._v_flat)
        z     = jnp.array(self._f_flat * np.pi)
        out   = nufft2(f_jnp, x, y, z, iflag=-1, eps=self.eps)
        return np.array(out).reshape(self.nfreq, self.nbl).astype(DTYPE_C)

    def _adj(self, vis_fnu):
        """Type-1 NUFFT adjoint: vis (nfreq, nbl) → coefficient maps.

        The mask is applied before basis projection so this is the exact
        adjoint of :meth:`_fwd`:
        ``_fwd = nufft2 ∘ mask ∘ coeff_to_product``
        ``_adj = product_to_coeff ∘ mask ∘ nufft1``
        """
        c   = jnp.array(vis_fnu.reshape(-1).astype(DTYPE_C))
        x   = jnp.array(self._u_flat)
        y   = jnp.array(self._v_flat)
        z   = jnp.array(self._f_flat * np.pi)
        dirty = nufft1((self.nl, self.nm, self.nfreq), c, x, y, z,
                       iflag=1, eps=self.eps)
        dirty_masked = np.array(dirty) * self.mask_horizon[..., None]
        return self._product_to_coeff(dirty_masked).astype(DTYPE_C)

    # ------------------------------------------------------------------
    # Gain helpers (4-DOF per-frequency model)
    # ------------------------------------------------------------------

    def _gain_factor(self, log_amp, phase, phi):
        """Compute per-baseline gain factor from 4-DOF params.

        Parameters
        ----------
        log_amp : (nfreq,)
        phase   : (nfreq,)
        phi     : (2, nfreq)
        bls     : (nbl, 2)

        Returns (nfreq, nbl) complex64.
        """
        # phi @ bls.T → (nfreq, nbl)
        phs_grad = (phi[0, :, None] * self._bls_en[None, :, 0]
                    + phi[1, :, None] * self._bls_en[None, :, 1])  # (nfreq, nbl)
        return (np.exp(log_amp[:, None]).astype(DTYPE_C)
                * np.exp(1j * (phase[:, None] + phs_grad)).astype(DTYPE_C))

    def _fit_gains_linear_grid(self, vis_model_fn, data_fnu):
        """4-DOF linear gain solve against the grid model.

        Replicates the math of :meth:`~newnucal.calibrator.Calibrator.fit_gains_linear`
        for a single time step.

        Parameters
        ----------
        vis_model_fn : (nfreq, nbl) complex — model visibilities
        data_fnu     : (nfreq, nbl) complex — observed data for this time

        Returns
        -------
        gain_params : dict with arrays
            ``log_amp`` (nfreq,), ``phase`` (nfreq,), ``phi`` (2, nfreq)
        """
        den    = np.abs(vis_model_fn) ** 2              # (nfreq, nbl)
        g_opt  = data_fnu * np.conj(vis_model_fn) / (den + 1e-30)
        log_g  = np.log(g_opt + 0j)                     # complex log

        w_sum  = den.sum(axis=1) + 1e-30                # (nfreq,)
        log_amp = (den * np.real(log_g)).sum(axis=1) / w_sum

        nbl = self.nbl
        bls = self._bls_en                              # (nbl, 2)
        X   = np.column_stack([np.ones(nbl), bls[:, 0], bls[:, 1]])  # (nbl, 3)
        Xy  = np.einsum('bk,fb->fk', X, den * np.imag(log_g))        # (nfreq, 3)
        XTX = np.einsum('bk,fb,bl->fkl', X, den, X) + 1e-10 * np.eye(3)
        theta = np.linalg.solve(XTX, Xy[..., None])[..., 0]          # (nfreq, 3)

        return {
            'log_amp': log_amp.astype(DTYPE_R),
            'phase':   theta[:, 0].astype(DTYPE_R),
            'phi':     theta[:, 1:].T.astype(DTYPE_R),   # (2, nfreq)
        }

    # ------------------------------------------------------------------
    # Operator-norm and preconditioner estimates
    # ------------------------------------------------------------------

    def estimate_lipschitz(self, niter: int = 8, seed: int = 42) -> float:
        """Estimate ‖F^H F‖ for step-size calibration via power iteration."""
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal((self.nl, self.nm, self.nbasis))
             + 1j * rng.standard_normal((self.nl, self.nm, self.nbasis))
             ).astype(DTYPE_C)
        x /= np.linalg.norm(x.ravel()) + 1e-12
        for _ in range(niter):
            y  = self._adj(self._fwd(x)).astype(DTYPE_C)
            yn = float(np.linalg.norm(y.ravel()))
            if not np.isfinite(yn) or yn == 0:
                break
            x = y / yn
        y = self._adj(self._fwd(x)).astype(DTYPE_C)
        return float(np.real(np.vdot(x.ravel(), y.ravel())))

    def estimate_lipschitz_preconditioned(
        self, precond: np.ndarray, niter: int = 8, seed: int = 43
    ) -> float:
        """Estimate ‖D⁻¹ F^H F‖ for the preconditioned operator."""
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal((self.nl, self.nm, self.nbasis))
             + 1j * rng.standard_normal((self.nl, self.nm, self.nbasis))
             ).astype(DTYPE_C)
        x /= np.linalg.norm(x.ravel()) + 1e-12
        for _ in range(niter):
            y  = (self._adj(self._fwd(x)) / precond.astype(DTYPE_C)).astype(DTYPE_C)
            yn = float(np.linalg.norm(y.ravel()))
            if not np.isfinite(yn) or yn == 0:
                break
            x = y / yn
        y = (self._adj(self._fwd(x)) / precond.astype(DTYPE_C)).astype(DTYPE_C)
        return float(np.real(np.vdot(x.ravel(), y.ravel())))

    def estimate_diagonal_precond(
        self, nprobe: int = 8, seed: int = 99
    ) -> np.ndarray:
        """Estimate the diagonal of F^H F as a preconditioner.

        Returns an array of shape ``(nl, nm, nbasis)`` whose entries
        approximate the per-coefficient sensitivity.  Dividing the
        gradient by this array normalises the curvature across pixels
        and modes, accelerating convergence.
        """
        rng = np.random.default_rng(seed)
        acc = np.zeros((self.nl, self.nm, self.nbasis), dtype=DTYPE_R)
        for _ in range(nprobe):
            z = (rng.standard_normal((self.nfreq, self.nbl))
                 + 1j * rng.standard_normal((self.nfreq, self.nbl))
                 ).astype(DTYPE_C)
            y = self._adj(z)
            acc += np.abs(y).astype(DTYPE_R) ** 2
        acc /= nprobe
        # Floor at the 5th percentile to avoid division by near-zero
        floor = float(np.percentile(acc[acc > 0], 5)) if np.any(acc > 0) else 1.0
        acc = np.maximum(acc, floor)
        acc /= np.median(acc)
        return acc

    # ------------------------------------------------------------------
    # Per-time solver
    # ------------------------------------------------------------------

    def fit_one_time(
        self,
        data_fnu,
        gain_params_init: dict | None = None,
        n_iter: int = 40,
        gain_refresh_every: int = 4,
        ridge: float = 1e-3,
        line_search_tol: float = 1.002,
        line_search_depth: int = 18,
        lipschitz: float | None = None,
        precond: np.ndarray | None = None,
        verbose: bool = False,
    ):
        """Fit sky×beam product and 4-DOF gains for one integration.

        The loop alternates between:
        - A (optionally preconditioned) gradient step in coefficient-map
          space (operating on gain-divided data) with a backtracking line
          search.
        - A linear 4-DOF gain solve against the current model.

        Parameters
        ----------
        data_fnu : array_like, (nfreq, nbl) complex
            Observed visibilities for this time step.
        gain_params_init : dict or None
            Initial ``{log_amp, phase, phi}`` (single-time, no time dim).
            ``None`` → zero gains.
        n_iter : int
        gain_refresh_every : int
        ridge : float
            L2 coefficient applied to ``coeff_est`` at each gradient step.
        line_search_tol : float
        line_search_depth : int
        lipschitz : float or None
            If ``None``, estimated via :meth:`estimate_lipschitz`.
        precond : ndarray, (nl, nm, nbasis) or None
            Diagonal preconditioner.  If ``None``, no preconditioning is
            applied.  Use :meth:`estimate_diagonal_precond` to build one.
        verbose : bool

        Returns
        -------
        coeff_maps : ndarray, (nl, nm, nbasis), complex64
        gain_params : dict
            ``log_amp`` (nfreq,), ``phase`` (nfreq,), ``phi`` (2, nfreq)
        history : dict
            ``chi2_cal`` and ``accepted_step`` lists.
        """
        data_fnu = np.asarray(data_fnu, dtype=DTYPE_C)   # (nfreq, nbl)

        if gain_params_init is None:
            gp = {
                'log_amp': np.zeros(self.nfreq, dtype=DTYPE_R),
                'phase':   np.zeros(self.nfreq, dtype=DTYPE_R),
                'phi':     np.zeros((2, self.nfreq), dtype=DTYPE_R),
            }
        else:
            gp = {k: np.asarray(v, dtype=DTYPE_R) for k, v in gain_params_init.items()}

        # Lipschitz is only used as a fallback: the main loop uses exact steps.
        # It remains an argument to allow callers that pre-compute it to pass it.
        _step_fallback = 0.2 / max(lipschitz, 1e-30) if lipschitz is not None else None

        # Initialise from adjoint dirty map, scaled to match data amplitude
        g_fac    = self._gain_factor(gp['log_amp'], gp['phase'], gp['phi'])
        data_cal = data_fnu / (g_fac + 1e-30)
        coeff_est = self._adj(data_cal)
        coeff_est = self._project_to_basis(coeff_est)
        vis_init  = self._fwd(coeff_est)
        norm_vis  = float(np.linalg.norm(vis_init.ravel()))
        if norm_vis > 1e-30:
            alpha = float(np.linalg.norm(data_cal.ravel())) / norm_vis
            coeff_est = coeff_est * alpha

        history = {'chi2_cal': [], 'accepted_step': []}

        for it in range(n_iter):
            model_ung = self._fwd(coeff_est)             # (nfreq, nbl)
            g_fac     = self._gain_factor(gp['log_amp'], gp['phase'], gp['phi'])
            data_cal  = data_fnu / (g_fac + 1e-30)
            resid_cal = data_cal - model_ung
            chi2_cal  = float(np.mean(np.abs(resid_cal) ** 2))
            history['chi2_cal'].append(chi2_cal)

            # Gradient step (with optional preconditioning)
            grad = self._adj(resid_cal).astype(DTYPE_C)
            if precond is not None:
                pgrad = grad / precond.astype(DTYPE_C)
            else:
                pgrad = grad

            # Exact steepest-descent step:
            #   α = Re(<r, A pgrad>) / ||A pgrad||²
            # This is the largest step that minimises chi2 along pgrad.
            vis_pgrad = self._fwd(pgrad)                          # (nfreq, nbl)
            num_sd    = float(np.real(np.vdot(resid_cal.ravel(), vis_pgrad.ravel())))
            den_sd    = float(np.linalg.norm(vis_pgrad.ravel()) ** 2) + 1e-30
            step_sd   = max(num_sd / den_sd, 0.0)

            # Use exact step as initial guess; backtrack only if ridge causes issues
            accepted   = False
            step_used  = 0.0
            for ls in range(line_search_depth):
                s           = step_sd * (0.5 ** ls)
                coeff_trial = self._project_to_basis(
                    (1.0 - ridge) * coeff_est + s * pgrad
                )
                chi2_trial = float(np.mean(
                    np.abs(data_cal - self._fwd(coeff_trial)) ** 2
                ))
                if np.isfinite(chi2_trial) and chi2_trial <= line_search_tol * chi2_cal:
                    coeff_est = coeff_trial
                    accepted  = True
                    step_used = s
                    break
            history['accepted_step'].append(step_used)

            # Gain refresh
            if (it + 1) % gain_refresh_every == 0:
                model_ung_new = self._fwd(coeff_est)
                gp_new = self._fit_gains_linear_grid(model_ung_new, data_fnu)
                # Accept only if loss doesn't increase
                g_new = self._gain_factor(gp_new['log_amp'], gp_new['phase'], gp_new['phi'])
                g_old = self._gain_factor(gp['log_amp'], gp['phase'], gp['phi'])
                chi2_new = float(np.mean(np.abs(data_fnu - model_ung_new * g_new) ** 2))
                chi2_old = float(np.mean(np.abs(data_fnu - model_ung_new * g_old) ** 2))
                if np.isfinite(chi2_new) and chi2_new <= 1.05 * chi2_old:
                    gp = gp_new

            if verbose:
                print(f"  iter {it:03d}: chi2_cal={chi2_cal:.4e}  "
                      f"step={'ok' if accepted else '--'}={step_used:.3e}")

        return coeff_est, gp, history

    # ------------------------------------------------------------------
    # Multi-time fitting
    # ------------------------------------------------------------------

    def fit_all_times(
        self,
        data,
        gain_params_init: dict | None = None,
        n_iter: int = 40,
        use_precond: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """Fit each time step independently.

        Parameters
        ----------
        data : array_like, (ntime, nfreq, nbl) complex
        gain_params_init : dict or None
            Per-time initial gains with time axis first:
            ``log_amp`` (ntime, nfreq), etc.  ``None`` → zero gains.
        n_iter : int
        use_precond : bool
            If ``True`` (default), estimate and apply a diagonal
            preconditioner (adds ``~nprobe`` NUFFT pairs to setup, but
            typically halves the number of iterations needed).
        verbose : bool
        **kwargs
            Forwarded to :meth:`fit_one_time`.

        Returns
        -------
        coeff_maps_all : list of ndarray, len ntime, each (nl, nm, nbasis)
        gain_params_all : dict
            ``log_amp`` (ntime, nfreq), ``phase`` (ntime, nfreq),
            ``phi`` (ntime, 2, nfreq)
        histories : list of dict
        """
        data   = np.asarray(data, dtype=DTYPE_C)
        ntime  = data.shape[0]

        precond = self.estimate_diagonal_precond() if use_precond else None

        coeff_maps_all = []
        log_amps       = []
        phases         = []
        phis           = []
        histories      = []

        for t in range(ntime):
            if verbose:
                print(f"[GridFitter] time {t + 1}/{ntime}")
            if gain_params_init is not None:
                gp0 = {
                    'log_amp': np.asarray(gain_params_init['log_amp'][t]),
                    'phase':   np.asarray(gain_params_init['phase'][t]),
                    'phi':     np.asarray(gain_params_init['phi'][t]),
                }
            else:
                gp0 = None

            cm, gp, hist = self.fit_one_time(
                data[t], gp0, n_iter=n_iter, precond=precond,
                verbose=verbose, **kwargs
            )
            coeff_maps_all.append(cm)
            log_amps.append(gp['log_amp'])
            phases.append(gp['phase'])
            phis.append(gp['phi'])
            histories.append(hist)

        gain_params_all = {
            'log_amp': np.stack(log_amps, axis=0),   # (ntime, nfreq)
            'phase':   np.stack(phases,   axis=0),   # (ntime, nfreq)
            'phi':     np.stack(phis,     axis=0),   # (ntime, 2, nfreq)
        }
        return coeff_maps_all, gain_params_all, histories

    # ------------------------------------------------------------------
    # Product accumulation and sky initialisation
    # ------------------------------------------------------------------

    def accumulate_product_healpix(
        self,
        coeff_maps_all,
        rot_matrices,
        sky_nside: int,
    ):
        """Scatter per-time topocentric product maps to equatorial HEALPix.

        For each time *t* and each above-horizon grid pixel (ℓ,m):
          1. Reconstruct ``P_t(ℓ,m,ν)`` from ``coeff_maps_all[t]``.
          2. Rotate the topocentric direction ``(ℓ,m,n)`` to equatorial via
             ``R_t^{-1} = R_t^T`` (the rotation matrices are orthogonal).
          3. Find the nearest HEALPix pixel and accumulate.

        Parameters
        ----------
        coeff_maps_all : list of ndarray, len ntime, each (nl, nm, nbasis)
        rot_matrices   : array_like, (ntime, 3, 3)
        sky_nside      : int

        Returns
        -------
        product_map : ndarray, (npix_sky, nfreq)
            Beam-weighted accumulated product.
        weights     : ndarray, (npix_sky,)
            Accumulation counts per pixel.
        """
        import healpy as hp

        npix_sky = hp.nside2npix(sky_nside)
        nfreq    = self.nfreq
        rot_ms   = np.asarray(rot_matrices, dtype=np.float64)

        # Flatten above-horizon pixels
        l_flat = self.ELL[self.mask_horizon].ravel().astype(np.float64)
        m_flat = self.MMM[self.mask_horizon].ravel().astype(np.float64)
        n_flat = np.sqrt(np.clip(1.0 - l_flat ** 2 - m_flat ** 2, 0.0, None))
        topo   = np.stack([l_flat, m_flat, n_flat], axis=0)  # (3, npix_vis)

        sky_num = np.zeros((npix_sky, nfreq), dtype=np.float64)
        sky_den = np.zeros(npix_sky, dtype=np.float64)

        for t, cm in enumerate(coeff_maps_all):
            prod_nm  = self._coeff_to_product(cm).real        # (nl, nm, nfreq)
            prod_vis = prod_nm[self.mask_horizon]             # (npix_vis, nfreq)

            # Topocentric → equatorial: R_t^T @ topo
            eq_xyz   = rot_ms[t].T @ topo                    # (3, npix_vis)
            eq_theta = np.arccos(np.clip(eq_xyz[2], -1.0, 1.0))
            eq_phi   = np.arctan2(eq_xyz[1], eq_xyz[0]) % (2 * np.pi)
            sky_pix  = hp.ang2pix(sky_nside, eq_theta, eq_phi)

            np.add.at(sky_num, sky_pix, prod_vis)
            np.add.at(sky_den, sky_pix, 1.0)

        valid = sky_den > 0
        product_map = np.zeros((npix_sky, nfreq), dtype=DTYPE_R)
        product_map[valid] = (sky_num[valid] / sky_den[valid, None]).astype(DTYPE_R)

        return product_map, sky_den.astype(DTYPE_R)

    def init_sky_coeffs_from_product(
        self,
        product_map,
        beam_model: BeamModel | None = None,
        beam_reg: float = 1e-3,
    ):
        """Project an accumulated product map onto the sky basis.

        Parameters
        ----------
        product_map : ndarray, (npix_sky, nfreq)
            Output of :meth:`accumulate_product_healpix`.
        beam_model : BeamModel or None
            If provided, the product is divided by the mean beam spectrum
            before projecting, yielding a rough sky-only estimate.
        beam_reg : float
            Regularisation added to the beam denominator.

        Returns
        -------
        sky_coeffs : jnp.ndarray, (npix_sky, nmodes_sky)
        """
        sky_map = np.asarray(product_map, dtype=DTYPE_R)  # (npix_sky, nfreq)

        if beam_model is not None:
            # Mean beam spectrum evaluated at each HEALPix pixel.
            # Approximate: use the pixel-averaged beam coefficients.
            beam_spec = (np.asarray(beam_model.coeffs) @
                         np.asarray(beam_model.A).T)    # (npix_beam, nfreq)
            # beam_nside may differ from sky_nside; use nearest-pixel lookup
            import healpy as hp
            sky_nside  = hp.npix2nside(sky_map.shape[0])
            beam_nside = beam_model.nside
            npix_sky   = sky_map.shape[0]
            th, ph = hp.pix2ang(sky_nside, np.arange(npix_sky))
            beam_pix   = hp.ang2pix(beam_nside, th, ph)
            beam_at_sky = beam_spec[beam_pix]                # (npix_sky, nfreq)
            sky_map = sky_map / (beam_at_sky + beam_reg)

        sky_coeffs = self.sky_basis.project(sky_map).astype(DTYPE_R)
        return jnp.array(sky_coeffs)
