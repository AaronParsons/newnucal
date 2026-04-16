"""
Calibrator — ties together ForwardModel, gain parameters, and optimisation.
"""

import signal as _signal
import time as _time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jaxopt import LBFGS

from .array import HERAArray
from .beam import BeamModel
from .dpss import dpss_project
from .simulate import ForwardModel
from .gains import apply_gains, init_gain_params
from .rfi import RFIConfig, prepare_initial_channel_weights, update_channel_weights_from_residuals

DTYPE_R = jnp.float32
DTYPE_C = jnp.complex64


class _StopFitFlag:
    def __init__(self):
        self.stop = False

    def __call__(self, signum, frame):
        self.stop = True
        print("\n  [stop requested — finishing current iteration and returning]", flush=True)

    @staticmethod
    def install(flag):
        try:
            return _signal.signal(_signal.SIGINT, flag)
        except (ValueError, OSError):
            return None

    @staticmethod
    def restore(old_handler):
        if old_handler is not None:
            try:
                _signal.signal(_signal.SIGINT, old_handler)
            except (ValueError, OSError):
                pass


class AndersonAccelerator:
    """Anderson acceleration buffer for fixed-point iterations.

    Accumulates iterates and residuals in a sliding window, then proposes a
    mixed candidate via constrained least-squares.  Works on flat numpy arrays;
    the caller is responsible for flattening inputs and reshaping outputs.

    Parameters
    ----------
    history : int
        Maximum number of past iterates to keep.  ``0`` disables AA.
    start : int
        Number of plain steps to take before the first AA proposal.
        The step counter increments after each :meth:`push` call, so
        ``start=2`` activates AA from the third push onward (consistent
        with the original per-loop counter behaviour).
    damping : float
        Mixing weight: proposal = ``(1-damping)*g_plain + damping*g_aa``.
    ridge : float
        Tikhonov regularisation coefficient for the least-squares solve,
        applied relative to the Gram-matrix trace.
    max_weight : float
        Reject proposals where any |beta_i| exceeds this value.
    """

    def __init__(
        self,
        history: int,
        start: int = 2,
        damping: float = 0.5,
        ridge: float = 1e-8,
        max_weight: float = 10.0,
    ):
        self.history    = history
        self.start      = start
        self.damping    = damping
        self.ridge      = ridge
        self.max_weight = max_weight
        self._hist_g: list[np.ndarray] = []
        self._hist_f: list[np.ndarray] = []
        self._step = 0

    def push(
        self, x_flat: np.ndarray, g_flat: np.ndarray
    ) -> np.ndarray | None:
        """Record one plain step and return an AA-mixed candidate, or None.

        Parameters
        ----------
        x_flat : np.ndarray, shape (n,)
            Current iterate, flattened to 1-D float64.
        g_flat : np.ndarray, shape (n,)
            Plain fixed-point step result, flattened to 1-D float64.

        Returns
        -------
        np.ndarray or None
            Damped AA proposal ``(1-damping)*g_flat + damping*g_aa``, or
            ``None`` when AA is disabled, not yet active, or the mixing
            coefficients fail validity checks (non-finite or |beta| > max_weight).
        """
        if self.history == 0:
            return None

        f_flat = g_flat - x_flat
        self._hist_g.append(g_flat.copy())
        self._hist_f.append(f_flat.copy())
        if len(self._hist_g) > self.history:
            self._hist_g.pop(0)
            self._hist_f.pop(0)

        candidate = None
        if self._step >= self.start and len(self._hist_f) >= 2:
            F    = np.stack(self._hist_f, axis=0)
            beta = self._solve_coeffs(F, self.ridge)
            if (np.all(np.isfinite(beta))
                    and float(np.max(np.abs(beta))) <= self.max_weight):
                g_aa = np.tensordot(beta, np.stack(self._hist_g, axis=0), axes=1)
                candidate = (1.0 - self.damping) * g_flat + self.damping * g_aa

        self._step += 1
        return candidate

    def clear(self):
        """Reset history and step counter (e.g. after an L-BFGS step)."""
        self._hist_g.clear()
        self._hist_f.clear()
        self._step = 0

    @staticmethod
    def _solve_coeffs(F: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
        """Constrained least-squares mixing coefficients.

        Solves ``min ||F^T beta||`` subject to ``sum(beta) = 1`` with
        Tikhonov regularisation ``ridge * trace(G) * I`` added to the
        Gram matrix ``G = F F^T``.

        Parameters
        ----------
        F : np.ndarray, shape (m, n)
            Residual matrix whose rows are past residuals ``f_i = g_i - x_i``.

        Returns
        -------
        beta : np.ndarray, shape (m,)
        """
        n = F.shape[0]
        gram = F @ F.T
        scale = float(np.trace(gram) / max(n, 1)) if n > 0 else 1.0
        gram = gram + ridge * max(scale, 1.0) * np.eye(n, dtype=gram.dtype)
        kkt = np.block([
            [gram, np.ones((n, 1), dtype=gram.dtype)],
            [np.ones((1, n), dtype=gram.dtype), np.zeros((1, 1), dtype=gram.dtype)],
        ])
        rhs = np.zeros(n + 1, dtype=gram.dtype)
        rhs[-1] = 1.0
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        return sol[:n]


@dataclass
class AlternatingDirtyFitState:
    """Persistent state for resumable fit_alternating_dirty runs."""
    params: dict
    settings: dict = field(default_factory=dict)
    loss: float | None = None
    reduced_chi2: float | None = None
    step: int = 0

    sky_acc: AndersonAccelerator | None = None
    beam_acc: AndersonAccelerator | None = None

    eff_sky: float | None = None
    eff_beam: float | None = None
    eff_gains: float | None = None
    eff_beam_lbfgs: float | None = None

    n_sky: int = 0
    n_beam: int = 0
    n_gains: int = 0
    n_beam_lbfgs: int = 0

    n_since_sky: int = 0
    n_since_beam: int = 0
    n_since_gains: int = 0
    n_since_beam_lbfgs: int = 0
    n_beam_since_beam_lbfgs: int = 0

    beam_dirty_pending: bool = False
    stop_reason: str | None = None


class Calibrator:
    def __init__(
        self,
        array: HERAArray,
        beam_model: BeamModel,
        sky_model,
        freqs,
        rot_matrices,
        data,
        eps: float = 1e-6,
        eta_padding: float = 0.0,
        channel_weights=None,
        inv_noise_var=None,
        noise_sigma=None,
    ):
        """
        Parameters
        ----------
        sky_model : SkyModel
            Sky model providing ``nside`` and the spectral basis ``A_sky``.
        """
        self.freqs = jnp.array(freqs, dtype=DTYPE_R)
        self.rot_matrices = jnp.array(rot_matrices, dtype=DTYPE_R)
        self.data = jnp.array(data, dtype=DTYPE_C)
        self.bls = jnp.array(array.bls, dtype=DTYPE_R)
        self.channel_weights = jnp.ones(self.data.shape[:2], dtype=DTYPE_R)
        self.inv_noise_var = jnp.ones(self.data.shape[:2], dtype=DTYPE_R)

        self.A_sky = np.asarray(sky_model.A_sky, dtype=np.float32)  # (nfreq, nmodes)
        sky_nside  = sky_model.nside

        self.fwd = ForwardModel(
            array,
            sky_nside,
            beam_model,
            freqs,
            eps=eps,
            eta_max=None,
            eta_padding=eta_padding,
        )
        self.fwd.set_sky_basis(self.A_sky)
        self.fwd.precompute_time_geometry(self.rot_matrices)

        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))
        self._jit_loss_variable_beam = jax.jit(self._loss_variable_beam)
        self._jit_simulate = jax.jit(self.fwd.simulate)
        self.set_channel_weights(channel_weights)
        self.set_inv_noise_var(inv_noise_var)
        if noise_sigma is not None:
            self.set_noise_sigma(noise_sigma)

    def _effective_weights(self):
        return (self.channel_weights * self.inv_noise_var).astype(DTYPE_R)

    def _weighted_chi2(self, resid):
        """Weighted chi2 using the current channel_weights * inv_noise_var."""
        w = self._effective_weights()
        return jnp.sum(w[:, :, None] * (jnp.abs(resid) ** 2))

    def _weighted_mean_chi2(self, resid):
        w = self._effective_weights()
        den = jnp.maximum(jnp.sum(w) * resid.shape[2], 1e-12)
        return self._weighted_chi2(resid) / den

    def _loss(self, params, weights):
        """Loss function with weights passed explicitly to avoid JIT constant baking."""
        vis_model = self.fwd.simulate(params['sky_coeffs'], self.rot_matrices)
        vis_cal = apply_gains(vis_model, params['log_amp'], params['phase'], params['phi'], self.bls)
        return jnp.sum(weights[:, :, None] * (jnp.abs(self.data - vis_cal) ** 2))

    def _loss_variable_beam(self, params, weights):
        """Loss function (variable beam) with weights passed explicitly."""
        vis_model = self.fwd.simulate_variable_beam(
            params['sky_coeffs'], params['beam_coeffs'], self.rot_matrices
        )
        vis_cal = apply_gains(vis_model, params['log_amp'], params['phase'], params['phi'], self.bls)
        return jnp.sum(weights[:, :, None] * (jnp.abs(self.data - vis_cal) ** 2))

    def calc_loss(self, params, explicit_beam: bool = False):
        # Fast path: return the loss already computed by fit_gains_linear when
        # called immediately after with the same (sky_coeffs, gain_params) arrays.
        # Object-identity checks are safe because JAX arrays are immutable.
        if not explicit_beam:
            cache = getattr(self, '_fit_gains_linear_cache', None)
            if cache is not None:
                sc, gp, cached_loss = cache
                if (params.get('sky_coeffs') is sc
                        and params.get('log_amp') is gp.get('log_amp')
                        and params.get('phase') is gp.get('phase')
                        and params.get('phi') is gp.get('phi')):
                    return cached_loss
        w = self._effective_weights()
        if explicit_beam:
            return float(self._jit_loss_variable_beam(params, w))
        return float(self._jit_loss(params, w))

    def calc_chi2(self, params, explicit_beam: bool = False):
        return self.calc_loss(params, explicit_beam=explicit_beam)

    def estimate_num_params(self, params=None):
        if params is None:
            return 0
        npar = 0
        if 'sky_coeffs' in params and params['sky_coeffs'] is not None:
            npar += int(np.prod(np.shape(params['sky_coeffs'])))
        if 'beam_coeffs' in params and params['beam_coeffs'] is not None:
            npar += int(np.prod(np.shape(params['beam_coeffs'])))
        for key in ('log_amp', 'phase', 'phi'):
            if key in params and params[key] is not None:
                npar += int(np.prod(np.shape(params[key])))
        return npar


    def set_channel_weights(self, channel_weights=None):
        """Set per-time/per-frequency soft reliability weights.

        Parameters
        ----------
        channel_weights : array_like or None
            Shape ``(ntime, nfreq)`` or ``(nfreq,)``. Values are clipped into
            ``[0, 1]``. ``None`` restores unit weights.
        """
        if channel_weights is None:
            arr = jnp.ones(self.data.shape[:2], dtype=DTYPE_R)
        else:
            arr = jnp.array(channel_weights, dtype=DTYPE_R)
            if arr.ndim == 1:
                arr = jnp.broadcast_to(arr[None, :], self.data.shape[:2])
            elif arr.shape != self.data.shape[:2]:
                raise ValueError(
                    f"channel_weights must have shape {(self.ntime, self.nfreq)} "
                    f"or {(self.nfreq,)}, got {arr.shape}"
                )
        self.channel_weights = jnp.clip(arr, 0.0, 1.0).astype(DTYPE_R)

    def set_inv_noise_var(self, inv_noise_var=None):
        """Set per-time/per-frequency inverse noise variance."""
        if inv_noise_var is None:
            arr = jnp.ones(self.data.shape[:2], dtype=DTYPE_R)
        else:
            arr = jnp.array(inv_noise_var, dtype=DTYPE_R)
            if arr.ndim == 1:
                arr = jnp.broadcast_to(arr[None, :], self.data.shape[:2])
            elif arr.shape != self.data.shape[:2]:
                raise ValueError(
                    f"inv_noise_var must have shape {(self.ntime, self.nfreq)} "
                    f"or {(self.nfreq,)}, got {arr.shape}"
                )
        self.inv_noise_var = jnp.clip(arr, 0.0, jnp.inf).astype(DTYPE_R)

    def set_noise_sigma(self, noise_sigma=None):
        """Set per-time/per-frequency noise sigma.

        Parameters
        ----------
        noise_sigma : array_like or None
            Shape ``(ntime, nfreq)`` or ``(nfreq,)``. ``None`` restores unit
            variance.
        """
        if noise_sigma is None:
            self.set_inv_noise_var(None)
            return
        arr = jnp.array(noise_sigma, dtype=DTYPE_R)
        if arr.ndim == 1:
            arr = jnp.broadcast_to(arr[None, :], self.data.shape[:2])
        elif arr.shape != self.data.shape[:2]:
            raise ValueError(
                f"noise_sigma must have shape {(self.ntime, self.nfreq)} "
                f"or {(self.nfreq,)}, got {arr.shape}"
            )
        arr = jnp.clip(arr, 1e-20, jnp.inf)
        self.inv_noise_var = (1.0 / (arr ** 2)).astype(DTYPE_R)

    def calc_reduced_chi2(self, params, explicit_beam: bool = False, subtract_params=0):
        """Approximate reduced chi-squared under the current weights/noise model.

        The denominator is the effective number of data points (sum of channel
        weights times number of baselines), optionally minus the number of free
        parameters.  The default ``subtract_params=0`` gives chi2 per data
        point, which equals 1 when the noise model is correct.  Pass
        ``subtract_params='auto'`` to subtract the full parameter count, or an
        integer to subtract a specific number.
        """
        chi2 = self.calc_chi2(params, explicit_beam=explicit_beam)
        if subtract_params == 'auto':
            npar = self.estimate_num_params(params)
        else:
            npar = int(subtract_params)
        dof = max(float(jnp.sum(self.channel_weights) * self.nbls) - npar, 1.0)
        return float(chi2 / dof)


    def init_params(self):
        npix_sky = self.fwd.npix_sky
        nmodes_sky = self.A_sky.shape[1]
        return {
            'sky_coeffs':  jnp.zeros((npix_sky, nmodes_sky), dtype=DTYPE_R),
            'beam_coeffs': jnp.array(self.fwd.beam_coeffs),
            **init_gain_params(self.ntime, self.nfreq),
        }

    def init_sky_from_flux(self, flux):
        return jnp.array(dpss_project(np.asarray(flux), self.A_sky), dtype=DTYPE_R)

    def simulate(self, params):
        """Return gain-calibrated model visibilities for the given parameters.

        Parameters
        ----------
        params : dict
            Must include ``sky_coeffs``, ``log_amp``, ``phase``, ``phi``.
            If ``beam_coeffs`` is present, ``simulate_variable_beam`` is used;
            otherwise the precomputed beam cache is used.

        Returns
        -------
        vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex64
        """
        if 'beam_coeffs' in params:
            vis_model = self.fwd.simulate_variable_beam(
                params['sky_coeffs'], params['beam_coeffs'], self.rot_matrices
            )
        else:
            vis_model = self.fwd.simulate(params['sky_coeffs'], self.rot_matrices)
        return apply_gains(
            vis_model, params['log_amp'], params['phase'], params['phi'], self.bls
        )

    @staticmethod
    def _invert_gains(gain_params):
        """Return gain parameters with negated log_amp, phase, and phi."""
        return {k: -gain_params[k] for k in ('log_amp', 'phase', 'phi')}

    def calibrated_residual(self, params):
        inv = self._invert_gains(params)
        data_cal = apply_gains(self.data, inv['log_amp'], inv['phase'], inv['phi'], self.bls)
        vis_model = self.fwd.simulate(params['sky_coeffs'], self.rot_matrices)
        resid = data_cal - vis_model
        return resid * (self.channel_weights * self.inv_noise_var)[:, :, None].astype(DTYPE_C)

    def calibrated_residual_variable_beam(self, params):
        inv = self._invert_gains(params)
        data_cal = apply_gains(self.data, inv['log_amp'], inv['phase'], inv['phi'], self.bls)
        vis_model = self.fwd.simulate_variable_beam(
            params['sky_coeffs'], params['beam_coeffs'], self.rot_matrices
        )
        resid = data_cal - vis_model
        return resid * (self.channel_weights * self.inv_noise_var)[:, :, None].astype(DTYPE_C)

    # ------------------------------------------------------------------
    # Pixel-cut helpers
    # ------------------------------------------------------------------

    def apply_horizon_cut(self):
        """Remove sky pixels that are never above the horizon.

        Permanently reduces ``npix_sky``; call *before* initialising sky
        coefficients.  Returns the number of pixels retained.
        """
        mask = self.fwd.build_ever_visible_mask(self.rot_matrices)
        self.fwd.apply_pixel_mask(mask)
        self.fwd.precompute_time_geometry(self.rot_matrices)
        n_keep = int(mask.sum())
        n_drop = int((~mask).sum())
        print(f'  Horizon cut: {n_keep} pixels retained, {n_drop} removed.')
        return n_keep

    def fit_gains_linear(self, sky_coeffs):
        vis_model = self._jit_simulate(sky_coeffs, self.rot_matrices)
        # Per-time optimal complex gain: data · conj(model) / |model|²
        # Shape: (ntime, nfreq, nbls)
        den = np.array(jnp.abs(vis_model) ** 2)          # (ntime, nfreq, nbls), weights
        g_opt = np.array(self.data * jnp.conj(vis_model)) / (den + 1e-30)
        log_g = np.log(g_opt + 0j)                        # complex log, (ntime, nfreq, nbls)

        # log_amp: baseline-weighted mean of Re(log g) per (time, freq)
        w_sum = den.sum(axis=2) + 1e-30                   # (ntime, nfreq)
        log_amp = (den * np.real(log_g)).sum(axis=2) / w_sum  # (ntime, nfreq)

        # phase and phi: weighted lstsq over baselines, independently per (time, freq)
        # Design matrix X: (nbls, 3) — [1, bl_E, bl_N]
        bls = np.array(self.bls)
        X = np.column_stack([np.ones(self.nbls), bls[:, 0], bls[:, 1]])  # (nbls, 3)

        # Xy[t, f, k] = sum_b X[b, k] * den[t, f, b] * Im(log_g[t, f, b])
        Xy = np.einsum('bk,tfb->tfk', X, den * np.imag(log_g))   # (ntime, nfreq, 3)
        # XTX[t, f, k, l] = sum_b X[b, k] * den[t, f, b] * X[b, l]
        XTX = np.einsum('bk,tfb,bl->tfkl', X, den, X)             # (ntime, nfreq, 3, 3)
        XTX += 1e-10 * np.eye(3)                                   # regularise

        # Solve all (time, freq) systems at once via numpy broadcasting.
        # Add trailing dim so solve sees (..., 3, 1) not (..., 3) — required
        # by numpy ≥2.0 which no longer treats b.ndim == a.ndim-1 as a vector.
        theta = np.linalg.solve(XTX, Xy[..., None])[..., 0]       # (ntime, nfreq, 3)

        gain_params = {
            'log_amp': jnp.array(log_amp, dtype=DTYPE_R),             # (ntime, nfreq)
            'phase':   jnp.array(theta[:, :, 0], dtype=DTYPE_R),      # (ntime, nfreq)
            'phi':     jnp.array(                                       # (ntime, 2, nfreq)
                theta[:, :, 1:].transpose(0, 2, 1), dtype=DTYPE_R
            ),
        }
        loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}, self._effective_weights()))
        # Cache by object identity so that an immediate calc_loss(params) call
        # returns the same float without a second NUFFT evaluation.
        # jax_finufft.nufft is non-deterministic: two calls with identical inputs
        # can give slightly different values, causing large relative differences
        # when the loss is near zero.  Returning the cached value avoids this.
        self._fit_gains_linear_cache = (sky_coeffs, gain_params, loss)
        return gain_params, loss

    def _recompile_jit(self):
        """Recompile cached-beam JIT functions after precomputed arrays change."""
        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))
        self._jit_loss_variable_beam = jax.jit(self._loss_variable_beam)
        self._jit_simulate = jax.jit(self.fwd.simulate)

    def fit_beam_only(self, params, maxiter: int = 10, tol: float = 1e-7):
        """L-BFGS on beam_coeffs with sky and gains held fixed.

        Uses :meth:`~newnucal.simulate.ForwardModel.simulate_variable_beam` so
        the loss is differentiable w.r.t. *beam_coeffs* via JAX AD.  After the
        solve the ForwardModel's precomputed beam cache is updated and the
        calibrator's JIT functions are recompiled so subsequent sky/gain solves
        use the new beam.

        Parameters
        ----------
        params : dict
            Must include ``sky_coeffs``, ``beam_coeffs``, ``log_amp``, ``phase``,
            ``phi``.
        maxiter : int
        tol : float

        Returns
        -------
        params : dict — updated with new ``beam_coeffs``
        loss : float
        """
        sky_coeffs = params['sky_coeffs']
        gain_params = {k: params[k] for k in ('log_amp', 'phase', 'phi')}
        bls = self.bls

        def beam_loss(bc):
            vis_model = self.fwd.simulate_variable_beam(sky_coeffs, bc, self.rot_matrices)
            vis_cal = apply_gains(
                vis_model, gain_params['log_amp'], gain_params['phase'],
                gain_params['phi'], bls,
            )
            return self._weighted_chi2(self.data - vis_cal)

        solver = LBFGS(
            fun=beam_loss, tol=tol, maxiter=maxiter, verbose=False,
            linesearch='backtracking', increase_factor=5, history_size=10, jit=True,
        )
        result = solver.run(params['beam_coeffs'])
        new_beam = result.params

        # Update the precomputed cache so sky/gain solves see the new beam.
        self.fwd.update_beam_cache(new_beam)
        self._recompile_jit()

        new_params = {**params, 'beam_coeffs': new_beam}
        loss = float(beam_loss(new_beam))
        return new_params, loss

    def fit_beam_dirty(
        self,
        params,
        n_iter: int = 3,
        step_size: float = 0.5,
        sky_reg: float = 1e-3,
        verbose: bool = False,
    ):
        """Dirty-map beam update using explicit beam coefficients.

        This version avoids mutating the ForwardModel cache and recompiling JITs
        on every inner beam step.  Candidate beam states are evaluated through
        the explicit-beam forward path and the cache is updated only once at the
        end with the best accepted beam.
        """
        sky_coeffs  = params['sky_coeffs']
        gain_params = {k: params[k] for k in ('log_amp', 'phase', 'phi')}
        beam_coeffs = params['beam_coeffs']

        best_bc = beam_coeffs
        _w = self._effective_weights()
        best_loss = float(self._jit_loss_variable_beam(
            {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}, _w
        ))
        current_bc = beam_coeffs

        for i in range(n_iter):
            resid = self.calibrated_residual_variable_beam(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': current_bc, **gain_params}
            )
            delta_bc = self.fwd.accumulate_beam_update(
                sky_coeffs, resid, step_size=step_size, sky_reg=sky_reg,
            )
            trial_bc = current_bc + delta_bc
            loss = float(self._jit_loss_variable_beam(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': trial_bc, **gain_params}, _w
            ))
            if loss < best_loss:
                best_loss = loss
                best_bc = trial_bc
                current_bc = trial_bc
            else:
                if verbose:
                    print('    WARNING: beam dirty loss increased, reverting')
                break
            if verbose:
                print(f'    dirty beam iter {i:03d}: loss={loss:.4e}')

        # Synchronize cached beam once at the end.
        self.fwd.update_beam_cache(best_bc)
        self._recompile_jit()
        return {**params, 'beam_coeffs': best_bc}, best_loss

    def fit_sky_only(self, sky_coeffs, gain_params, maxiter: int = 30, tol: float = 1e-7):
        inv_gain_params = {
            'log_amp': -gain_params['log_amp'],
            'phase': -gain_params['phase'],
            'phi': -gain_params['phi'],
        }
        data_cal = apply_gains(self.data, inv_gain_params['log_amp'], inv_gain_params['phase'], inv_gain_params['phi'], self.bls)

        def sky_loss(sc):
            vis_model = self.fwd.simulate(sc, self.rot_matrices)
            return self._weighted_chi2(data_cal - vis_model)

        solver = LBFGS(fun=sky_loss, tol=tol, maxiter=maxiter, verbose=False, linesearch='backtracking', increase_factor=5, history_size=10, jit=True)
        result = solver.run(sky_coeffs)
        sky_new = result.params
        loss = float(self._jit_loss({'sky_coeffs': sky_new, **gain_params}, self._effective_weights()))
        return sky_new, loss

    def fit_sky_dirty(
        self,
        sky_coeffs,
        gain_params,
        n_iter: int = 3,
        step_size: float = 0.5,
        beam_reg: float = 1e-3,
        momentum: float = 0.0,
        verbose: bool = False,
    ):
        velocity = jnp.zeros_like(sky_coeffs)
        best_sky = sky_coeffs
        _w = self._effective_weights()
        best_loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}, _w))
        for i in range(n_iter):
            resid = self.calibrated_residual({'sky_coeffs': sky_coeffs, **gain_params})
            delta = self.fwd.accumulate_equatorial_sky_update(resid, step_size=step_size, beam_reg=beam_reg)
            velocity = momentum * velocity + delta
            sky_coeffs = sky_coeffs + velocity
            loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}, _w))
            if loss < best_loss:
                best_loss = loss
                best_sky = sky_coeffs
            else:
                print('    WARNING: loss increased')
            if verbose:
                print(f'    dirty iter {i:03d}: loss={loss:.4e}')
        return best_sky, best_loss

    def fit_with_rfi_reweighting(
        self,
        params,
        *,
        initial_channel_weights=None,
        initial_flags=None,
        inv_noise_var=None,
        n_rounds: int = 3,
        fit_method: str = 'fit_alternating_dirty',
        fit_kwargs: dict | None = None,
        rfi_config: RFIConfig | None = None,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
        verbose: bool = False,
    ):
        """Iteratively refit with residual-driven channel reweighting.

        The initial weights should come from an external DPSS/high-pass RFI
        detector.  After each fit round, residuals are turned into updated soft
        time/frequency channel weights and the model is refit.
        """
        fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
        cfg = rfi_config or RFIConfig()
        if inv_noise_var is not None:
            self.set_inv_noise_var(inv_noise_var)
        base_weights = prepare_initial_channel_weights(
            ntime=self.ntime,
            nfreq=self.nfreq,
            initial_weights=initial_channel_weights,
            initial_flags=initial_flags,
            flagged_weight=cfg.flagged_weight,
        )
        self.set_channel_weights(base_weights)

        history = []
        runner = getattr(self, fit_method)
        current_weights = np.array(base_weights, copy=True)
        for iround in range(n_rounds):
            params, loss = runner(params, **fit_kwargs)
            resid = np.asarray(self.calibrated_residual_variable_beam(params), dtype=np.complex64)
            new_weights, diagnostics = update_channel_weights_from_residuals(
                residual=resid,
                inv_noise_var=np.asarray(self.inv_noise_var),
                prior_weights=base_weights,
                current_weights=current_weights,
                config=cfg,
            )
            current_weights = new_weights
            self.set_channel_weights(current_weights)
            chi2_red = self.calc_reduced_chi2(params, explicit_beam=('beam_coeffs' in params), subtract_params='auto')
            history.append({
                'round': iround,
                'loss': float(loss),
                'reduced_chi2': float(chi2_red),
                'weights': current_weights.copy(),
                'diagnostics': diagnostics,
            })
            if verbose:
                flagged_frac = float(np.mean(current_weights < 0.5))
                print(f'[rfi round {iround:02d}] chi2={loss:.4e}  red_chi2={chi2_red:.3f}  frac(w<0.5)={flagged_frac:.3f}')
            if target_reduced_chi2 is not None and ((iround + 1) % max(reduced_chi2_check_every, 1) == 0):
                if chi2_red <= target_reduced_chi2:
                    if verbose:
                        print(f'[rfi round {iround:02d}] target reduced chi^2 reached.')
                    break
        return params, {'channel_weights': current_weights, 'history': history}



    def init_alternating_dirty_state(
        self,
        params,
        *,
        sky_step_size: float = 0.5,
        sky_beam_reg: float = 1e-3,
        sky_anderson_history: int = 0,
        sky_aa_start: int = 2,
        sky_aa_damping: float = 0.5,
        sky_aa_ridge: float = 1e-8,
        sky_aa_max_weight: float = 10.0,
        sky_lbfgs_maxiter: int = 5,
        beam_step_size: float = 0.5,
        beam_sky_reg: float = 1e-3,
        beam_anderson_history: int | None = None,
        beam_aa_start: int = 1,
        beam_aa_damping: float = 0.5,
        beam_aa_ridge: float = 1e-8,
        beam_aa_max_weight: float = 10.0,
        beam_lbfgs_maxiter: int = 5,
        solve_every: dict | None = None,
        eff_alpha: float = 0.4,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
    ):
        """Create a persistent state for resumable dirty alternating fits."""
        if solve_every is None:
            solve_every = {}
        if beam_anderson_history is None:
            beam_anderson_history = sky_anderson_history

        state = AlternatingDirtyFitState(
            params={
                'sky_coeffs': params['sky_coeffs'],
                'beam_coeffs': params.get('beam_coeffs', self.fwd.beam_coeffs),
                'log_amp': params['log_amp'],
                'phase': params['phase'],
                'phi': params['phi'],
            },
            settings=dict(
                sky_step_size=sky_step_size,
                sky_beam_reg=sky_beam_reg,
                sky_lbfgs_maxiter=sky_lbfgs_maxiter,
                beam_step_size=beam_step_size,
                beam_sky_reg=beam_sky_reg,
                beam_lbfgs_maxiter=beam_lbfgs_maxiter,
                solve_every=dict(solve_every),
                eff_alpha=eff_alpha,
                target_reduced_chi2=target_reduced_chi2,
                reduced_chi2_check_every=max(int(reduced_chi2_check_every), 1),
            ),
            sky_acc=AndersonAccelerator(
                sky_anderson_history, sky_aa_start, sky_aa_damping,
                sky_aa_ridge, sky_aa_max_weight
            ),
            beam_acc=AndersonAccelerator(
                beam_anderson_history, beam_aa_start, beam_aa_damping,
                beam_aa_ridge, beam_aa_max_weight
            ),
        )

        # Only update beam cache and recompile JIT when the beam actually changed.
        # update_beam_cache always creates a new fwd._jit_one, and _recompile_jit
        # creates several new JIT wrappers — unnecessary churn when beam is
        # unchanged fills JAX's LRU compilation cache, evicting previous compiled
        # functions and causing non-deterministic recompilation.
        new_bc = state.params['beam_coeffs']
        if not np.array_equal(np.asarray(new_bc), np.asarray(self.fwd.beam_coeffs)):
            self.fwd.update_beam_cache(new_bc)
            self._recompile_jit()
        state.loss = float(self._jit_loss(state.params, self._effective_weights()))
        if target_reduced_chi2 is not None:
            state.reduced_chi2 = self.calc_reduced_chi2(
                state.params, explicit_beam=True, subtract_params=0
            )
        return state

    def _sync_beam_cache_from_state(self, state: AlternatingDirtyFitState):
        self.fwd.update_beam_cache(state.params['beam_coeffs'])
        self._recompile_jit()
        state.beam_dirty_pending = False

    def _dirty_step_from_state(self, state: AlternatingDirtyFitState, verbose: bool = False):
        """Advance a persistent alternating-dirty state by one step."""
        if state.stop_reason is not None:
            return state

        s = state.settings
        gains_max_every = s['solve_every'].get('gains', 1)
        sky_max_every = s['solve_every'].get('sky_max', 0)
        beam_max_every = s['solve_every'].get('beam_max', 0)
        beam_lbfgs_max_every = s['solve_every'].get('beam_lbfgs_max', 25)
        gains_enabled = (gains_max_every != 0)
        beam_enabled = (s['solve_every'].get('beam', 1) != 0)
        sky_lbfgs_every = s['solve_every'].get('sky_lbfgs', 0)
        beam_lbfgs_every = s['solve_every'].get('beam_lbfgs', 0)

        sky_coeffs = state.params['sky_coeffs']
        beam_coeffs = state.params['beam_coeffs']
        gain_params = {k: state.params[k] for k in ('log_amp', 'phase', 'phi')}
        loss = float(state.loss)
        step = state.step

        def _full_params(sky, beam, gains):
            return {'sky_coeffs': sky, 'beam_coeffs': beam, **gains}

        def _sky_plain_step(sky, gains, current_loss):
            resid = self.calibrated_residual({'sky_coeffs': sky, **gains})
            delta = self.fwd.accumulate_equatorial_sky_update(
                resid, step_size=s['sky_step_size'], beam_reg=s['sky_beam_reg']
            )
            sky_trial = (sky + delta).astype(DTYPE_R)
            loss_trial = float(self._jit_loss({'sky_coeffs': sky_trial, **gains}, self._effective_weights()))
            if loss_trial < current_loss:
                return sky_trial, loss_trial
            return sky, current_loss

        def _beam_plain_step(sky, beam, gains, current_loss):
            resid = self.calibrated_residual_variable_beam(
                {'sky_coeffs': sky, 'beam_coeffs': beam, **gains}
            )
            delta_bc = self.fwd.accumulate_beam_update(
                sky, resid, step_size=s['beam_step_size'], sky_reg=s['beam_sky_reg']
            )
            beam_trial = (beam + delta_bc).astype(DTYPE_R)
            loss_trial = float(self._jit_loss_variable_beam(
                {'sky_coeffs': sky, 'beam_coeffs': beam_trial, **gains},
                self._effective_weights(),
            ))
            if loss_trial < current_loss:
                return beam_trial, loss_trial
            return beam, current_loss

        state.n_since_beam_lbfgs += 1
        overdue = {}
        if sky_max_every > 0 and state.n_since_sky >= sky_max_every:
            overdue['sky'] = state.n_since_sky / sky_max_every
        if beam_enabled and beam_max_every > 0 and state.n_since_beam >= beam_max_every:
            overdue['beam'] = state.n_since_beam / beam_max_every
        if gains_enabled and gains_max_every > 0 and state.n_since_gains >= gains_max_every:
            overdue['gains'] = state.n_since_gains / gains_max_every
        if beam_enabled and beam_lbfgs_max_every > 0 and state.n_since_beam_lbfgs >= beam_lbfgs_max_every:
            overdue['beam_lbfgs'] = state.n_since_beam_lbfgs / beam_lbfgs_max_every
        if (beam_enabled and beam_lbfgs_every > 0 and state.n_beam_since_beam_lbfgs > 0
                and state.n_beam_since_beam_lbfgs % beam_lbfgs_every == 0):
            overdue['beam_lbfgs'] = max(overdue.get('beam_lbfgs', 0.0), 1.0)

        if overdue:
            step_type = max(overdue, key=overdue.get)
        elif state.eff_gains is None and gains_enabled:
            step_type = 'gains'
        elif state.eff_sky is None:
            step_type = 'sky'
        elif state.eff_beam is None and beam_enabled:
            step_type = 'beam'
        else:
            cands = {'sky': state.eff_sky or 0.0}
            if beam_enabled:
                cands['beam'] = state.eff_beam or 0.0
            if gains_enabled:
                cands['gains'] = state.eff_gains or 0.0
            if beam_enabled and state.eff_beam_lbfgs is not None:
                cands['beam_lbfgs'] = state.eff_beam_lbfgs or 0.0
            step_type = max(cands, key=cands.get)

        do_sky = (step_type == 'sky')
        do_gains = (step_type == 'gains')
        do_beam_lbfgs = (step_type == 'beam_lbfgs')

        t_step = _time.perf_counter()
        loss_pre = loss

        if do_gains:
            if state.beam_dirty_pending:
                self._sync_beam_cache_from_state(state)
            sky_coeffs = state.params['sky_coeffs']
            gain_params, loss = self.fit_gains_linear(sky_coeffs)
            dt = max(_time.perf_counter() - t_step, 1e-3)
            dloss = max(0.0, loss_pre - loss)
            eff = dloss / (max(loss_pre, 1e-30) * dt)
            if state.eff_gains is None:
                state.eff_gains = eff
            else:
                ema = s['eff_alpha'] * eff + (1.0 - s['eff_alpha']) * state.eff_gains
                state.eff_gains = min(ema, eff)
            state.n_gains += 1
            state.n_since_gains = 0
            state.n_since_sky += 1
            state.n_since_beam += 1
            if verbose:
                print(f"    [gains {state.n_gains - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s")
        elif do_sky:
            if state.beam_dirty_pending:
                self._sync_beam_cache_from_state(state)
            sky_coeffs = state.params['sky_coeffs']
            sky_plain, loss_plain = _sky_plain_step(sky_coeffs, gain_params, loss)
            sky_next = sky_plain
            gain_next = gain_params
            loss_next = loss_plain
            used_aa = False
            cand_flat = state.sky_acc.push(
                np.asarray(sky_coeffs, dtype=np.float64).ravel(),
                np.asarray(sky_plain, dtype=np.float64).ravel(),
            )
            if cand_flat is not None:
                sky_cand = jnp.array(cand_flat.reshape(sky_coeffs.shape), dtype=DTYPE_R)
                gain_cand, _ = self.fit_gains_linear(sky_cand)
                loss_cand = float(self._jit_loss(_full_params(sky_cand, beam_coeffs, gain_cand), self._effective_weights()))
                if loss_cand < loss_plain:
                    sky_next = sky_cand
                    gain_next = gain_cand
                    loss_next = loss_cand
                    used_aa = True
            sky_coeffs = sky_next
            gain_params = gain_next
            loss = loss_next
            dt = max(_time.perf_counter() - t_step, 1e-3)
            dloss = max(0.0, loss_pre - loss)
            eff = dloss / (max(loss_pre, 1e-30) * dt)
            state.eff_sky = eff if state.eff_sky is None else s['eff_alpha'] * eff + (1.0 - s['eff_alpha']) * state.eff_sky
            state.n_sky += 1
            state.n_since_sky = 0
            state.n_since_beam += 1
            state.n_since_gains += 1
            if verbose:
                tag = 'AA' if used_aa else '  '
                print(f"    [sky {tag} {state.n_sky - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s")
        elif step_type == 'beam':
            beam_plain, loss_plain = _beam_plain_step(sky_coeffs, beam_coeffs, gain_params, loss)
            beam_next = beam_plain
            loss_next = loss_plain
            used_aa = False
            cand_flat = state.beam_acc.push(
                np.asarray(beam_coeffs, dtype=np.float64).ravel(),
                np.asarray(beam_plain, dtype=np.float64).ravel(),
            )
            if cand_flat is not None:
                beam_cand = jnp.array(cand_flat.reshape(beam_coeffs.shape), dtype=DTYPE_R)
                loss_cand = float(self._jit_loss_variable_beam(_full_params(sky_coeffs, beam_cand, gain_params), self._effective_weights()))
                if loss_cand < loss_plain:
                    beam_next = beam_cand
                    loss_next = loss_cand
                    used_aa = True
            beam_coeffs = beam_next
            loss = loss_next
            state.beam_dirty_pending = True
            dt = max(_time.perf_counter() - t_step, 1e-3)
            dloss = max(0.0, loss_pre - loss)
            eff = dloss / (max(loss_pre, 1e-30) * dt)
            state.eff_beam = eff if state.eff_beam is None else s['eff_alpha'] * eff + (1.0 - s['eff_alpha']) * state.eff_beam
            state.n_beam += 1
            state.n_since_sky += 1
            state.n_since_beam = 0
            state.n_since_gains += 1
            state.n_beam_since_beam_lbfgs += 1
            if verbose:
                tag = 'AA' if used_aa else '  '
                print(f"    [beam {tag} {state.n_beam - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s")
        elif do_beam_lbfgs:
            if state.beam_dirty_pending:
                self._sync_beam_cache_from_state(state)
            bp, loss = self.fit_beam_only(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                maxiter=s['beam_lbfgs_maxiter'], tol=1e-7,
            )
            beam_coeffs = bp['beam_coeffs']
            dt = max(_time.perf_counter() - t_step, 1e-3)
            dloss = max(0.0, loss_pre - loss)
            eff = dloss / (max(loss_pre, 1e-30) * dt)
            if state.eff_beam_lbfgs is None:
                state.eff_beam_lbfgs = eff
            else:
                ema = s['eff_alpha'] * eff + (1.0 - s['eff_alpha']) * state.eff_beam_lbfgs
                state.eff_beam_lbfgs = min(ema, eff)
            state.beam_acc.clear()
            state.eff_beam = None
            state.n_beam_lbfgs += 1
            state.n_since_beam_lbfgs = 0
            state.n_beam_since_beam_lbfgs = 0
            state.n_since_sky += 1
            state.n_since_beam += 1
            state.n_since_gains += 1
            if verbose:
                print(f"    [beam_lbfgs {state.n_beam_lbfgs - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s")

        if sky_lbfgs_every > 0 and (step + 1) % sky_lbfgs_every == 0:
            if state.beam_dirty_pending:
                self._sync_beam_cache_from_state(state)
            if verbose:
                print("    [sky_lbfgs]: starting L-BFGS sky solve ...")
            sky_coeffs, loss = self.fit_sky_only(
                sky_coeffs, gain_params, maxiter=s['sky_lbfgs_maxiter'], tol=1e-7
            )
            gain_params, _ = self.fit_gains_linear(sky_coeffs)
            state.sky_acc.clear()
            state.eff_sky = None
            if verbose:
                print(f"    [sky_lbfgs]: loss={loss:.4e}")

        state.params = {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}
        state.loss = float(loss)
        state.step += 1

        check_every = s['reduced_chi2_check_every']
        target = s['target_reduced_chi2']
        if target is not None and (state.step % check_every == 0):
            state.reduced_chi2 = self.calc_reduced_chi2(
                state.params, explicit_beam=True, subtract_params=0
            )
            if state.reduced_chi2 <= target:
                state.stop_reason = 'target_reduced_chi2'
        elif target is None:
            state.reduced_chi2 = None

        if verbose:
            elapsed = _time.perf_counter() - state.settings.setdefault('_t0', _time.perf_counter())
            eff_s = f'{state.eff_sky:.2e}' if state.eff_sky is not None else ' N/A  '
            eff_b = f'{state.eff_beam:.2e}' if state.eff_beam is not None else ' N/A  '
            eff_g = f'{state.eff_gains:.2e}' if state.eff_gains is not None else ' N/A  '
            eff_bp = f'{state.eff_beam_lbfgs:.2e}' if state.eff_beam_lbfgs is not None else ' N/A  '
            msg = (f"  step {state.step - 1:04d} [{step_type:<11}]: chi2={state.loss:.4e}"
                   f"  eff[sky={eff_s} beam={eff_b} gains={eff_g} lbfgs={eff_bp}]"
                   f"  t={elapsed:.1f}s")
            if state.reduced_chi2 is not None:
                msg += f"  red_chi2={state.reduced_chi2:.4f}"
            print(msg)

        return state

    def run_alternating_dirty_state(self, state: AlternatingDirtyFitState, n_iter: int = 1, verbose: bool = False, _stop_flag=None):
        """Advance a persistent alternating-dirty state by ``n_iter`` steps."""
        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None
        try:
            for _ in range(n_iter):
                if stop.stop:
                    state.stop_reason = 'user_stop'
                    break
                if state.stop_reason is not None:
                    break
                self._dirty_step_from_state(state, verbose=verbose)
                if state.stop_reason is not None:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return state

    def iter_alternating_dirty(self, state: AlternatingDirtyFitState, n_iter: int, yield_every: int = 1, verbose: bool = False, _stop_flag=None):
        """Yield a persistent state while advancing the fit, preserving Anderson history."""
        state = self.run_alternating_dirty_state(state, 0, verbose=False, _stop_flag=_stop_flag)
        for i in range(n_iter):
            state = self.run_alternating_dirty_state(state, 1, verbose=verbose, _stop_flag=_stop_flag)
            if ((i + 1) % max(int(yield_every), 1) == 0) or state.stop_reason is not None:
                yield state
            if state.stop_reason is not None:
                break


    def fit_alternating_dirty(
        self,
        params,
        n_iter: int = 30,
        sky_step_size: float = 0.5,
        sky_beam_reg: float = 1e-3,
        sky_anderson_history: int = 0,
        sky_aa_start: int = 2,
        sky_aa_damping: float = 0.5,
        sky_aa_ridge: float = 1e-8,
        sky_aa_max_weight: float = 10.0,
        sky_lbfgs_maxiter: int = 5,
        beam_step_size: float = 0.5,
        beam_sky_reg: float = 1e-3,
        beam_anderson_history: int | None = None,
        beam_aa_start: int = 1,
        beam_aa_damping: float = 0.5,
        beam_aa_ridge: float = 1e-8,
        beam_aa_max_weight: float = 10.0,
        beam_lbfgs_maxiter: int = 5,
        solve_every: dict | None = None,
        eff_alpha: float = 0.4,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Adaptive alternating dirty-map minimisation.

        One-shot wrapper around the resumable state-machine interface.
        Use :meth:`init_alternating_dirty_state`, :meth:`run_alternating_dirty_state`,
        and :meth:`iter_alternating_dirty` to pause and resume fits while preserving
        Anderson histories, efficiency estimates, and cadence counters.
        """
        state = self.init_alternating_dirty_state(
            params,
            sky_step_size=sky_step_size,
            sky_beam_reg=sky_beam_reg,
            sky_anderson_history=sky_anderson_history,
            sky_aa_start=sky_aa_start,
            sky_aa_damping=sky_aa_damping,
            sky_aa_ridge=sky_aa_ridge,
            sky_aa_max_weight=sky_aa_max_weight,
            sky_lbfgs_maxiter=sky_lbfgs_maxiter,
            beam_step_size=beam_step_size,
            beam_sky_reg=beam_sky_reg,
            beam_anderson_history=beam_anderson_history,
            beam_aa_start=beam_aa_start,
            beam_aa_damping=beam_aa_damping,
            beam_aa_ridge=beam_aa_ridge,
            beam_aa_max_weight=beam_aa_max_weight,
            beam_lbfgs_maxiter=beam_lbfgs_maxiter,
            solve_every=solve_every,
            eff_alpha=eff_alpha,
            target_reduced_chi2=target_reduced_chi2,
            reduced_chi2_check_every=reduced_chi2_check_every,
        )
        state = self.run_alternating_dirty_state(state, n_iter=n_iter, verbose=verbose, _stop_flag=_stop_flag)
        return state.params, float(state.loss)


    def fit_alternating(
        self,
        params,
        n_iter: int = 10,
        sky_maxiter: int = 10,
        sky_tol: float = 1e-7,
        solve_every: dict | None = None,
        beam_maxiter: int = 10,
        beam_tol: float = 1e-7,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Alternating L-BFGS sky / linear-gain minimisation.

        Parameters
        ----------
        params : dict with keys ``sky_coeffs``, ``beam_coeffs``, ``log_amp``,
            ``phase``, ``phi``
        n_iter : int
            Number of iterations.  The sky is updated (L-BFGS) every
            iteration.
        solve_every : dict or None
            Controls how often non-sky components are updated.  Keys:

            ``'gains'`` (int, default 1)
                Solve gains every *N* iterations.
            ``'beam'`` (int, default 0)
                Solve beam every *N* iterations using
                :meth:`fit_beam_only`.  ``0`` (default) disables beam solving.

        Returns
        -------
        params : dict
            Updated parameters with keys ``sky_coeffs``, ``beam_coeffs``,
            ``log_amp``, ``phase``, ``phi``.
        loss : float
        """
        if solve_every is None:
            solve_every = {}
        gains_every = solve_every.get('gains', 1)
        beam_every  = solve_every.get('beam',  0)

        sky_coeffs  = params['sky_coeffs']
        gain_params = {k: params[k] for k in ('log_amp', 'phase', 'phi')}
        beam_coeffs = params.get('beam_coeffs', self.fwd.beam_coeffs)
        self.fwd.update_beam_cache(beam_coeffs)
        self._recompile_jit()

        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None
        try:
            loss = float(self._jit_loss(params, self._effective_weights()))
            _t0 = _time.perf_counter()
            for i in range(n_iter):
                # --- Sky L-BFGS update (every step) ---
                if verbose:
                    print(f'    [sky]: starting L-BFGS sky solve ...')
                sky_coeffs, _sl = self.fit_sky_only(
                    sky_coeffs, gain_params, maxiter=sky_maxiter, tol=sky_tol
                )
                if verbose:
                    print(f'    [sky]: loss={_sl:.4e}')

                # --- Gain solve (if scheduled) ---
                if gains_every > 0 and (i + 1) % gains_every == 0:
                    gain_params, loss = self.fit_gains_linear(sky_coeffs)
                    if verbose:
                        print(f'    [gains]: loss={loss:.4e}')
                else:
                    loss = float(self._jit_loss(
                        {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                        self._effective_weights(),
                    ))

                # --- Beam solve (if scheduled) ---
                if beam_every > 0 and (i + 1) % beam_every == 0:
                    if verbose:
                        print(f'    [beam]: starting L-BFGS beam solve ...')
                    bp, loss = self.fit_beam_only(
                        {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                        maxiter=beam_maxiter, tol=beam_tol,
                    )
                    beam_coeffs = bp['beam_coeffs']
                    if verbose:
                        print(f'    [beam]: loss={loss:.4e}')

                if verbose:
                    elapsed = _time.perf_counter() - _t0
                    msg = f'  iter {i:04d}: chi2={loss:.4e}  t={elapsed:.1f}s'
                    if target_reduced_chi2 is not None and ((i + 1) % max(reduced_chi2_check_every, 1) == 0):
                        chi2r = self.calc_reduced_chi2(
                            {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                            explicit_beam=True,
                            subtract_params='auto',
                        )
                        msg += f'  red_chi2={chi2r:.4f}'
                    print(msg)
                if target_reduced_chi2 is not None and ((i + 1) % max(reduced_chi2_check_every, 1) == 0):
                    chi2r = self.calc_reduced_chi2(
                        {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                        explicit_beam=True,
                        subtract_params='auto',
                    )
                    if chi2r <= target_reduced_chi2:
                        if verbose:
                            print('  target reduced chi^2 reached.')
                        break
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}, loss

    # ------------------------------------------------------------------
    # Grid-based fast pre-optimizer
    # ------------------------------------------------------------------

    def init_params_from_grid_fit(
        self,
        grid_fitter,
        n_iter: int = 40,
        beam_model=None,
        beam_reg: float = 1e-3,
        verbose: bool = False,
        **fit_kwargs,
    ):
        """Initialise sky and gain parameters from a fast per-time grid fit.

        Runs :meth:`~newnucal.grid_fitter.GridFitter.fit_all_times` on
        ``self.data``, converts the per-time gains to the Calibrator 4-DOF
        format, accumulates the per-time product maps on an equatorial
        HEALPix grid, divides by the beam, and projects onto the DPSS sky
        basis.

        Parameters
        ----------
        grid_fitter : GridFitter
        n_iter : int
            Iterations per time step in the grid fit.
        beam_model : BeamModel or None
            If provided, the accumulated product is divided by the beam
            before projecting onto the sky DPSS basis.  ``None`` uses
            ``self.fwd.beam_coeffs`` to construct a beam estimate.
        beam_reg : float
        verbose : bool
        **fit_kwargs
            Extra keyword arguments forwarded to :meth:`GridFitter.fit_all_times`.

        Returns
        -------
        params : dict
            ``sky_coeffs``, ``log_amp``, ``phase``, ``phi`` ready to pass
            to any Calibrator optimiser.
        """
        import jax.numpy as jnp

        data_np = np.asarray(self.data)   # (ntime, nfreq, nbl)

        coeff_maps_all, gain_params_all, _ = grid_fitter.fit_all_times(
            data_np, n_iter=n_iter, verbose=verbose, **fit_kwargs
        )

        # Per-time 4-DOF gains → Calibrator format (add time axis)
        gain_params = {
            'log_amp': jnp.array(gain_params_all['log_amp'], dtype=jnp.float32),
            'phase':   jnp.array(gain_params_all['phase'],   dtype=jnp.float32),
            'phi':     jnp.array(gain_params_all['phi'],     dtype=jnp.float32),
        }

        # Accumulate product on equatorial HEALPix
        sky_nside = self.fwd.sky_nside
        product_map, _weights = grid_fitter.accumulate_product_healpix(
            coeff_maps_all,
            np.asarray(self.rot_matrices),
            sky_nside=sky_nside,
        )

        # Resolve beam model for beam division
        if beam_model is None:
            from .beam import BeamModel as _BeamModel
            # Build a minimal BeamModel proxy carrying the same coeffs/A_beam
            bm_proxy = object.__new__(_BeamModel)
            bm_proxy.nside   = self.fwd.beam_nside
            bm_proxy.freqs   = np.asarray(self.freqs)
            bm_proxy.coeffs  = np.asarray(self.fwd.beam_coeffs)
            bm_proxy.A_beam  = np.asarray(self.fwd.A_beam)
            beam_model = bm_proxy

        sky_coeffs = grid_fitter.init_sky_coeffs_from_product(
            product_map,
            A_sky=np.asarray(self.A_sky),
            beam_model=beam_model,
            beam_reg=beam_reg,
        )

        return {'sky_coeffs': sky_coeffs, **gain_params}

    def fit_grid_fast(
        self,
        params,
        grid_fitter,
        n_iter: int = 40,
        beam_model=None,
        beam_reg: float = 1e-3,
        verbose: bool = False,
        **fit_kwargs,
    ):
        """Pre-optimise using the fast grid-based solver, then return params.

        Equivalent to calling :meth:`init_params_from_grid_fit` with the
        current params' gain values as the initialisation, but also accepts
        and returns a ``params`` dict so it can be scheduled like other
        optimisers (``fit_alternating``, ``fit_alternating_dirty``, …).

        Parameters
        ----------
        params : dict
            Current parameters (gain init is extracted from this dict).
        grid_fitter : GridFitter
        n_iter : int
        beam_model : BeamModel or None
        beam_reg : float
        verbose : bool
        **fit_kwargs
            Forwarded to :meth:`GridFitter.fit_all_times`.

        Returns
        -------
        params : dict
        loss  : float
        """
        gain_init = {
            'log_amp': np.asarray(params['log_amp']),   # (ntime, nfreq)
            'phase':   np.asarray(params['phase']),
            'phi':     np.asarray(params['phi']),
        }
        new_params = self.init_params_from_grid_fit(
            grid_fitter,
            n_iter=n_iter,
            beam_model=beam_model,
            beam_reg=beam_reg,
            verbose=verbose,
            gain_params_init=gain_init,
            **fit_kwargs,
        )
        if 'beam_coeffs' in params:
            new_params['beam_coeffs'] = params['beam_coeffs']
        loss = float(self._jit_loss(new_params, self._effective_weights()))
        return new_params, loss

    def fit_lbfgs(self, params, maxiter: int = 30, tol: float = 1e-7):
        w = self._effective_weights()
        solver = LBFGS(fun=lambda p: self._loss(p, w), tol=tol, maxiter=maxiter, verbose=False, linesearch='backtracking', increase_factor=5, history_size=10, jit=True)
        result = solver.run(params)
        return result.params, float(result.state.value)

    def fit_optax(self, params, maxiter: int = 60, lr: float = 3e-2, opt_type=None, verbose: bool = False, **opt_kwargs):
        if opt_type is None:
            opt_type = optax.adam
        optimizer = opt_type(learning_rate=lr, **opt_kwargs)
        opt_state = optimizer.init(params)
        best_params = params
        best_loss = jnp.inf
        for i in range(maxiter):
            loss, grads = self._jit_val_grad(params, self._effective_weights())
            if loss < best_loss:
                best_loss = loss
                best_params = params
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if verbose:
                print(f'  iter {i:04d}: loss={float(loss):.4e}')
        return best_params, float(best_loss)

    @property
    def nfreq(self) -> int:
        return int(self.freqs.shape[0])

    @property
    def ntime(self) -> int:
        return int(self.rot_matrices.shape[0])

    @property
    def nbls(self) -> int:
        return int(self.data.shape[2])

    @property
    def npix_beam(self) -> int:
        return self.fwd.npix_beam

    @property
    def nmodes_beam(self) -> int:
        return self.fwd.nmodes_beam
