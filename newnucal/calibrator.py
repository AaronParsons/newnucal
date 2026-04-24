"""
Calibrator — ties together ForwardModel, gain parameters, and optimisation.
"""

import signal as _signal
import time as _time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import healpy

from .array import HERAArray
from .beam import BeamModel
from .basis import basis_project
from .simulate import ForwardModel
from .gains import apply_gains, init_gain_params
from .rfi import RFIConfig, prepare_initial_channel_weights, update_channel_weights_from_residuals
from .utils import DTYPE_R_JAX, DTYPE_R_NPY, DTYPE_C_JAX

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
    """

    def __init__(
        self,
        history: int,
        start: int = 2,
        damping: float = 0.5,
        ridge: float = 1e-4,
        step_gain_max: float = 20.0,
        min_step_gain: float = 0.1,
        step_gain_factor: float = 2.0,
    ):
        self.history    = history
        self.start      = start
        self.damping    = damping
        self.ridge      = ridge
        self._step_gain_max = step_gain_max
        self._min_step_gain = min_step_gain
        self._step_gain_factor = step_gain_factor
        self.step_gain = 1.0
        self._hist_g: deque[np.ndarray] = deque(maxlen=history)
        self._hist_f: deque[np.ndarray] = deque(maxlen=history)
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
            coefficients fail validity checks (non-finite beta values).
        """
        if self.history == 0:
            return None

        f_flat = g_flat - x_flat
        self._hist_g.append(g_flat.copy())
        self._hist_f.append(f_flat.copy())

        candidate = None
        if self._step >= self.start and len(self._hist_f) >= 2:
            F    = np.stack(self._hist_f, axis=0)
            beta = self._solve_coeffs(F, self.ridge)
            if beta is not None and np.all(np.isfinite(beta)):
                g_aa = np.tensordot(beta, np.stack(self._hist_g, axis=0), axes=1)
                candidate = (1.0 - self.damping) * g_flat + self.damping * g_aa
            else:
                # Gram ill-conditioned or non-finite beta: boost step_gain and clear history
                self.step_gain = min(self.step_gain * self._step_gain_factor, self._step_gain_max)
                self.clear(reset_step_gain=False)

        self._step += 1
        return candidate

    def clear(self, reset_step_gain: bool = True):
        """Reset history and step counter.

        Parameters
        ----------
        reset_step_gain : bool
            If True (default), also reset step_gain to 1.0. Set to False when
            clearing history as part of step_gain adjustment.
        """
        self._hist_g.clear()
        self._hist_f.clear()
        self._step = 0
        if reset_step_gain:
            self.step_gain = 1.0

    def report_step_gain(self, eff_gain: float):
        """Persist the effective step gain used by the caller.

        Parameters
        ----------
        eff_gain : float
            Step gain multiplier that was successfully used (or 0 if no improvement).
        """
        if eff_gain > 0:
            self.step_gain = eff_gain
        else:
            self.step_gain = max(self.step_gain / self._step_gain_factor, self._min_step_gain)

    @staticmethod
    def _solve_coeffs(F: np.ndarray, ridge: float = 1e-8) -> np.ndarray | None:
        """Constrained least-squares mixing coefficients.

        Solves ``min ||F^T beta||`` subject to ``sum(beta) = 1`` with
        Tikhonov regularisation ``ridge * trace(G) * I`` added to the
        Gram matrix ``G = F F^T``.

        Returns None if Gram matrix is ill-conditioned (eigenvalue ratio > 1/ridge),
        signalling that larger plain steps should be tried.

        Parameters
        ----------
        F : np.ndarray, shape (m, n)
            Residual matrix whose rows are past residuals ``f_i = g_i - x_i``.

        Returns
        -------
        beta : np.ndarray, shape (m,) or None
        """
        n = F.shape[0]
        assert n >= 2  # only call when len(hist_f) >= 2
        gram = F @ F.T
        eigs = np.linalg.eigvalsh(gram)  # ascending order: eigs[-1] is largest
        eig_max = eigs[-1]
        # Check conditioning: if largest eigenvalue >> second-largest, skip AA
        eig_second = eigs[-2]
        if eig_max * ridge > eig_second:
            return None  # ill-conditioned: signal that larger steps are needed

        gram = gram + ridge * eig_max * np.eye(n, dtype=gram.dtype)
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

    n_sky: int = 0
    n_beam: int = 0
    n_gains: int = 0

    n_since_sky: int = 0
    n_since_beam: int = 0
    n_since_gains: int = 0

    beam_dirty_pending: bool = False
    stop_reason: str | None = None
    subtract_static_sky: bool = False  # Whether to subtract cached static sky from data


@dataclass
class JointSkyBeamDirtyFitState:
    """Persistent state for resumable fit_joint_sky_beam_dirty runs."""
    params: dict
    settings: dict = field(default_factory=dict)
    loss: float | None = None
    reduced_chi2: float | None = None
    step: int = 0

    joint_acc: AndersonAccelerator | None = None

    eff_joint: float | None = None
    eff_gains: float | None = None

    n_joint: int = 0
    n_gains: int = 0

    n_since_joint: int = 0
    n_since_gains: int = 0

    stop_reason: str | None = None
    subtract_static_sky: bool = False


class Calibrator:
    _GAIN_PARAM_KEYS = ('log_amp', 'phase', 'phi')

    def __init__(
        self,
        array: HERAArray,
        beam_model: BeamModel,
        sky_model,
        freqs,
        rot_matrices,
        data,
        eps: float = 1e-6,
        eta_max: float | None = None,
        eta_padding: float = 0.0,
        channel_weights=None,
        inv_noise_var=None,
        noise_sigma=None,
        method: str = '3d',
    ):
        """
        Parameters
        ----------
        sky_model : SkyModel
            Sky model providing ``nside`` and the spectral basis ``A_sky``.
        method : {'3d', '2d'}
            Forward-model path.  ``'3d'`` uses a single type-3 NUFFT per time
            step (default).  ``'2d'`` uses per-frequency type-1 2D NUFFTs on
            the compact hex-rect grid — typically 10–30× faster when the
            channel spacing satisfies the critical Nyquist condition.
        """
        if method not in ('3d', '2d'):
            raise ValueError(f"method must be '3d' or '2d', got {method!r}")
        self.method = method

        self.freqs = jnp.array(freqs, dtype=DTYPE_R_JAX)
        self.rot_matrices = jnp.array(rot_matrices, dtype=DTYPE_R_JAX)
        self.data = jnp.array(data, dtype=DTYPE_C_JAX)
        self.bls = jnp.array(array.bls, dtype=DTYPE_R_JAX)
        self.channel_weights = jnp.ones(self.data.shape[:2], dtype=DTYPE_R_JAX)
        self.inv_noise_var = jnp.ones(self.data.shape[:2], dtype=DTYPE_R_JAX)

        self.A_sky = np.asarray(sky_model.A, dtype=DTYPE_R_NPY)  # (nfreq, nmodes)
        sky_nside  = sky_model.nside
        self.beam_model = beam_model

        self.fwd = ForwardModel(
            array,
            sky_model,
            beam_model,
            freqs,
            eps=eps,
            eta_max=eta_max,
            eta_padding=eta_padding,
        )
        self.fwd.precompute_time_geometry(self.rot_matrices)

        self.pixel_mask = None  # No mask initially
        self.beam_mask = None   # No beam mask initially
        self._npix_full = self.fwd.npix_sky  # Store original full-sky size (before any masking)
        self._active_size = self._npix_full   # cached; updated by apply_sky_mask
        self._pixel_indices = None            # int32 indices of active pixels; None means all
        self._cached_static_vis = None        # Cached visibility of static (unmasked) pixels
        self._static_sky_cached = False       # Whether static sky contribution is cached

        self._select_methods()
        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))
        self._jit_loss_variable_beam = jax.jit(self._loss_variable_beam)
        self._jit_simulate = jax.jit(self._sim_fn)
        self.set_channel_weights(channel_weights)
        self.set_inv_noise_var(inv_noise_var)
        if noise_sigma is not None:
            self.set_noise_sigma(noise_sigma)

        # Warn if there are permanently below-horizon pixels
        altitude_mask = self.build_sky_mask_altitude(min_altitude_deg=0.0)
        npix_below_horizon = int((~altitude_mask).sum())
        if npix_below_horizon > 0:
            npix_above_horizon = int(altitude_mask.sum())
            print(f"  WARNING: {npix_below_horizon} sky pixels are never above the horizon. "
                  f"Call cal.apply_sky_mask(cal.build_sky_mask_altitude(...)) to remove them before fitting.")
            print(f"  ({npix_above_horizon} above-horizon pixels retained after cut)")

    def _select_methods(self):
        """Bind forward/adjoint callables based on self.method."""
        if self.method == '2d':
            self._sim_fn            = self.fwd.simulate_2d
            self._var_beam_sim_fn   = self.fwd.simulate_variable_beam_2d
            self._sky_update_fn     = self.fwd.accumulate_equatorial_sky_update_2d
            self._beam_update_fn    = self.fwd.accumulate_beam_update_2d
            self._combined_update_fn = self.fwd.accumulate_sky_and_beam_update_2d
        else:
            self._sim_fn            = self.fwd.simulate
            self._var_beam_sim_fn   = self.fwd.simulate_variable_beam
            self._sky_update_fn     = self.fwd.accumulate_equatorial_sky_update
            self._beam_update_fn    = self.fwd.accumulate_beam_update
            self._combined_update_fn = self.fwd.accumulate_sky_and_beam_update

    def _get_active_size(self):
        """Return number of active pixels after masking."""
        return self._active_size

    def _ensure_sky_is_active(self, sky_coeffs):
        """Convert full-sky to active pixels if needed, return active sky."""
        if sky_coeffs.shape[0] == self._npix_full:
            if self._pixel_indices is not None:
                return sky_coeffs[self._pixel_indices]
            return sky_coeffs
        elif sky_coeffs.shape[0] == self._active_size:
            return sky_coeffs
        else:
            raise ValueError(
                f"sky_coeffs size {sky_coeffs.shape[0]} doesn't match "
                f"full {self._npix_full} or active {self._active_size}"
            )

    def _params_to_active_space(self, params):
        """Convert parameter dict from full-sky to active pixels if masked.

        Only sky_coeffs are masked; beam_coeffs are always full-size.
        """
        out = {}
        if 'sky_coeffs' in params and params['sky_coeffs'] is not None:
            out['sky_coeffs'] = self._ensure_sky_is_active(params['sky_coeffs'])
        else:
            out['sky_coeffs'] = params.get('sky_coeffs')

        # Beam coeffs are never masked (independent of sky pixel mask)
        out['beam_coeffs'] = params.get('beam_coeffs')

        for key in self._GAIN_PARAM_KEYS:
            out[key] = params.get(key)
        return out

    def _params_to_full_space(self, params_active, params_input_full=None):
        """Convert parameter dict from active space back to full-sky.

        Only sky_coeffs are expanded to full-sky; beam_coeffs are never masked.
        Unsolved sky pixels keep their original values from params_input_full.
        """
        if self.pixel_mask is None:
            return params_active

        out = {}

        # Expand sky_coeffs to full-sky
        if 'sky_coeffs' in params_active and params_active['sky_coeffs'] is not None:
            val_active = np.asarray(params_active['sky_coeffs'])
            nmodes = val_active.shape[1] if val_active.ndim > 1 else 1
            val_full = np.zeros((self._npix_full, nmodes), dtype=val_active.dtype)
            if params_input_full is not None and 'sky_coeffs' in params_input_full:
                orig = params_input_full['sky_coeffs']
                if orig is not None and orig.shape[0] == self._npix_full:
                    val_full[:] = np.asarray(orig)
            val_full[self._pixel_indices] = val_active
            out['sky_coeffs'] = jnp.array(val_full, dtype=DTYPE_R_JAX)
        else:
            out['sky_coeffs'] = params_active.get('sky_coeffs')

        # Beam coeffs are never masked (pass through as-is)
        out['beam_coeffs'] = params_active.get('beam_coeffs')

        for key in self._GAIN_PARAM_KEYS:
            out[key] = params_active.get(key)
        return out

    def _effective_weights(self):
        return (self.channel_weights * self.inv_noise_var).astype(DTYPE_R_JAX)

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
        sky_coeffs = self._ensure_sky_is_active(params['sky_coeffs'])
        vis_model = self._sim_fn(sky_coeffs, self.rot_matrices)
        vis_cal = apply_gains(vis_model, params['log_amp'], params['phase'], params['phi'], self.bls)
        return jnp.sum(weights[:, :, None] * (jnp.abs(self.data - vis_cal) ** 2))

    def _loss_variable_beam(self, params, weights):
        """Loss function (variable beam) with weights passed explicitly."""
        sky_coeffs = self._ensure_sky_is_active(params['sky_coeffs'])
        vis_model = self._var_beam_sim_fn(
            sky_coeffs, params['beam_coeffs'], self.rot_matrices
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
            arr = jnp.ones(self.data.shape[:2], dtype=DTYPE_R_JAX)
        else:
            arr = jnp.array(channel_weights, dtype=DTYPE_R_JAX)
            if arr.ndim == 1:
                arr = jnp.broadcast_to(arr[None, :], self.data.shape[:2])
            elif arr.shape != self.data.shape[:2]:
                raise ValueError(
                    f"channel_weights must have shape {(self.ntime, self.nfreq)} "
                    f"or {(self.nfreq,)}, got {arr.shape}"
                )
        self.channel_weights = jnp.clip(arr, 0.0, 1.0).astype(DTYPE_R_JAX)

    def set_inv_noise_var(self, inv_noise_var=None):
        """Set per-time/per-frequency inverse noise variance."""
        if inv_noise_var is None:
            arr = jnp.ones(self.data.shape[:2], dtype=DTYPE_R_JAX)
        else:
            arr = jnp.array(inv_noise_var, dtype=DTYPE_R_JAX)
            if arr.ndim == 1:
                arr = jnp.broadcast_to(arr[None, :], self.data.shape[:2])
            elif arr.shape != self.data.shape[:2]:
                raise ValueError(
                    f"inv_noise_var must have shape {(self.ntime, self.nfreq)} "
                    f"or {(self.nfreq,)}, got {arr.shape}"
                )
        self.inv_noise_var = jnp.clip(arr, 0.0, jnp.inf).astype(DTYPE_R_JAX)

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
        arr = jnp.array(noise_sigma, dtype=DTYPE_R_JAX)
        if arr.ndim == 1:
            arr = jnp.broadcast_to(arr[None, :], self.data.shape[:2])
        elif arr.shape != self.data.shape[:2]:
            raise ValueError(
                f"noise_sigma must have shape {(self.ntime, self.nfreq)} "
                f"or {(self.nfreq,)}, got {arr.shape}"
            )
        arr = jnp.clip(arr, 1e-20, jnp.inf)
        self.inv_noise_var = (1.0 / (arr ** 2)).astype(DTYPE_R_JAX)

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
        npix_sky = self._npix_full
        nmodes_sky = self.A_sky.shape[1]
        return {
            'sky_coeffs':  jnp.zeros((npix_sky, nmodes_sky), dtype=DTYPE_R_JAX),
            'beam_coeffs': jnp.array(self.fwd.beam_coeffs),
            **init_gain_params(self.ntime, self.nfreq),
        }

    def init_sky_from_flux(self, flux):
        return jnp.array(basis_project(np.asarray(flux), self.A_sky), dtype=DTYPE_R_JAX)

    def simulate(self, params):
        """Return gain-calibrated model visibilities for the given parameters.

        Parameters
        ----------
        params : dict
            Must include ``sky_coeffs``, ``log_amp``, ``phase``, ``phi``.
            If ``beam_coeffs`` is present, ``simulate_variable_beam`` is used;
            otherwise the precomputed beam cache is used.
            sky_coeffs can be full-sky or masked format.

        Returns
        -------
        vis : jnp.ndarray, shape (ntime, nfreq, nbls), complex64
        """
        sky_coeffs = self._ensure_sky_is_active(params['sky_coeffs'])
        if 'beam_coeffs' in params:
            vis_model = self._var_beam_sim_fn(
                sky_coeffs, params['beam_coeffs'], self.rot_matrices
            )
        else:
            vis_model = self._sim_fn(sky_coeffs, self.rot_matrices)
        return apply_gains(
            vis_model, params['log_amp'], params['phase'], params['phi'], self.bls
        )

    def _invert_gains(self, gain_params):
        """Return gain parameters with negated log_amp, phase, and phi."""
        return {k: -gain_params[k] for k in self._GAIN_PARAM_KEYS}

    def calibrated_residual(self, params):
        inv = self._invert_gains(params)
        data_cal = apply_gains(self.data, inv['log_amp'], inv['phase'], inv['phi'], self.bls)
        sky_coeffs = self._ensure_sky_is_active(params['sky_coeffs'])
        vis_model = self._sim_fn(sky_coeffs, self.rot_matrices)
        resid = data_cal - vis_model
        return resid * (self.channel_weights * self.inv_noise_var)[:, :, None].astype(DTYPE_C_JAX)

    def calibrated_residual_variable_beam(self, params):
        inv = self._invert_gains(params)
        data_cal = apply_gains(self.data, inv['log_amp'], inv['phase'], inv['phi'], self.bls)
        sky_coeffs = self._ensure_sky_is_active(params['sky_coeffs'])
        vis_model = self._var_beam_sim_fn(
            sky_coeffs, params['beam_coeffs'], self.rot_matrices
        )
        resid = data_cal - vis_model
        return resid * (self.channel_weights * self.inv_noise_var)[:, :, None].astype(DTYPE_C_JAX)

    # ------------------------------------------------------------------
    # Pixel-cut helpers
    # ------------------------------------------------------------------

    def apply_sky_mask(self, mask):
        """Apply an arbitrary pixel mask controlling which pixels are solved.

        Pixels where mask is True will be simulated and solved for.
        Pixels where mask is False are left at their original values.

        Note: Masks are permanent and cannot be removed; create a new Calibrator
        if you need to work with different masks.

        Parameters
        ----------
        mask : array_like, shape (npix_full,)
            Boolean mask of pixels to keep.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self._npix_full:
            raise ValueError(
                f"mask size {mask.shape[0]} != full sky npix_full {self._npix_full}"
            )

        self.pixel_mask = mask.copy()
        self._active_size = int(mask.sum())
        self._pixel_indices = np.where(mask)[0].astype(np.int32)
        self.fwd.apply_sky_mask(mask)
        self.fwd.precompute_time_geometry(self.rot_matrices)
        self._recompile_jit()

        n_keep = int(mask.sum())
        n_drop = int((~mask).sum())
        print(f'  Pixel mask: {n_keep} pixels retained, {n_drop} removed.')


    def build_sky_mask_altitude(self, min_altitude_deg=0.0):
        """Build mask for sky pixels above minimum altitude.

        Computes the maximum altitude reached by each sky pixel across all times,
        and returns a mask of pixels that reach the specified minimum altitude.

        Parameters
        ----------
        min_altitude_deg : float, optional
            Minimum altitude in degrees. Default 0 (horizon).
            Use 0 to remove permanently below-horizon pixels.

        Returns
        -------
        np.ndarray, dtype bool, shape (npix_full,)
            Boolean mask where True indicates pixels above the altitude threshold.
        """
        return self.fwd.build_sky_mask_altitude(self.rot_matrices, min_altitude_deg)

    def build_beam_mask_altitude(self, max_zenith_angle_deg=90.0):
        """Build mask for beam pixels within zenith angle limit.

        Masks beam pixels to only those within the specified angular distance from
        zenith. The beam is fixed in the topocentric frame.

        Parameters
        ----------
        max_zenith_angle_deg : float, optional
            Maximum zenith angle in degrees. Default 90 (all sky above horizon).
            Use 90 for all above-horizon pixels, lower values for narrower beams.

        Returns
        -------
        np.ndarray, dtype bool, shape (npix_beam_full,)
            Boolean mask where True indicates pixels within zenith angle threshold.
        """
        return self.fwd.build_beam_mask_altitude(self.rot_matrices, max_zenith_angle_deg)

    def build_sky_mask_from_beam_pixels(self, beam_pixel_mask):
        """Build mask for sky pixels illuminated by selected beam pixels.

        Given a mask of beam pixels, returns a mask of sky pixels whose
        HEALPix interpolation neighborhood touches at least one of the
        selected beam pixels. This is the reverse operation of selecting
        beam pixels that touch a given set of sky pixels.

        Parameters
        ----------
        beam_pixel_mask : array_like, shape (npix_beam_full,), dtype bool
            Boolean mask of beam pixels to consider. Sky pixels are selected if
            their interpolation stencil touches any beam pixel where mask is True.

        Returns
        -------
        sky_mask : np.ndarray, shape (npix_full,), dtype bool
            True for sky pixels whose interpolation touches selected beam pixels.

        Example
        -------
        Select beam pixels at high altitude, then find which sky pixels they illuminate:

            beam_mask = cal.build_beam_mask_altitude(max_zenith_angle_deg=45.0)
            sky_mask = cal.build_sky_mask_from_beam_pixels(beam_mask)
            cal.apply_sky_mask(sky_mask)
        """
        return self.fwd.build_sky_mask_from_beam_pixels(beam_pixel_mask)

    def build_beam_mask_from_sky_pixels(self, sky_pixel_mask):
        """Build mask for beam pixels illuminated by selected sky pixels.

        Given a mask of sky pixels, returns a mask of beam pixels that are
        touched by at least one of the selected sky pixels' HEALPix interpolation
        neighborhoods. This is the reverse operation of selecting sky pixels
        illuminated by a given set of beam pixels.

        Parameters
        ----------
        sky_pixel_mask : array_like, dtype bool
            Boolean mask of sky pixels to consider. The shape depends on whether
            a sky mask has been applied:
            - If no sky mask applied: shape (npix_full,)
            - If sky mask applied: shape (npix_masked,)
            Beam pixels are selected if they appear in the interpolation
            neighborhood of any sky pixel where mask is True.

        Returns
        -------
        beam_mask : np.ndarray, shape (npix_beam_full,), dtype bool
            True for beam pixels that touch selected sky pixels' neighborhoods.

        Example
        -------
        Select high-altitude sky pixels, then find which beam pixels illuminate them:

            sky_mask = cal.build_sky_mask_altitude(min_altitude_deg=30.0)
            beam_mask = cal.build_beam_mask_from_sky_pixels(sky_mask)
            cal.apply_beam_mask(beam_mask)
        """
        return self.fwd.build_beam_mask_from_sky_pixels(sky_pixel_mask)

    def cache_static_sky_coeffs(self, sky_coeffs):
        """Cache visibility contribution of unmasked (static) sky pixels.

        Uses the currently applied pixel_mask to identify static pixels
        (those where mask is False). Simulates their contribution and caches it
        for efficient subtraction during fitting via the subtract_static_sky flag.

        Parameters
        ----------
        sky_coeffs : array, shape (npix_full, nmodes_sky)
            Full-sky coefficients. Pixels where pixel_mask is False are treated
            as static and their visibilities are cached.

        Raises
        ------
        ValueError
            If no pixel mask has been applied.
        """
        if self.pixel_mask is None:
            raise ValueError("No pixel mask applied. Cannot cache static sky.")

        sky_coeffs = np.asarray(sky_coeffs, dtype=DTYPE_R_NPY)
        if sky_coeffs.shape[0] != self._npix_full:
            raise ValueError(
                f"sky_coeffs shape {sky_coeffs.shape[0]} != npix_full {self._npix_full}"
            )

        # Extract static pixels (unmasked) in masked coordinate system
        static_mask_full = ~self.pixel_mask
        # Create array with shape matching the masked forward model
        static_sky = np.zeros((len(self._pixel_indices), sky_coeffs.shape[1]), dtype=DTYPE_R_NPY)

        # Find which masked pixels are static pixels
        for i, full_idx in enumerate(self._pixel_indices):
            if static_mask_full[full_idx]:
                static_sky[i] = sky_coeffs[full_idx]

        # Simulate and cache
        self._cached_static_vis = self._sim_fn(
            jnp.array(static_sky, dtype=DTYPE_R_JAX), self.rot_matrices
        )
        self._static_sky_cached = True

    def expand_sky_to_full(self, sky_coeffs_masked):
        """Expand masked sky coefficients back onto the full sky grid."""
        if self._pixel_indices is None:
            raise RuntimeError('No pixel mask has been applied.')
        sky = np.asarray(sky_coeffs_masked, dtype=DTYPE_R_NPY)
        full = np.zeros((self._npix_full, sky.shape[1]), dtype=DTYPE_R_NPY)
        full[self._pixel_indices] = sky
        return full

    def apply_beam_mask(self, mask):
        """Apply an arbitrary beam pixel mask controlling which pixels are updated.

        Beam pixels where mask is True will be updated during dirty beam steps.
        Beam pixels where mask is False are left at their original values.

        Note: Beam masks are permanent and cannot be removed; create a new Calibrator
        if you need to work with different masks.

        Parameters
        ----------
        mask : array_like, shape (npix_beam_full,)
            Boolean mask of beam pixels to update.
        """
        mask = np.asarray(mask, dtype=bool)
        npix_beam_full = len(mask)
        if npix_beam_full != healpy.nside2npix(self.beam_model.nside):
            raise ValueError(
                f"mask size {npix_beam_full} != beam npix_full {healpy.nside2npix(self.beam_model.nside)}"
            )

        self.beam_mask = mask.copy()
        self.fwd.apply_beam_mask(mask)
        self.fwd.precompute_time_geometry(self.rot_matrices)
        self._recompile_jit()

        n_keep = int(mask.sum())
        n_drop = int((~mask).sum())
        print(f'  Beam mask: {n_keep} pixels retained, {n_drop} removed.')

    def fit_gains_linear(self, sky_coeffs, subtract_static_sky=False):
        """Solve for gains with sky held fixed.

        Parameters
        ----------
        sky_coeffs : array, shape (npix,) or (npix, nmodes)
            Sky coefficients. Can be full-sky or active (masked) pixels.
        subtract_static_sky : bool, optional
            If True, subtract cached static sky contribution from data before fitting.
            Requires cache_static_sky_coeffs() to be called first.

        Returns
        -------
        gain_params : dict
            Gain parameters (per time/frequency, same for full/masked).
        loss : float
            Chi-squared loss.
        """
        if subtract_static_sky:
            if not self._static_sky_cached:
                raise RuntimeError("Static sky not cached. Call cache_static_sky_coeffs() first.")
            data_for_fit = self.data - self._cached_static_vis
        else:
            data_for_fit = self.data

        sky_active = self._ensure_sky_is_active(sky_coeffs)
        vis_model = self._jit_simulate(sky_active, self.rot_matrices)

        den = jnp.abs(vis_model) ** 2
        g_opt = data_for_fit * jnp.conj(vis_model) / (den + 1e-30)
        log_g = jnp.log(g_opt + 0j)

        w_sum = den.sum(axis=2) + 1e-30
        log_amp = (den * jnp.real(log_g)).sum(axis=2) / w_sum

        X = jnp.column_stack([
            jnp.ones(self.nbls, dtype=DTYPE_R_JAX),
            self.bls[:, 0].astype(DTYPE_R_JAX),
            self.bls[:, 1].astype(DTYPE_R_JAX),
        ])

        Xy = jnp.einsum('bk,tfb->tfk', X, den * jnp.imag(log_g))
        XTX = jnp.einsum('bk,tfb,bl->tfkl', X, den, X)
        XTX = XTX + 1e-10 * jnp.eye(3, dtype=DTYPE_R_JAX)

        theta = jnp.linalg.solve(XTX, Xy[..., None])[..., 0]

        gain_params = {
            'log_amp': log_amp.astype(DTYPE_R_JAX),
            'phase':   theta[:, :, 0].astype(DTYPE_R_JAX),
            'phi':     theta[:, :, 1:].transpose(0, 2, 1).astype(DTYPE_R_JAX),
        }
        # Temporarily adjust data for loss computation if subtracting static sky
        if subtract_static_sky:
            orig_data = self.data
            self.data = data_for_fit
            try:
                loss = float(self._jit_loss({'sky_coeffs': sky_active, **gain_params}, self._effective_weights()))
            finally:
                self.data = orig_data
        else:
            loss = float(self._jit_loss({'sky_coeffs': sky_active, **gain_params}, self._effective_weights()))
        self._fit_gains_linear_cache = (sky_active, gain_params, loss)
        return gain_params, loss


    def _recompile_jit(self):
        """Recompile cached-beam JIT functions after precomputed arrays change."""
        self._select_methods()
        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))
        self._jit_loss_variable_beam = jax.jit(self._loss_variable_beam)
        self._jit_simulate = jax.jit(self._sim_fn)

    def compute_adjoint_updates(self, sky_coeffs, residual_vis, update_mode='both', **kwargs):
        """Unified interface for sky/beam adjoint updates.

        Computes corrections to sky and/or beam coefficients from calibrated
        residuals via dirty-map adjoints. The mode controls whether sky, beam,
        or both are updated; when both are updated, a single shared adjoint
        computation is used for efficiency.

        Parameters
        ----------
        sky_coeffs : array, shape (npix_sky, nmodes_sky)
            Sky coefficients (full-sky or active/masked). Required for mode='beam'
            and mode='both'; can be None for mode='sky'.
        residual_vis : array, shape (ntime, nfreq, nbls), complex
            Gain-calibrated residuals.
        update_mode : {'sky', 'beam', 'both'}
            Which parameters to update:
            - 'sky': sky coefficients only (uses beam_reg param)
            - 'beam': beam coefficients only (uses sky_reg param)
            - 'both': both parameters from shared adjoint (uses sky_step_size,
              beam_step_size, beam_reg, sky_reg params)
        **kwargs : dict
            Mode-specific keyword arguments. See Notes for details.

        Returns
        -------
        updates : dict
            Keys present depend on update_mode:
            - 'sky': always returned for modes 'sky' and 'both'
            - 'beam': always returned for modes 'beam' and 'both'
            Each value is the corresponding coefficient correction.

        Notes
        -----
        Keyword arguments by update_mode:

        For update_mode='sky':
            step_size : float, default 1.0
            beam_reg : float, default 1e-3
                Beam regularisation for division.

        For update_mode='beam':
            step_size : float, default 1.0
            sky_reg : float, default 1e-3
                Sky regularisation for division.

        For update_mode='both':
            sky_step_size : float, default 1.0
            beam_step_size : float, default 1.0
            beam_reg : float, default 1e-3
            sky_reg : float, default 1e-3

        The 'both' mode computes a single dirty-map adjoint and applies
        step-weighted sky and beam corrections in sequence (Jacobi coupling
        rather than Gauss-Seidel). This is more efficient than calling sky
        and beam updates separately, but the sequence reflects the shared
        residual linearisation. For tight control, use mode='sky' and
        mode='beam' separately with re-solved gains between.
        """
        if update_mode == 'sky':
            step_size = kwargs.get('step_size', 1.0)
            beam_reg = kwargs.get('beam_reg', 1e-3)
            sky_upd = self._sky_update_fn(residual_vis, step_size=step_size, beam_reg=beam_reg)
            return {'sky': sky_upd}

        elif update_mode == 'beam':
            if sky_coeffs is None:
                raise ValueError("sky_coeffs is required for update_mode='beam'")
            step_size = kwargs.get('step_size', 1.0)
            sky_reg = kwargs.get('sky_reg', 1e-3)
            beam_upd = self._beam_update_fn(sky_coeffs, residual_vis, step_size=step_size, sky_reg=sky_reg)
            return {'beam': beam_upd}

        elif update_mode == 'both':
            if sky_coeffs is None:
                raise ValueError("sky_coeffs is required for update_mode='both'")
            sky_step_size = kwargs.get('sky_step_size', 1.0)
            beam_step_size = kwargs.get('beam_step_size', 1.0)
            beam_reg = kwargs.get('beam_reg', 1e-3)
            sky_reg = kwargs.get('sky_reg', 1e-3)
            sky_upd, beam_upd = self._combined_update_fn(
                sky_coeffs, residual_vis,
                sky_step_size=sky_step_size,
                beam_step_size=beam_step_size,
                beam_reg=beam_reg,
                sky_reg=sky_reg,
            )
            return {'sky': sky_upd, 'beam': beam_upd}

        else:
            raise ValueError(f"update_mode must be 'sky', 'beam', or 'both', got {update_mode!r}")

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

        Parameters are always returned in full-sky format.
        """
        params_input = params
        params = self._params_to_active_space(params)

        sky_coeffs  = params['sky_coeffs']
        gain_params = {k: params[k] for k in self._GAIN_PARAM_KEYS}
        beam_coeffs = params['beam_coeffs']

        best_bc = beam_coeffs
        _w = self._effective_weights()
        best_loss_jax = self._jit_loss_variable_beam(
            {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}, _w
        )
        current_bc = beam_coeffs

        for i in range(n_iter):
            resid = self.calibrated_residual_variable_beam(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': current_bc, **gain_params}
            )
            updates = self.compute_adjoint_updates(
                sky_coeffs, resid, update_mode='beam',
                step_size=step_size, sky_reg=sky_reg
            )
            delta_bc = updates['beam']
            trial_bc = current_bc + delta_bc
            loss_jax = self._jit_loss_variable_beam(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': trial_bc, **gain_params}, _w
            )

            # Use JAX operations to conditionally update (no blocking)
            improved = loss_jax < best_loss_jax
            best_loss_jax = jnp.where(improved, loss_jax, best_loss_jax)
            best_bc = jnp.where(improved, trial_bc, best_bc)
            current_bc = jnp.where(improved, trial_bc, current_bc)

            # Only materialize for verbose output (unavoidable if verbose)
            if verbose:
                loss_val = float(loss_jax)
                print(f'    dirty beam iter {i:03d}: loss={loss_val:.4e}')

        # Synchronize cached beam once at the end.
        self.fwd.update_beam_cache(best_bc)
        self._recompile_jit()

        params_out = {'sky_coeffs': sky_coeffs, 'beam_coeffs': best_bc, **gain_params}
        params_out_full = self._params_to_full_space(params_out, params_input)
        return params_out_full, float(best_loss_jax)

    def fit_sky_and_beam_dirty(
        self,
        params,
        n_iter: int = 3,
        sky_step_size: float = 0.5,
        beam_step_size: float = 0.5,
        beam_reg: float = 1e-3,
        sky_reg: float = 1e-3,
        anderson_history: int = 0,
        anderson_damping: float = 0.5,
        anderson_ridge: float = 1e-8,
        verbose: bool = False,
        subtract_static_sky: bool = False,
    ):
        """Dirty-map sky and beam update from shared adjoint.

        Simultaneously updates sky coefficients and beam coefficients by computing
        the dirty apparent sky map once per iteration and deriving both corrections
        from it. More efficient than calling :meth:`fit_sky_dirty` and
        :meth:`fit_beam_dirty` separately when both are needed.

        Anderson acceleration can be applied to the joint (sky, beam) parameter
        vector to accelerate convergence. This respects the coupling between
        sky and beam updates inherent in the shared adjoint.

        Parameters
        ----------
        params : dict
            Parameter dict with 'sky_coeffs' and 'beam_coeffs'.
        n_iter : int
        sky_step_size : float
        beam_step_size : float
        beam_reg : float
            Beam regularisation
        sky_reg : float
            Sky regularisation
        anderson_history : int, default 0
            Number of past iterates to keep for Anderson acceleration. 0 disables AA.
        anderson_damping : float, default 0.5
            Mixing weight for AA proposal: ``(1-damping)*plain + damping*aa``.
        anderson_ridge : float, default 1e-8
            Tikhonov regularisation for AA least-squares.
        verbose : bool
        subtract_static_sky : bool, optional
            If True, subtract cached static sky contribution from data before fitting.
            Requires cache_static_sky_coeffs() to be called first.

        Returns
        -------
        params_out : dict
            Updated parameters in full-sky format.
        best_loss : float
            Best loss achieved.
        """
        if subtract_static_sky:
            if not self._static_sky_cached:
                raise RuntimeError("Static sky not cached. Call cache_static_sky_coeffs() first.")
            orig_data = self.data
            self.data = self.data - self._cached_static_vis
        else:
            orig_data = None

        try:
            params_input = params
            params = self._params_to_active_space(params)

            sky_coeffs = params['sky_coeffs']
            gain_params = {k: params[k] for k in self._GAIN_PARAM_KEYS}
            beam_coeffs = params['beam_coeffs']

            # Initialize joint Anderson accelerator for (sky, beam) pair
            aa = AndersonAccelerator(
                anderson_history, start=2, damping=anderson_damping,
                ridge=anderson_ridge
            ) if anderson_history > 0 else None

            best_sky = sky_coeffs
            best_bc = beam_coeffs
            _w = self._effective_weights()
            best_loss_jax = self._jit_loss_variable_beam(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}, _w
            )
            current_sky = sky_coeffs
            current_bc = beam_coeffs

            for i in range(n_iter):
                resid = self.calibrated_residual_variable_beam(
                    {'sky_coeffs': current_sky, 'beam_coeffs': current_bc, **gain_params}
                )
                updates = self.compute_adjoint_updates(
                    current_sky, resid, update_mode='both',
                    sky_step_size=sky_step_size,
                    beam_step_size=beam_step_size,
                    beam_reg=beam_reg,
                    sky_reg=sky_reg,
                )
                delta_sky = updates['sky']
                delta_bc = updates['beam']
                trial_sky = current_sky + delta_sky
                trial_bc = current_bc + delta_bc

                # Apply Anderson acceleration to joint (sky, beam) pair if enabled
                if aa is not None:
                    # Flatten both sky and beam to 1D arrays for AA
                    current_flat = np.concatenate([
                        np.asarray(current_sky, dtype=np.float64).ravel(),
                        np.asarray(current_bc, dtype=np.float64).ravel(),
                    ])
                    trial_flat = np.concatenate([
                        np.asarray(trial_sky, dtype=np.float64).ravel(),
                        np.asarray(trial_bc, dtype=np.float64).ravel(),
                    ])

                    # Try AA proposal
                    aa_candidate_flat = aa.push(current_flat, trial_flat)
                    if aa_candidate_flat is not None:
                        # Unflatten the candidate
                        sky_shape = current_sky.shape
                        bc_shape = current_bc.shape
                        sky_size = int(np.prod(sky_shape))
                        sky_cand = jnp.array(
                            aa_candidate_flat[:sky_size].reshape(sky_shape),
                            dtype=DTYPE_R_JAX
                        )
                        bc_cand = jnp.array(
                            aa_candidate_flat[sky_size:].reshape(bc_shape),
                            dtype=DTYPE_R_JAX
                        )
                        loss_cand = self._jit_loss_variable_beam(
                            {'sky_coeffs': sky_cand, 'beam_coeffs': bc_cand, **gain_params}, _w
                        )
                        # Use AA candidate if better than plain step
                        improved_aa = loss_cand < self._jit_loss_variable_beam(
                            {'sky_coeffs': trial_sky, 'beam_coeffs': trial_bc, **gain_params}, _w
                        )
                        if improved_aa:
                            trial_sky = sky_cand
                            trial_bc = bc_cand

                loss_jax = self._jit_loss_variable_beam(
                    {'sky_coeffs': trial_sky, 'beam_coeffs': trial_bc, **gain_params}, _w
                )

                # Use JAX operations to conditionally update (no blocking)
                improved = loss_jax < best_loss_jax
                best_loss_jax = jnp.where(improved, loss_jax, best_loss_jax)
                best_sky = jnp.where(improved, trial_sky, best_sky)
                best_bc = jnp.where(improved, trial_bc, best_bc)
                current_sky = jnp.where(improved, trial_sky, current_sky)
                current_bc = jnp.where(improved, trial_bc, current_bc)

                # Only materialize for verbose output (unavoidable if verbose)
                if verbose:
                    loss_val = float(loss_jax)
                    print(f'    dirty sky+beam iter {i:03d}: loss={loss_val:.4e}')

            # Synchronize cached beam once at the end.
            self.fwd.update_beam_cache(best_bc)
            self._recompile_jit()

            params_out = {'sky_coeffs': best_sky, 'beam_coeffs': best_bc, **gain_params}
            params_out_full = self._params_to_full_space(params_out, params_input)
            return params_out_full, float(best_loss_jax)
        finally:
            if orig_data is not None:
                self.data = orig_data

    def get_sky_beam_weighting(self):
        """Return design matrix normal (Gram) diagonal for all sky pixels.

        Computes the sum of squared beam-weighting coefficients across all times
        and frequencies. This is the effective per-pixel weighting in the fixed-point
        sky update iteration and is independent of data.

        Returns
        -------
        beam_weights : np.ndarray, shape (npix_full,)
            Sum of |beam_spec_h|² per sky pixel across all times and frequencies,
            where beam_spec_h = beam × horizon mask in topocentric frame.
            If a pixel mask has been applied, masked pixels have zero weighting.
        """
        ntime = self.ntime
        npix_sky = self._npix_full

        beam_weights_full = np.zeros(npix_sky, dtype=DTYPE_R_NPY)

        for tind in range(ntime):
            beam_spec_h = np.asarray(self.fwd._beam_spec_horizon_all[tind], dtype=DTYPE_R_NPY)
            weight = (beam_spec_h ** 2).sum(axis=-1)

            if self.pixel_mask is not None:
                # weight is in masked space; expand to full-sky before accumulating
                weight_full = np.zeros(npix_sky, dtype=DTYPE_R_NPY)
                weight_full[self.fwd._pixel_indices] = weight
                beam_weights_full += weight_full
            else:
                beam_weights_full += weight

        return beam_weights_full

    def get_beam_sky_weighting(self):
        """Return design matrix normal (Gram) diagonal for all beam pixels.

        Computes the sum of squared beam-weighting coefficients across all times,
        frequencies, and sky positions that illuminate each beam pixel. This is the
        effective per-pixel weighting in the fixed-point beam update iteration and
        is independent of data.

        Returns
        -------
        beam_weights : np.ndarray, shape (npix_beam_full,)
            Sum of weighted |beam_spec_h|² per beam pixel across all times, frequencies,
            and illuminated sky pixels. Higher values indicate beam pixels that are
            more strongly constrained by the observations.
        """
        npix_beam_full = healpy.nside2npix(self.beam_model.nside)
        beam_weights = np.zeros(npix_beam_full, dtype=DTYPE_R_NPY)

        # For each time, accumulate contributions from sky pixels that illuminate each beam pixel
        for tind in range(self.ntime):
            beam_spec_h = np.asarray(self.fwd._beam_spec_horizon_all[tind], dtype=DTYPE_R_NPY)
            weight_per_sky = (beam_spec_h ** 2).sum(axis=-1)  # (npix_sky, )

            # Get the beam interpolation stencil for this time
            px = np.asarray(self.fwd._interp_px_all[tind], dtype=np.int32)   # (4, npix_sky)
            wgt = np.asarray(self.fwd._interp_wgt_all[tind], dtype=DTYPE_R_NPY)  # (4, npix_sky)

            np.add.at(beam_weights, px.ravel(), (wgt ** 2 * weight_per_sky[None, :]).ravel())

        return beam_weights

    def fit_sky_dirty(
        self,
        sky_coeffs,
        gain_params,
        n_iter: int = 3,
        step_size: float = 0.5,
        beam_reg: float = 1e-3,
        momentum: float = 0.0,
        verbose: bool = False,
        subtract_static_sky: bool = False,
    ):
        """Dirty-map sky update with optional momentum.

        Parameters
        ----------
        sky_coeffs : array
            Sky coefficients (full-sky or active/masked).
        gain_params : dict
            Gain parameters.
        subtract_static_sky : bool, optional
            If True, subtract cached static sky contribution from data before fitting.
            Requires cache_static_sky_coeffs() to be called first.

        Returns
        -------
        best_sky : array
            Updated sky coefficients in full-sky format.
        best_loss : float
            Best loss achieved (computed on adjusted data if subtract_static_sky=True).
        """
        if subtract_static_sky:
            if not self._static_sky_cached:
                raise RuntimeError("Static sky not cached. Call cache_static_sky_coeffs() first.")
            orig_data = self.data
            self.data = self.data - self._cached_static_vis
        else:
            orig_data = None

        try:
            sky_input_full = None
            sky_active = self._ensure_sky_is_active(sky_coeffs)
            if sky_coeffs.shape[0] == self._npix_full:
                sky_input_full = np.asarray(sky_coeffs)

            velocity = jnp.zeros_like(sky_active)
            best_sky_active = sky_active
            _w = self._effective_weights()
            best_loss_jax = self._jit_loss({'sky_coeffs': sky_active, **gain_params}, _w)

            for i in range(n_iter):
                resid = self.calibrated_residual({'sky_coeffs': sky_active, **gain_params})
                updates = self.compute_adjoint_updates(
                    sky_active, resid, update_mode='sky',
                    step_size=step_size, beam_reg=beam_reg
                )
                delta = updates['sky']
                velocity = momentum * velocity + delta
                sky_active = sky_active + velocity
                loss_jax = self._jit_loss({'sky_coeffs': sky_active, **gain_params}, _w)

                # Use JAX operations to conditionally update (no blocking)
                improved = loss_jax < best_loss_jax
                best_loss_jax = jnp.where(improved, loss_jax, best_loss_jax)
                best_sky_active = jnp.where(improved, sky_active, best_sky_active)

                # Only materialize for verbose output (unavoidable if verbose)
                if verbose:
                    loss_val = float(loss_jax)
                    print(f'    dirty iter {i:03d}: loss={loss_val:.4e}')

            best_sky_full = self._params_to_full_space(
                {'sky_coeffs': best_sky_active}, {'sky_coeffs': sky_input_full}
            )['sky_coeffs']
            return best_sky_full, float(best_loss_jax)
        finally:
            if orig_data is not None:
                self.data = orig_data

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
        sky_beam_reg: float = 1e-3,
        sky_anderson_history: int = 0,
        sky_aa_start: int = 2,
        sky_aa_damping: float = 0.5,
        sky_aa_ridge: float = 1e-8,
        beam_sky_reg: float = 1e-3,
        beam_anderson_history: int | None = None,
        beam_aa_start: int = 1,
        beam_aa_damping: float = 0.5,
        beam_aa_ridge: float = 1e-8,
        sky_initial_step: float | list = 1.0,
        beam_initial_step: float | list = 1.0,
        sky_step_gain_factor: float = 2.0,
        beam_step_gain_factor: float = 2.0,
        solve_every: dict | None = None,
        eff_alpha: float = 0.4,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
    ):
        """Create a persistent state for resumable dirty alternating fits.

        Input params can be in full-sky or active (masked) format.
        Internal state uses active format; output params are always full-sky.
        """
        if solve_every is None:
            solve_every = {}
        if beam_anderson_history is None:
            beam_anderson_history = sky_anderson_history

        params_active = self._params_to_active_space(params)

        state = AlternatingDirtyFitState(
            params={
                'sky_coeffs': params_active['sky_coeffs'],
                'beam_coeffs': params_active.get('beam_coeffs', self.fwd.beam_coeffs),
                'log_amp': params_active['log_amp'],
                'phase': params_active['phase'],
                'phi': params_active['phi'],
            },
            settings=dict(
                sky_beam_reg=sky_beam_reg,
                sky_initial_step=sky_initial_step,
                beam_sky_reg=beam_sky_reg,
                beam_initial_step=beam_initial_step,
                solve_every=dict(solve_every),
                eff_alpha=eff_alpha,
                target_reduced_chi2=target_reduced_chi2,
                reduced_chi2_check_every=max(int(reduced_chi2_check_every), 1),
            ),
            sky_acc=AndersonAccelerator(
                sky_anderson_history, sky_aa_start, sky_aa_damping,
                sky_aa_ridge,
                step_gain_factor=sky_step_gain_factor
            ),
            beam_acc=AndersonAccelerator(
                beam_anderson_history, beam_aa_start, beam_aa_damping,
                beam_aa_ridge,
                step_gain_factor=beam_step_gain_factor
            ),
        )

        # Store full input params for later expansion to full-sky
        state.settings['params_input_full'] = params

        # Only update beam cache and recompile JIT when the beam actually changed.
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

    def init_joint_sky_beam_dirty_state(
        self,
        params,
        *,
        sky_beam_reg: float = 1e-3,
        joint_anderson_history: int = 0,
        joint_aa_start: int = 2,
        joint_aa_damping: float = 0.5,
        joint_aa_ridge: float = 1e-8,
        joint_initial_step: float | list = 1.0,
        joint_step_gain_factor: float = 2.0,
        solve_every: dict | None = None,
        eff_alpha: float = 0.4,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
    ):
        """Create a persistent state for resumable joint sky/beam dirty fits.

        Input params can be in full-sky or active (masked) format.
        Internal state uses active format; output params are always full-sky.
        """
        if solve_every is None:
            solve_every = {}

        params_active = self._params_to_active_space(params)

        state = JointSkyBeamDirtyFitState(
            params={
                'sky_coeffs': params_active['sky_coeffs'],
                'beam_coeffs': params_active.get('beam_coeffs', self.fwd.beam_coeffs),
                'log_amp': params_active['log_amp'],
                'phase': params_active['phase'],
                'phi': params_active['phi'],
            },
            settings=dict(
                sky_beam_reg=sky_beam_reg,
                joint_initial_step=joint_initial_step,
                solve_every=dict(solve_every),
                eff_alpha=eff_alpha,
                target_reduced_chi2=target_reduced_chi2,
                reduced_chi2_check_every=max(int(reduced_chi2_check_every), 1),
            ),
            joint_acc=AndersonAccelerator(
                joint_anderson_history, joint_aa_start, joint_aa_damping,
                joint_aa_ridge,
                step_gain_factor=joint_step_gain_factor
            ),
        )

        # Store full input params for later expansion to full-sky
        state.settings['params_input_full'] = params

        # Only update beam cache and recompile JIT when the beam actually changed.
        new_bc = state.params['beam_coeffs']
        if not np.array_equal(np.asarray(new_bc), np.asarray(self.fwd.beam_coeffs)):
            self.fwd.update_beam_cache(new_bc)
            self._recompile_jit()
        state.loss = float(self._jit_loss_variable_beam(state.params, self._effective_weights()))
        if target_reduced_chi2 is not None:
            state.reduced_chi2 = self.calc_reduced_chi2(
                state.params, explicit_beam=True, subtract_params=0
            )
        return state

    def _dirty_step_from_state(self, state: AlternatingDirtyFitState, verbose: bool = False):
        """Advance a persistent alternating-dirty state by one step."""
        if state.stop_reason is not None:
            return state

        s = state.settings
        gains_max_every = s['solve_every'].get('gains', 1)
        sky_max_every = s['solve_every'].get('sky_max', 0)
        beam_max_every = s['solve_every'].get('beam_max', 0)
        gains_enabled = (gains_max_every != 0)
        beam_enabled = (s['solve_every'].get('beam', 1) != 0)

        sky_coeffs = state.params['sky_coeffs']
        beam_coeffs = state.params['beam_coeffs']
        gain_params = {k: state.params[k] for k in self._GAIN_PARAM_KEYS}
        loss = float(state.loss)
        step = state.step
        weights = self._effective_weights()

        def _full_params(sky, beam, gains):
            return {'sky_coeffs': sky, 'beam_coeffs': beam, **gains}

        def _sky_plain_step(sky, gains, current_loss, step_gain=1.0):
            resid = self.calibrated_residual({'sky_coeffs': sky, **gains})
            # Get base update with step_size=1.0 to enable efficient scaling
            updates = self.compute_adjoint_updates(
                sky, resid, update_mode='sky',
                step_size=1.0, beam_reg=s['sky_beam_reg']
            )
            delta_base = updates['sky']

            best_sky = sky
            best_loss = current_loss
            eff_gain = 0.0
            initial_step = s.get('sky_initial_step', 1.0)

            # If initial_step is a list, do one-time line search to pick the best
            if isinstance(initial_step, (list, tuple)) and len(state.sky_acc._hist_f) == 0 and step_gain == 1.0:
                selected_step = 1.0
                for candidate_step in initial_step:
                    sky_trial = (sky + candidate_step * delta_base).astype(DTYPE_R_JAX)
                    loss_trial = float(self._jit_loss({'sky_coeffs': sky_trial, **gains}, weights))
                    if loss_trial < best_loss:
                        best_sky = sky_trial
                        best_loss = loss_trial
                        selected_step = candidate_step
                # Save the best step as a scalar for future use
                if best_loss < current_loss:
                    s['sky_initial_step'] = selected_step
                    if verbose:
                        print(f"      [initial line search: selected sky_initial_step={selected_step:.2f}]")
                eff_gain = 1.0  # Return 1.0 so report_step_gain(1.0) keeps step_gain=1.0
            else:
                # Normal retry loop: try current step_gain, then halve if unsuccessful
                gain_to_try = step_gain
                while gain_to_try >= state.sky_acc._min_step_gain:
                    sky_trial = (sky + gain_to_try * initial_step * delta_base).astype(DTYPE_R_JAX)
                    loss_trial = float(self._jit_loss({'sky_coeffs': sky_trial, **gains}, weights))
                    if loss_trial < best_loss:
                        best_sky = sky_trial
                        best_loss = loss_trial
                        eff_gain = gain_to_try
                    if best_loss < current_loss:
                        break
                    gain_to_try *= 0.5

            return best_sky, best_loss, eff_gain

        def _beam_plain_step(sky, beam, gains, current_loss, step_gain=1.0):
            resid = self.calibrated_residual_variable_beam(
                {'sky_coeffs': sky, 'beam_coeffs': beam, **gains}
            )
            # Get base update with step_size=1.0 to enable efficient scaling
            updates = self.compute_adjoint_updates(
                sky, resid, update_mode='beam',
                step_size=1.0, sky_reg=s['beam_sky_reg']
            )
            delta_bc_base = updates['beam']

            best_beam = beam
            best_loss = current_loss
            eff_gain = 0.0
            initial_step = s.get('beam_initial_step', 1.0)

            # If initial_step is a list, do one-time line search to pick the best
            if isinstance(initial_step, (list, tuple)) and len(state.beam_acc._hist_f) == 0 and step_gain == 1.0:
                selected_step = 1.0
                for candidate_step in initial_step:
                    beam_trial = (beam + candidate_step * delta_bc_base).astype(DTYPE_R_JAX)
                    loss_trial = float(self._jit_loss_variable_beam(
                        {'sky_coeffs': sky, 'beam_coeffs': beam_trial, **gains},
                        weights,
                    ))
                    if loss_trial < best_loss:
                        best_beam = beam_trial
                        best_loss = loss_trial
                        selected_step = candidate_step
                # Save the best step as a scalar for future use
                if best_loss < current_loss:
                    s['beam_initial_step'] = selected_step
                    if verbose:
                        print(f"      [initial line search: selected beam_initial_step={selected_step:.2f}]")
                eff_gain = 1.0  # Return 1.0 so report_step_gain(1.0) keeps step_gain=1.0
            else:
                # Normal retry loop: try current step_gain, then halve if unsuccessful
                gain_to_try = step_gain
                while gain_to_try >= state.beam_acc._min_step_gain:
                    beam_trial = (beam + gain_to_try * initial_step * delta_bc_base).astype(DTYPE_R_JAX)
                    loss_trial = float(self._jit_loss_variable_beam(
                        {'sky_coeffs': sky, 'beam_coeffs': beam_trial, **gains},
                        weights,
                    ))
                    if loss_trial < best_loss:
                        best_beam = beam_trial
                        best_loss = loss_trial
                        eff_gain = gain_to_try
                    if best_loss < current_loss:
                        break
                    gain_to_try *= 0.5

            return best_beam, best_loss, eff_gain

        overdue = {}
        if sky_max_every > 0 and state.n_since_sky >= sky_max_every:
            overdue['sky'] = state.n_since_sky / sky_max_every
        if beam_enabled and beam_max_every > 0 and state.n_since_beam >= beam_max_every:
            overdue['beam'] = state.n_since_beam / beam_max_every
        if gains_enabled and gains_max_every > 0 and state.n_since_gains >= gains_max_every:
            overdue['gains'] = state.n_since_gains / gains_max_every

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
            step_type = max(cands, key=cands.get)

        do_sky = (step_type == 'sky')
        do_gains = (step_type == 'gains')

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
            sky_plain, loss_plain, eff_gain = _sky_plain_step(sky_coeffs, gain_params, loss, state.sky_acc.step_gain)
            state.sky_acc.report_step_gain(eff_gain)
            sky_next = sky_plain
            gain_next = gain_params
            loss_next = loss_plain
            used_aa = False
            cand_flat = state.sky_acc.push(
                np.asarray(sky_coeffs, dtype=np.float64).ravel(),
                np.asarray(sky_plain, dtype=np.float64).ravel(),
            )
            aa_proposed = False
            if cand_flat is not None:
                sky_cand = jnp.array(cand_flat.reshape(sky_coeffs.shape), dtype=DTYPE_R_JAX)
                gain_cand, _ = self.fit_gains_linear(sky_cand)
                loss_cand = float(self._jit_loss(_full_params(sky_cand, beam_coeffs, gain_cand), weights))
                aa_proposed = True
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
                line = f"    [sky {tag} {state.n_sky - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s  step_gain={state.sky_acc.step_gain:.2f}"
                if aa_proposed and not used_aa:
                    line += f"  [AA rejected: cand={loss_cand:.4e} vs plain={loss_plain:.4e}]"
                elif not aa_proposed and cand_flat is None and state.sky_acc.history > 0 and state.sky_acc._step > state.sky_acc.start and len(state.sky_acc._hist_f) >= 2:
                    line += "  [AA rejected: non-finite coefficients]"
                print(line)
        elif step_type == 'beam':
            beam_plain, loss_plain, eff_gain = _beam_plain_step(sky_coeffs, beam_coeffs, gain_params, loss, state.beam_acc.step_gain)
            state.beam_acc.report_step_gain(eff_gain)
            beam_next = beam_plain
            loss_next = loss_plain
            used_aa = False
            cand_flat = state.beam_acc.push(
                np.asarray(beam_coeffs, dtype=np.float64).ravel(),
                np.asarray(beam_plain, dtype=np.float64).ravel(),
            )
            if cand_flat is not None:
                beam_cand = jnp.array(cand_flat.reshape(beam_coeffs.shape), dtype=DTYPE_R_JAX)
                loss_cand = float(self._jit_loss_variable_beam(_full_params(sky_coeffs, beam_cand, gain_params), weights))
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
            if verbose:
                tag = 'AA' if used_aa else '  '
                print(f"    [beam {tag} {state.n_beam - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s  step_gain={state.beam_acc.step_gain:.2f}")

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
            msg = (f"  step {state.step - 1:04d} [{step_type:<11}]: chi2={state.loss:.4e}"
                   f"  eff[sky={eff_s} beam={eff_b} gains={eff_g}]"
                   f"  t={elapsed:.1f}s")
            if state.reduced_chi2 is not None:
                msg += f"  red_chi2={state.reduced_chi2:.4f}"
            print(msg)

        return state

    def _joint_sky_beam_dirty_step_from_state(self, state: JointSkyBeamDirtyFitState, verbose: bool = False):
        """Advance a persistent joint sky+beam dirty state by one step."""
        if state.stop_reason is not None:
            return state

        s = state.settings
        gains_max_every = s['solve_every'].get('gains', 1)
        gains_enabled = (gains_max_every != 0)

        sky_coeffs = state.params['sky_coeffs']
        beam_coeffs = state.params['beam_coeffs']
        gain_params = {k: state.params[k] for k in self._GAIN_PARAM_KEYS}
        loss = float(state.loss)
        step = state.step
        weights = self._effective_weights()

        t_step = _time.perf_counter()
        loss_pre = loss

        # Determine step type: gains or joint sky+beam
        overdue = {}
        if gains_enabled and gains_max_every > 0 and state.n_since_gains >= gains_max_every:
            overdue['gains'] = state.n_since_gains / gains_max_every

        if overdue or (state.eff_gains is None and gains_enabled):
            do_gains = True
        else:
            do_gains = False

        if do_gains:
            # Solve gains with sky and beam held fixed
            sky_coeffs_for_gains = state.params['sky_coeffs']
            gain_params, loss = self.fit_gains_linear(sky_coeffs_for_gains)
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
            state.n_since_joint += 1
            if verbose:
                print(f"    [gains {state.n_gains - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s")
        else:
            # Joint sky+beam step with Anderson acceleration
            resid = self.calibrated_residual_variable_beam(
                {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}
            )
            updates = self.compute_adjoint_updates(
                sky_coeffs, resid, update_mode='both',
                sky_step_size=1.0, beam_step_size=1.0,
                beam_reg=s['sky_beam_reg'], sky_reg=s['sky_beam_reg']
            )
            delta_sky_base = updates['sky']
            delta_beam_base = updates['beam']

            joint_sky_trial = sky_coeffs
            joint_beam_trial = beam_coeffs
            joint_loss_trial = loss
            eff_gain = 0.0
            initial_step = s.get('joint_initial_step', 1.0)

            # If initial_step is a list, do one-time line search to pick the best
            if isinstance(initial_step, (list, tuple)) and len(state.joint_acc._hist_f) == 0 and state.joint_acc.step_gain == 1.0:
                selected_step = 1.0
                for candidate_step in initial_step:
                    joint_sky_step = (sky_coeffs + candidate_step * delta_sky_base).astype(DTYPE_R_JAX)
                    joint_beam_step = (beam_coeffs + candidate_step * delta_beam_base).astype(DTYPE_R_JAX)
                    joint_loss_step = float(self._jit_loss_variable_beam(
                        {'sky_coeffs': joint_sky_step, 'beam_coeffs': joint_beam_step, **gain_params},
                        weights
                    ))
                    if joint_loss_step < joint_loss_trial:
                        joint_sky_trial = joint_sky_step
                        joint_beam_trial = joint_beam_step
                        joint_loss_trial = joint_loss_step
                        selected_step = candidate_step
                # Save the best step as a scalar for future use
                if joint_loss_trial < loss:
                    s['joint_initial_step'] = selected_step
                    if verbose:
                        print(f"      [initial line search: selected joint_initial_step={selected_step:.2f}]")
                eff_gain = 1.0  # Return 1.0 so report_step_gain keeps step_gain=1.0
            else:
                # Normal retry loop: try current step_gain, then halve if unsuccessful
                gain_to_try = state.joint_acc.step_gain
                while gain_to_try >= state.joint_acc._min_step_gain:
                    joint_sky_step = (sky_coeffs + gain_to_try * initial_step * delta_sky_base).astype(DTYPE_R_JAX)
                    joint_beam_step = (beam_coeffs + gain_to_try * initial_step * delta_beam_base).astype(DTYPE_R_JAX)
                    joint_loss_step = float(self._jit_loss_variable_beam(
                        {'sky_coeffs': joint_sky_step, 'beam_coeffs': joint_beam_step, **gain_params},
                        weights
                    ))
                    if joint_loss_step < joint_loss_trial:
                        joint_sky_trial = joint_sky_step
                        joint_beam_trial = joint_beam_step
                        joint_loss_trial = joint_loss_step
                        eff_gain = gain_to_try
                    if joint_loss_trial < loss:
                        break
                    gain_to_try *= 0.5

            state.joint_acc.report_step_gain(eff_gain)
            joint_sky_plain = joint_sky_trial
            joint_beam_plain = joint_beam_trial
            joint_loss_plain = joint_loss_trial

            joint_sky_next = joint_sky_plain
            joint_beam_next = joint_beam_plain
            joint_loss_next = joint_loss_plain
            used_aa = False

            # Apply Anderson acceleration to joint (sky, beam) pair
            if state.joint_acc is not None:
                current_flat = np.concatenate([
                    np.asarray(sky_coeffs, dtype=np.float64).ravel(),
                    np.asarray(beam_coeffs, dtype=np.float64).ravel(),
                ])
                plain_flat = np.concatenate([
                    np.asarray(joint_sky_plain, dtype=np.float64).ravel(),
                    np.asarray(joint_beam_plain, dtype=np.float64).ravel(),
                ])

                aa_candidate_flat = state.joint_acc.push(current_flat, plain_flat)
                if aa_candidate_flat is not None:
                    sky_shape = sky_coeffs.shape
                    beam_shape = beam_coeffs.shape
                    sky_size = int(np.prod(sky_shape))
                    joint_sky_cand = jnp.array(
                        aa_candidate_flat[:sky_size].reshape(sky_shape),
                        dtype=DTYPE_R_JAX
                    )
                    joint_beam_cand = jnp.array(
                        aa_candidate_flat[sky_size:].reshape(beam_shape),
                        dtype=DTYPE_R_JAX
                    )
                    joint_loss_cand = float(self._jit_loss_variable_beam(
                        {'sky_coeffs': joint_sky_cand, 'beam_coeffs': joint_beam_cand, **gain_params},
                        weights
                    ))
                    if joint_loss_cand < joint_loss_plain:
                        joint_sky_next = joint_sky_cand
                        joint_beam_next = joint_beam_cand
                        joint_loss_next = joint_loss_cand
                        used_aa = True

            sky_coeffs = joint_sky_next
            beam_coeffs = joint_beam_next
            loss = joint_loss_next

            dt = max(_time.perf_counter() - t_step, 1e-3)
            dloss = max(0.0, loss_pre - loss)
            eff = dloss / (max(loss_pre, 1e-30) * dt)
            state.eff_joint = eff if state.eff_joint is None else s['eff_alpha'] * eff + (1.0 - s['eff_alpha']) * state.eff_joint
            state.n_joint += 1
            state.n_since_joint = 0
            state.n_since_gains += 1
            if verbose:
                tag = 'AA' if used_aa else '  '
                line = f"    [joint {tag} {state.n_joint - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s  step_gain={state.joint_acc.step_gain:.2f}"
                if used_aa:
                    line += f"  [AA improved: cand={joint_loss_cand:.4e} vs plain={joint_loss_plain:.4e}]"
                print(line)

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
            eff_j = f'{state.eff_joint:.2e}' if state.eff_joint is not None else ' N/A  '
            eff_g = f'{state.eff_gains:.2e}' if state.eff_gains is not None else ' N/A  '
            step_type = 'gains' if do_gains else 'joint'
            msg = (f"  step {state.step - 1:04d} [{step_type:<11}]: chi2={state.loss:.4e}"
                   f"  eff[joint={eff_j} gains={eff_g}]"
                   f"  t={elapsed:.1f}s")
            if state.reduced_chi2 is not None:
                msg += f"  red_chi2={state.reduced_chi2:.4f}"
            print(msg)

        return state

    def run_alternating_dirty_state(self, state: AlternatingDirtyFitState, n_iter: int = 1, verbose: bool = False, _stop_flag=None):
        """Advance a persistent alternating-dirty state by ``n_iter`` steps."""
        # Handle subtract_static_sky: temporarily adjust data if flag is set
        if state.subtract_static_sky:
            if not self._static_sky_cached:
                raise RuntimeError("Static sky not cached. Call cache_static_sky_coeffs() first.")
            orig_data = self.data
            self.data = self.data - self._cached_static_vis
        else:
            orig_data = None

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
            if orig_data is not None:
                self.data = orig_data
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


    def run_joint_sky_beam_dirty_state(self, state: JointSkyBeamDirtyFitState, n_iter: int = 1, verbose: bool = False, _stop_flag=None):
        """Advance a persistent joint sky+beam dirty state by ``n_iter`` steps."""
        if state.subtract_static_sky:
            if not self._static_sky_cached:
                raise RuntimeError("Static sky not cached. Call cache_static_sky_coeffs() first.")
            orig_data = self.data
            self.data = self.data - self._cached_static_vis
        else:
            orig_data = None

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
                self._joint_sky_beam_dirty_step_from_state(state, verbose=verbose)
                if state.stop_reason is not None:
                    break
        finally:
            if orig_data is not None:
                self.data = orig_data
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return state

    def iter_joint_sky_beam_dirty(self, state: JointSkyBeamDirtyFitState, n_iter: int, yield_every: int = 1, verbose: bool = False, _stop_flag=None):
        """Yield a persistent state while advancing the fit."""
        state = self.run_joint_sky_beam_dirty_state(state, 0, verbose=False, _stop_flag=_stop_flag)
        for i in range(n_iter):
            state = self.run_joint_sky_beam_dirty_state(state, 1, verbose=verbose, _stop_flag=_stop_flag)
            if ((i + 1) % max(int(yield_every), 1) == 0) or state.stop_reason is not None:
                yield state
            if state.stop_reason is not None:
                break

    def fit_joint_sky_beam_dirty(
        self,
        params,
        n_iter: int = 30,
        sky_beam_reg: float = 1e-3,
        joint_anderson_history: int = 0,
        joint_aa_start: int = 2,
        joint_aa_damping: float = 0.5,
        joint_aa_ridge: float = 1e-8,
        joint_initial_step: float | list = 1.0,
        joint_step_gain_factor: float = 2.0,
        solve_every: dict | None = None,
        eff_alpha: float = 0.4,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
        subtract_static_sky: bool = False,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Joint sky+beam dirty-map minimisation with optional gain updates.

        Simultaneously updates sky and beam coefficients from a shared adjoint,
        with optional periodic gain solves via the ``solve_every`` dict.
        Provides the same resumable state-machine interface as
        :meth:`fit_alternating_dirty`.

        Parameters are always returned in full-sky format.

        Parameters
        ----------
        params : dict
            Parameter dict with 'sky_coeffs' and 'beam_coeffs' (full-sky or masked).
        n_iter : int
        sky_beam_reg : float, default 1e-3
            Regularisation for sky and beam divisions.
        anderson_history : int, default 0
            Number of past iterates for Anderson acceleration on joint (sky, beam) vector.
            0 disables AA.
        joint_aa_start : int, default 2
            Number of plain steps before AA activation.
        joint_aa_damping : float, default 0.5
            AA mixing weight.
        joint_aa_ridge : float, default 1e-8
            AA Tikhonov regularisation.
        joint_initial_step : float or list, default 1.0
            Step size for plain joint step. If a list, performs one-time line search
            on first iteration to select the best value.
        joint_step_gain_factor : float, default 2.0
            Factor for adaptive step scaling; step_gain is multiplied by this when
            iterations stall or boosted when they succeed.
        solve_every : dict, optional
            Cadence control: ``{'gains': n}`` solves gains every n steps.
            Default is ``{'gains': 1}`` (solve gains every step).
        eff_alpha : float, default 0.4
            EMA factor for efficiency tracking.
        target_reduced_chi2 : float, optional
            Stop when reduced chi-squared reaches this threshold.
        reduced_chi2_check_every : int, default 1
            Check reduced chi-squared every this many steps.
        subtract_static_sky : bool, default False
            Subtract cached static sky contribution from data before fitting.
        verbose : bool
        """
        state = self.init_joint_sky_beam_dirty_state(
            params,
            sky_beam_reg=sky_beam_reg,
            joint_anderson_history=joint_anderson_history,
            joint_aa_start=joint_aa_start,
            joint_aa_damping=joint_aa_damping,
            joint_aa_ridge=joint_aa_ridge,
            joint_initial_step=joint_initial_step,
            joint_step_gain_factor=joint_step_gain_factor,
            solve_every=solve_every,
            eff_alpha=eff_alpha,
            target_reduced_chi2=target_reduced_chi2,
            reduced_chi2_check_every=reduced_chi2_check_every,
        )
        state.subtract_static_sky = subtract_static_sky
        state = self.run_joint_sky_beam_dirty_state(state, n_iter=n_iter, verbose=verbose, _stop_flag=_stop_flag)
        params_full = self._params_to_full_space(state.params, state.settings.get('params_input_full'))
        return params_full, float(state.loss)

    def fit_alternating_dirty(
        self,
        params,
        n_iter: int = 30,
        sky_beam_reg: float = 1e-3,
        sky_anderson_history: int = 0,
        sky_aa_start: int = 2,
        sky_aa_damping: float = 0.5,
        sky_aa_ridge: float = 1e-8,
        beam_sky_reg: float = 1e-3,
        beam_anderson_history: int | None = None,
        beam_aa_start: int = 1,
        beam_aa_damping: float = 0.5,
        beam_aa_ridge: float = 1e-8,
        sky_initial_step: float | list = 1.0,
        beam_initial_step: float | list = 1.0,
        sky_step_gain_factor: float = 2.0,
        beam_step_gain_factor: float = 2.0,
        solve_every: dict | None = None,
        eff_alpha: float = 0.4,
        target_reduced_chi2: float | None = None,
        reduced_chi2_check_every: int = 1,
        subtract_static_sky: bool = False,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Adaptive alternating dirty-map minimisation.

        One-shot wrapper around the resumable state-machine interface.
        Use :meth:`init_alternating_dirty_state`, :meth:`run_alternating_dirty_state`,
        and :meth:`iter_alternating_dirty` to pause and resume fits while preserving
        Anderson histories, efficiency estimates, and cadence counters.

        Parameters are always returned in full-sky format.

        Parameters
        ----------
        subtract_static_sky : bool, optional
            If True, subtract cached static sky contribution from data before fitting.
            Requires cache_static_sky_coeffs() to be called first.
        """
        state = self.init_alternating_dirty_state(
            params,
            sky_beam_reg=sky_beam_reg,
            sky_anderson_history=sky_anderson_history,
            sky_aa_start=sky_aa_start,
            sky_aa_damping=sky_aa_damping,
            sky_aa_ridge=sky_aa_ridge,
            sky_initial_step=sky_initial_step,
            beam_sky_reg=beam_sky_reg,
            beam_anderson_history=beam_anderson_history,
            beam_aa_start=beam_aa_start,
            beam_aa_damping=beam_aa_damping,
            beam_aa_ridge=beam_aa_ridge,
            beam_initial_step=beam_initial_step,
            solve_every=solve_every,
            eff_alpha=eff_alpha,
            target_reduced_chi2=target_reduced_chi2,
            reduced_chi2_check_every=reduced_chi2_check_every,
        )
        state.subtract_static_sky = subtract_static_sky
        state = self.run_alternating_dirty_state(state, n_iter=n_iter, verbose=verbose, _stop_flag=_stop_flag)
        params_full = self._params_to_full_space(state.params, state.settings.get('params_input_full'))
        return params_full, float(state.loss)

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
