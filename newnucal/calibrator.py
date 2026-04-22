"""
Calibrator — ties together ForwardModel, gain parameters, and optimisation.
"""

import signal as _signal
import time as _time
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
        """Reset history and step counter."""
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

    n_sky: int = 0
    n_beam: int = 0
    n_gains: int = 0

    n_since_sky: int = 0
    n_since_beam: int = 0
    n_since_gains: int = 0

    beam_dirty_pending: bool = False
    stop_reason: str | None = None


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
        self._active_size = self._npix_full   # cached; updated by apply_pixel_mask
        self._pixel_indices = None            # int32 indices of active pixels; None means all

        self._select_methods()
        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))
        self._jit_loss_variable_beam = jax.jit(self._loss_variable_beam)
        self._jit_simulate = jax.jit(self._sim_fn)
        self.set_channel_weights(channel_weights)
        self.set_inv_noise_var(inv_noise_var)
        if noise_sigma is not None:
            self.set_noise_sigma(noise_sigma)

    def _select_methods(self):
        """Bind forward/adjoint callables based on self.method."""
        if self.method == '2d':
            self._sim_fn          = self.fwd.simulate_2d
            self._var_beam_sim_fn = self.fwd.simulate_variable_beam_2d
            self._sky_update_fn   = self.fwd.accumulate_equatorial_sky_update_2d
            self._beam_update_fn  = self.fwd.accumulate_beam_update_2d
        else:
            self._sim_fn          = self.fwd.simulate
            self._var_beam_sim_fn = self.fwd.simulate_variable_beam
            self._sky_update_fn   = self.fwd.accumulate_equatorial_sky_update
            self._beam_update_fn  = self.fwd.accumulate_beam_update

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

    def apply_pixel_mask(self, mask):
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
        self.fwd.apply_pixel_mask(mask)
        self.fwd.precompute_time_geometry(self.rot_matrices)
        self._recompile_jit()

        n_keep = int(mask.sum())
        n_drop = int((~mask).sum())
        print(f'  Pixel mask: {n_keep} pixels retained, {n_drop} removed.')

    def apply_horizon_cut(self, params=None):
        """Remove sky pixels that are never above the horizon.

        Convenience wrapper around apply_pixel_mask.
        Always returns number of kept pixels (no longer takes params).
        """
        mask = self.fwd.build_ever_visible_mask(self.rot_matrices)
        mask = np.asarray(mask, dtype=bool)

        if self.pixel_mask is None or mask.shape[0] == self._npix_full:
            self.apply_pixel_mask(mask)
        else:
            raise RuntimeError('Cannot apply horizon cut after a mask is already applied.')

        if params is not None:
            raise ValueError(
                'apply_horizon_cut no longer takes params. '
                'Parameters are always returned in full-sky format.'
            )
        return int(mask.sum())

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

    def apply_ever_illuminated_beam_mask(self):
        """Restrict beam updates to pixels that are ever illuminated by the sky.

        Convenience wrapper around apply_beam_mask.
        This can substantially speed up beam dirty updates.
        """
        if not self.fwd._geom_ready:
            raise RuntimeError('Call precompute_time_geometry() first.')

        mask = self.fwd.build_ever_illuminated_beam_mask(self.rot_matrices)
        mask = np.asarray(mask, dtype=bool)
        self.apply_beam_mask(mask)

    def fit_gains_linear(self, sky_coeffs):
        """Solve for gains with sky held fixed.

        Parameters
        ----------
        sky_coeffs : array, shape (npix,) or (npix, nmodes)
            Sky coefficients. Can be full-sky or active (masked) pixels.

        Returns
        -------
        gain_params : dict
            Gain parameters (per time/frequency, same for full/masked).
        loss : float
            Chi-squared loss.
        """
        sky_active = self._ensure_sky_is_active(sky_coeffs)
        vis_model = self._jit_simulate(sky_active, self.rot_matrices)

        den = jnp.abs(vis_model) ** 2
        g_opt = self.data * jnp.conj(vis_model) / (den + 1e-30)
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
            delta_bc = self._beam_update_fn(
                sky_coeffs, resid, step_size=step_size, sky_reg=sky_reg,
            )
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
        """Dirty-map sky update with optional momentum.

        Parameters
        ----------
        sky_coeffs : array
            Sky coefficients (full-sky or active/masked).
        gain_params : dict
            Gain parameters.

        Returns
        -------
        best_sky : array
            Updated sky coefficients in full-sky format.
        best_loss : float
            Best loss achieved.
        """
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
            delta = self._sky_update_fn(resid, step_size=step_size, beam_reg=beam_reg)
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
        beam_step_size: float = 0.5,
        beam_sky_reg: float = 1e-3,
        beam_anderson_history: int | None = None,
        beam_aa_start: int = 1,
        beam_aa_damping: float = 0.5,
        beam_aa_ridge: float = 1e-8,
        beam_aa_max_weight: float = 10.0,
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
                sky_step_size=sky_step_size,
                sky_beam_reg=sky_beam_reg,
                beam_step_size=beam_step_size,
                beam_sky_reg=beam_sky_reg,
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

        def _full_params(sky, beam, gains):
            return {'sky_coeffs': sky, 'beam_coeffs': beam, **gains}

        def _sky_plain_step(sky, gains, current_loss):
            resid = self.calibrated_residual({'sky_coeffs': sky, **gains})
            delta = self._sky_update_fn(
                resid, step_size=s['sky_step_size'], beam_reg=s['sky_beam_reg']
            )
            sky_trial = (sky + delta).astype(DTYPE_R_JAX)
            loss_trial = float(self._jit_loss({'sky_coeffs': sky_trial, **gains}, self._effective_weights()))
            if loss_trial < current_loss:
                return sky_trial, loss_trial
            return sky, current_loss

        def _beam_plain_step(sky, beam, gains, current_loss):
            resid = self.calibrated_residual_variable_beam(
                {'sky_coeffs': sky, 'beam_coeffs': beam, **gains}
            )
            delta_bc = self._beam_update_fn(
                sky, resid, step_size=s['beam_step_size'], sky_reg=s['beam_sky_reg']
            )
            beam_trial = (beam + delta_bc).astype(DTYPE_R_JAX)
            loss_trial = float(self._jit_loss_variable_beam(
                {'sky_coeffs': sky, 'beam_coeffs': beam_trial, **gains},
                self._effective_weights(),
            ))
            if loss_trial < current_loss:
                return beam_trial, loss_trial
            return beam, current_loss

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
                sky_cand = jnp.array(cand_flat.reshape(sky_coeffs.shape), dtype=DTYPE_R_JAX)
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
                beam_cand = jnp.array(cand_flat.reshape(beam_coeffs.shape), dtype=DTYPE_R_JAX)
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
            if verbose:
                tag = 'AA' if used_aa else '  '
                print(f"    [beam {tag} {state.n_beam - 1:03d}]: loss={loss:.4e}  eff={eff:.2e} frac_Δloss/s")

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
        beam_step_size: float = 0.5,
        beam_sky_reg: float = 1e-3,
        beam_anderson_history: int | None = None,
        beam_aa_start: int = 1,
        beam_aa_damping: float = 0.5,
        beam_aa_ridge: float = 1e-8,
        beam_aa_max_weight: float = 10.0,
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

        Parameters are always returned in full-sky format.
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
            beam_step_size=beam_step_size,
            beam_sky_reg=beam_sky_reg,
            beam_anderson_history=beam_anderson_history,
            beam_aa_start=beam_aa_start,
            beam_aa_damping=beam_aa_damping,
            beam_aa_ridge=beam_aa_ridge,
            beam_aa_max_weight=beam_aa_max_weight,
            solve_every=solve_every,
            eff_alpha=eff_alpha,
            target_reduced_chi2=target_reduced_chi2,
            reduced_chi2_check_every=reduced_chi2_check_every,
        )
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
