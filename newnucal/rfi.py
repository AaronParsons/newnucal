"""RFI-aware channel weighting utilities for newnucal.

Maintains soft reliability weights for time/frequency channels via regularized
optimization, suitable for use in weighted losses and weighted dirty-map updates.
The module does not attempt to model RFI itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from .utils import DTYPE_R_NPY as DTYPE_R



def _as_tf(arr: np.ndarray | None, ntime: int, nfreq: int, fill_value: float = 1.0) -> np.ndarray:
    if arr is None:
        return np.full((ntime, nfreq), fill_value, dtype=DTYPE_R)
    arr = np.asarray(arr, dtype=DTYPE_R)
    if arr.shape == (nfreq,):
        arr = np.broadcast_to(arr[None, :], (ntime, nfreq))
    elif arr.shape == (ntime, nfreq):
        pass
    else:
        raise ValueError(f"Expected shape ({nfreq},) or ({ntime}, {nfreq}), got {arr.shape}")
    return np.array(arr, dtype=DTYPE_R, copy=True)



def prepare_initial_channel_weights(
    *,
    ntime: int,
    nfreq: int,
    initial_weights: np.ndarray | None = None,
    initial_flags: np.ndarray | None = None,
    flagged_weight: float = 0.05,
) -> np.ndarray:
    """Build initial time/frequency channel weights.

    Parameters
    ----------
    initial_weights : array_like, optional
        External soft weights, shape ``(nfreq,)`` or ``(ntime, nfreq)``.
    initial_flags : array_like of bool, optional
        External hard flags, shape ``(nfreq,)`` or ``(ntime, nfreq)``.
    flagged_weight : float
        Soft weight applied to flagged channels.
    """
    w = _as_tf(initial_weights, ntime, nfreq, fill_value=1.0)
    if initial_flags is not None:
        flags = _as_tf(np.asarray(initial_flags, dtype=bool), ntime, nfreq, fill_value=False).astype(bool)
        w = np.where(flags, DTYPE_R(flagged_weight), w)
    return np.clip(w, 0.0, 1.0).astype(DTYPE_R)



def channel_chi2_statistic(
    residual: np.ndarray,
    inv_noise_var: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-time, per-frequency whitened residual power.

    Parameters
    ----------
    residual : array_like, shape (ntime, nfreq, nbls)
    inv_noise_var : array_like, shape (nfreq,) or (ntime, nfreq), optional
        Inverse variance for a single complex visibility sample.

    Returns
    -------
    chi2_tf : np.ndarray, shape (ntime, nfreq)
        Mean whitened residual power across baselines.
    """
    resid = np.asarray(residual)
    if resid.ndim != 3:
        raise ValueError(f"Expected residual with shape (ntime, nfreq, nbls), got {resid.shape}")
    ntime, nfreq, _ = resid.shape
    inv_var_tf = _as_tf(inv_noise_var, ntime, nfreq, fill_value=1.0)
    chi2_tf = np.mean(np.abs(resid) ** 2, axis=2) * inv_var_tf
    return chi2_tf.astype(DTYPE_R)





def fit_soft_channel_weights(
    *,
    residual: np.ndarray,
    inv_noise_var: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    regularization: float = 1.0,
    regularization_power: float = 2.0,
    min_weight: float = 0.01,
    max_weight: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit soft channel weights via regularized optimization.

    Directly optimizes weights
    to minimize weighted loss while penalizing aggressive downweighting.
    A soft regularization term discourages deviations from 1, but allows
    targeted reduction for non-spectrally-redundant features.

    Parameters
    ----------
    residual : array_like, shape (ntime, nfreq, nbls)
        Gain-calibrated model residuals (or |residuals|^2).
    inv_noise_var : array_like, optional
        Inverse variance per time/frequency channel.
    prior_weights : array_like, optional
        External bootstrap prior from DPSS / high-pass flagging.
        Acts as an upper bound (cap): optimized weights will not exceed prior.
        Flagged channels with prior << 1 are effectively hard-capped.
    regularization : float, default 1.0
        Strength of regularization penalizing deviation from weight=1.
        Higher values discourage aggressive downweighting.
    regularization_power : float, default 2.0
        Power of the regularization term. 2.0 = quadratic (smooth),
        values > 2 penalize aggressive downweighting more heavily.
    min_weight : float, default 0.01
        Minimum allowed weight.
    max_weight : float, default 1.0
        Maximum allowed weight.

    Returns
    -------
    weights : np.ndarray, shape (ntime, nfreq)
        Fitted soft channel weights.
    diagnostics : dict
        Contains 'residual_power_tf', 'residual_power_f', 'weights_f',
        'prior_weights_f', and 'loss_history' if optimization was iterative.
    """
    from scipy.optimize import minimize

    resid = np.asarray(residual)
    if resid.ndim != 3:
        raise ValueError(f"Expected residual with shape (ntime, nfreq, nbls), got {resid.shape}")
    ntime, nfreq, nbls = resid.shape

    inv_var_tf = _as_tf(inv_noise_var, ntime, nfreq, fill_value=1.0)
    prior_tf = _as_tf(prior_weights, ntime, nfreq, fill_value=1.0)

    # Compute per-channel residual power (mean |residual|^2 across baselines)
    residual_power_tf = np.mean(np.abs(resid) ** 2, axis=2) * inv_var_tf
    residual_power_f = np.median(residual_power_tf, axis=0)

    # Normalize to avoid scale-dependence of regularization
    residual_power_f = np.clip(residual_power_f, 1e-30, np.inf)
    norm_factor = np.median(residual_power_f)
    norm_power_f = residual_power_f / max(norm_factor, 1e-30)

    # Define per-frequency optimization problem
    def loss_and_grad(w_flat):
        """Loss function for soft weight optimization.

        w_flat: weights per frequency, shape (nfreq,)
        Loss = sum_f norm_power[f] * w[f] + reg * sum_f (1 - w[f])^power
        Bounds are enforced by L-BFGS-B, so no clipping here.
        """
        data_term = np.sum(norm_power_f * w_flat)
        reg_term = regularization * np.sum(np.power(np.maximum(1.0 - w_flat, 0.0), regularization_power))
        loss = data_term + reg_term
        return float(loss)

    # Prior acts as an upper bound cap (e.g., flagged channel with prior=0.05 cannot exceed 0.05)
    upper_f = np.clip(prior_tf[0, :], min_weight, max_weight)

    # Optimize per-frequency weights with proper bounds
    # Lower bound: min_weight
    # Upper bound: prior weight (hard constraint from external flagging)
    w_init = np.clip(np.ones(nfreq, dtype=DTYPE_R), min_weight, upper_f).astype(DTYPE_R)
    w_bounds = [(min_weight, upper_f[f]) for f in range(nfreq)]

    result = minimize(
        loss_and_grad,
        w_init,
        method='L-BFGS-B',
        bounds=w_bounds,
        options={'ftol': 1e-8, 'maxiter': 500},
    )

    weights_f = np.clip(result.x, min_weight, upper_f).astype(DTYPE_R)
    weights_tf = np.broadcast_to(weights_f[None, :], (ntime, nfreq)).astype(DTYPE_R)
    weights_final = weights_tf.astype(DTYPE_R)

    diagnostics = {
        'residual_power_tf': residual_power_tf,
        'residual_power_f': residual_power_f,
        'weights_f': weights_f,
        'weights_final_f': weights_final[0, :],
        'prior_weights_f': prior_tf[0, :],
        'optimization_success': result.success,
        'optimization_message': result.message if hasattr(result, 'message') else '',
        'final_loss': float(result.fun),
    }
    return weights_final, diagnostics


def fit_soft_channel_weights_closed_form_jax(
    *,
    residual,
    inv_noise_var=None,
    prior_weights=None,
    regularization: float = 1.0,
    min_weight: float = 0.01,
    max_weight: float = 1.0,
) -> tuple:
    """Fit soft channel weights with closed-form solution for quadratic regularization.

    For the common case of regularization_power=2, the optimal weights have a
    closed-form solution: w_f = 1 - a_f / (2*λ), then clipped to bounds.
    This is faster, deterministic, fully JAX-compatible, and JIT-friendly.

    Avoids iterative optimization entirely—no convergence issues, no initialization.

    Parameters
    ----------
    residual : array_like, shape (ntime, nfreq, nbls)
        Gain-calibrated model residuals.
    inv_noise_var : array_like, optional
        Inverse variance per time/frequency channel.
    prior_weights : array_like, optional
        External bootstrap prior; acts as upper bound on weights.
    regularization : float, default 1.0
        Strength of quadratic regularization λ.
    min_weight : float, default 0.01
        Minimum allowed weight.
    max_weight : float, default 1.0
        Maximum allowed weight.

    Returns
    -------
    weights : array, shape (ntime, nfreq)
        Fitted soft channel weights (JAX array).
    diagnostics : dict
        Contains residual statistics and final weights.
    """
    import jax.numpy as jnp

    resid = jnp.asarray(residual)
    if resid.ndim != 3:
        raise ValueError(f"Expected residual with shape (ntime, nfreq, nbls), got {resid.shape}")
    ntime, nfreq, _ = resid.shape

    # Compute residual power per channel
    inv_var_tf = jnp.ones((ntime, nfreq)) if inv_noise_var is None else jnp.asarray(inv_noise_var)
    if inv_var_tf.ndim == 1:
        inv_var_tf = jnp.broadcast_to(inv_var_tf[None, :], (ntime, nfreq))

    residual_power_tf = jnp.mean(jnp.abs(resid) ** 2, axis=2) * inv_var_tf
    residual_power_f = jnp.median(residual_power_tf, axis=0)

    # Normalize to scale-invariant form
    norm_factor = jnp.maximum(jnp.median(residual_power_f), 1e-30)
    a_f = residual_power_f / norm_factor

    # Prior acts as upper bound
    if prior_weights is None:
        upper_f = jnp.ones(nfreq, dtype=DTYPE_R)
    else:
        prior_arr = jnp.asarray(prior_weights, dtype=DTYPE_R)
        if prior_arr.ndim == 2:
            upper_f = prior_arr[0, :]  # Take first time slice
        else:
            upper_f = prior_arr
    upper_f = jnp.clip(upper_f, min_weight, max_weight)

    # Closed-form solution for quadratic regularization
    # Unconstrained optimum: w_f = 1 - a_f / (2*λ)
    lam = jnp.maximum(regularization, 1e-30)
    weights_f_unconstrained = 1.0 - a_f / (2.0 * lam)

    # Clip to valid range [min_weight, upper_f]
    weights_f = jnp.clip(weights_f_unconstrained, min_weight, upper_f)
    weights_tf = jnp.broadcast_to(weights_f[None, :], (ntime, nfreq))

    diagnostics = {
        'residual_power_tf': jnp.asarray(residual_power_tf),
        'residual_power_f': jnp.asarray(residual_power_f),
        'weights_f': jnp.asarray(weights_f),
        'weights_final_f': jnp.asarray(weights_f),
        'upper_f': jnp.asarray(upper_f),
        'method': 'closed_form_quadratic',
    }
    return weights_tf, diagnostics


def fit_soft_channel_weights_jax(
    *,
    residual,
    inv_noise_var=None,
    prior_weights=None,
    regularization: float = 1.0,
    regularization_power: float = 2.0,
    min_weight: float = 0.01,
    max_weight: float = 1.0,
    use_jax: bool = True,
) -> tuple:
    """Fit soft channel weights via regularized optimization (JAX-compatible).

    JAX-native implementation using jaxopt.LBFGS for GPU/TPU acceleration.
    Functionally identical to fit_soft_channel_weights but maintains JAX arrays
    throughout computation and supports JIT compilation.

    Parameters
    ----------
    residual : array_like, shape (ntime, nfreq, nbls)
        Gain-calibrated model residuals (JAX array or convertible).
    inv_noise_var : array_like, optional
        Inverse variance per time/frequency channel (JAX array or convertible).
    prior_weights : array_like, optional
        External bootstrap prior from DPSS / high-pass flagging (JAX array).
    regularization : float, default 1.0
        Strength of regularization penalizing deviation from weight=1.
    regularization_power : float, default 2.0
        Power of the regularization term. 2.0 = quadratic (smooth),
        values > 2 penalize aggressive downweighting more heavily.
    min_weight : float, default 0.01
        Minimum allowed weight.
    max_weight : float, default 1.0
        Maximum allowed weight.
    use_jax : bool, default True
        If True, use jaxopt.LBFGS (JAX-native). If False, use scipy (NumPy-compatible).

    Returns
    -------
    weights : array, shape (ntime, nfreq)
        Fitted soft channel weights (JAX array if use_jax=True, else NumPy).
    diagnostics : dict
        Contains optimization results and intermediate values.
    """
    try:
        import jax
        import jax.numpy as jnp
        import jaxopt
    except ImportError:
        raise ImportError("JAX and jaxopt required for fit_soft_channel_weights_jax. "
                         "Install with: pip install jax jaxopt")

    if not use_jax:
        return fit_soft_channel_weights(
            residual=residual,
            inv_noise_var=inv_noise_var,
            prior_weights=prior_weights,
            regularization=regularization,
            regularization_power=regularization_power,
            min_weight=min_weight,
            max_weight=max_weight,
        )

    # Convert inputs to JAX arrays
    resid = jnp.asarray(residual)
    if resid.ndim != 3:
        raise ValueError(f"Expected residual with shape (ntime, nfreq, nbls), got {resid.shape}")
    ntime, nfreq, nbls = resid.shape

    # Prepare arrays in JAX
    inv_var_tf = jnp.asarray(_as_tf(inv_noise_var, ntime, nfreq, fill_value=1.0))
    prior_tf = jnp.asarray(_as_tf(prior_weights, ntime, nfreq, fill_value=1.0))

    # Compute per-channel residual power (median across time)
    residual_power_tf = jnp.mean(jnp.abs(resid) ** 2, axis=2) * inv_var_tf
    residual_power_f = jnp.median(residual_power_tf, axis=0)

    # Normalize to avoid scale-dependence of regularization
    norm_factor = jnp.median(residual_power_f)
    norm_power_f = residual_power_f / jnp.maximum(norm_factor, 1e-30)

    # Prior acts as upper bound
    upper_f = jnp.clip(prior_tf[0, :], min_weight, max_weight)

    # Use logit parameterization to avoid hard clipping in gradient:
    # w = min_weight + (upper - min_weight) * sigmoid(z)
    # This keeps gradients smooth everywhere.
    def loss_fn_logit(z_flat):
        """Loss in terms of unconstrained logits z."""
        import jax.scipy as jsp
        sigmoid_z = jsp.special.expit(z_flat)  # sigmoid
        w = min_weight + (upper_f - min_weight) * sigmoid_z
        data_term = jnp.sum(norm_power_f * w)
        reg_term = regularization * jnp.sum(
            jnp.power(jnp.maximum(1.0 - w, 0.0), regularization_power)
        )
        return data_term + reg_term

    # Initialize logits at z=0 (w=midpoint between min and upper)
    z_init = jnp.zeros(nfreq, dtype=DTYPE_R)

    # Optimize unconstrained logits
    solver = jaxopt.LBFGS(fun=loss_fn_logit, maxiter=500, tol=1e-8)
    result = solver.run(z_init)
    z_opt = result.params

    # Convert back to weights
    import jax.scipy as jsp
    sigmoid_z = jsp.special.expit(z_opt)
    weights_f = min_weight + (upper_f - min_weight) * sigmoid_z
    weights_f = jnp.clip(weights_f, min_weight, upper_f)  # numerical safety
    weights_tf = jnp.broadcast_to(weights_f[None, :], (ntime, nfreq))
    weights_final = weights_tf

    # Convert diagnostics to NumPy for output (JAX arrays can be expensive to inspect)
    diagnostics = {
        'residual_power_tf': np.asarray(residual_power_tf),
        'residual_power_f': np.asarray(residual_power_f),
        'weights_f': np.asarray(weights_f),
        'weights_final_f': np.asarray(weights_final[0, :]),
        'prior_weights_f': np.asarray(prior_tf[0, :]),
        'final_loss': float(result.state.value),
        'num_iterations': int(result.state.iter_num),
    }

    return weights_final, diagnostics


def fit_soft_channel_weights_thresholded(
    *,
    residual: np.ndarray,
    inv_noise_var: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    threshold: float = 1.5,
    softness: float = 2.0,
    min_weight: float = 0.10,
    max_weight: float = 1.0,
    statistic: str = 'median',
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust thresholded soft channel weights via logistic excess penalty.

    Computes per-frequency whitened residual power, normalizes by a robust
    statistic, and only downweights channels above a threshold. Uses a logistic
    rule to smoothly penalize excess, avoiding the over-aggressive behavior of
    quadratic regularization. Intended for RFI weight fitting.

    Parameters
    ----------
    residual : np.ndarray, shape (ntime, nfreq, nbls)
        Gain-calibrated model residuals (unweighted).
    inv_noise_var : np.ndarray, optional
        Inverse variance per time/frequency channel. If None, assume uniform.
    prior_weights : np.ndarray, optional
        External bootstrap prior (e.g., from DPSS or flagging).
        Fitted weights will not exceed prior values.
    threshold : float, default 1.5
        Normalized residual power above which downweighting begins.
        Channels with a_f <= threshold keep full weight.
    softness : float, default 2.0
        Controls steepness of logistic downweighting. Larger values = gentler.
        w_target = 1 / (1 + excess / softness), where excess = max(a_f - threshold, 0).
    min_weight : float, default 0.10
        Minimum allowed weight (avoid making channels dynamically irrelevant).
    max_weight : float, default 1.0
        Maximum allowed weight.
    statistic : {'median', 'mean'}, default 'median'
        Robust statistic for normalizing residual power per frequency.

    Returns
    -------
    weights : np.ndarray, shape (ntime, nfreq)
        Fitted soft channel weights (constant across time).
    diagnostics : dict
        Contains 'residual_power_tf', 'residual_power_f', 'norm_power_f',
        'weights_f', 'threshold', 'softness', 'method', etc.
    """
    resid = np.asarray(residual)
    if resid.ndim != 3:
        raise ValueError(
            f"Expected residual with shape (ntime, nfreq, nbls), got {resid.shape}"
        )
    ntime, nfreq, nbls = resid.shape

    inv_var_tf = _as_tf(inv_noise_var, ntime, nfreq, fill_value=1.0)
    prior_tf = _as_tf(prior_weights, ntime, nfreq, fill_value=max_weight)

    # Compute per-channel residual power (mean |residual|^2 across baselines)
    residual_power_tf = np.mean(np.abs(resid) ** 2, axis=2) * inv_var_tf

    # Normalize by robust statistic (median or mean per frequency)
    if statistic == 'median':
        residual_power_f = np.median(residual_power_tf, axis=0)
    elif statistic == 'mean':
        residual_power_f = np.mean(residual_power_tf, axis=0)
    else:
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic}")

    # Avoid division by zero
    floor = np.maximum(np.median(residual_power_f), 1e-30)
    a_f = residual_power_f / floor

    # Logistic rule: only penalize channels above threshold
    excess = np.maximum(a_f - threshold, 0.0)
    weights_f = 1.0 / (1.0 + excess / softness)

    # Apply prior bounds (prior acts as upper-bound cap from flagging)
    upper_f = np.clip(prior_tf[0, :], min_weight, max_weight)
    weights_f = np.clip(weights_f, min_weight, upper_f)

    # Broadcast to full time/frequency grid
    weights_tf = np.broadcast_to(weights_f[None, :], (ntime, nfreq)).astype(DTYPE_R)

    diagnostics = {
        'residual_power_tf': residual_power_tf,
        'residual_power_f': residual_power_f,
        'norm_power_f': a_f,
        'weights_f': weights_f,
        'weights_final_f': weights_f,
        'upper_f': upper_f,
        'threshold': threshold,
        'softness': softness,
        'statistic': statistic,
        'method': 'thresholded_logistic_excess',
    }
    return weights_tf, diagnostics


def fit_channel_weights_dof_conservative(
    *,
    residual: np.ndarray,
    A: np.ndarray,
    inv_noise_var: np.ndarray | None = None,
    old_weights: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    min_weight: float = 0.25,
    max_weight: float = 1.0,
    threshold: float = 3.0,
    softness: float = 3.0,
    leverage_floor: float = 0.05,
    alpha_down: float = 0.10,
    alpha_up: float = 0.70,
    max_drop_per_update: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservative DoF-aware spectral-inconsistency channel weights.

    Only downweight channels when removing them costs very little information
    about the spectrally smooth model but removes statistically significant
    out-of-subspace residual power. Uses spectral leverage (from the hat matrix)
    to distinguish high-residual channels that the model depends on from those
    it doesn't.

    Parameters
    ----------
    residual : np.ndarray, shape (ntime, nfreq, nbls)
        Unweighted gain-calibrated residuals.
    A : np.ndarray, shape (nfreq, nmodes)
        Spectral model basis (e.g., DPSS sky or beam basis).
    inv_noise_var : np.ndarray, optional
        Inverse variance per channel, shape (nfreq,) or (ntime, nfreq).
    old_weights : np.ndarray, optional
        Current per-frequency weights for asymmetric smoothing, shape (nfreq,).
    prior_weights : np.ndarray, optional
        Upper bound on weights (e.g., from flagging), shape (nfreq,) or (ntime, nfreq).
    min_weight : float, default 0.25
        Minimum allowed weight (conservative: 0.25 instead of 0.01).
    max_weight : float, default 1.0
        Maximum allowed weight.
    threshold : float, default 3.0
        Statistic threshold for downweighting (in units of median residual_dof).
        Only channels with z > threshold are downweighted. Conservative default.
    softness : float, default 3.0
        Steepness of logistic transition (larger = gentler).
    leverage_floor : float, default 0.05
        Minimum effective residual DoF (prevents division by zero).
    alpha_down : float, default 0.10
        Asymmetric smoothing: slow downweighting (requires repeated evidence).
    alpha_up : float, default 0.70
        Asymmetric smoothing: fast upweighting/recovery.
    max_drop_per_update : float, default 0.10
        Maximum weight drop allowed in a single update (prevents overshooting).

    Returns
    -------
    weights : np.ndarray, shape (ntime, nfreq)
        Fitted soft channel weights (broadcast across time).
    diagnostics : dict
        Contains 'leverage_f', 'residual_dof_f', 'hp_power_f', 'z_f',
        'target_weights_f', 'weights_f', and method identifier.
    """
    residual = np.asarray(residual)
    A = np.asarray(A)

    ntime, nfreq, nbls = residual.shape

    if inv_noise_var is None:
        inv_noise_var = np.ones((ntime, nfreq), dtype=DTYPE_R)
    else:
        inv_noise_var = _as_tf(inv_noise_var, ntime, nfreq, fill_value=1.0)

    if old_weights is None:
        old_weights = np.ones(nfreq, dtype=DTYPE_R)
    else:
        old_weights = np.asarray(old_weights, dtype=DTYPE_R)
        if old_weights.ndim == 2:
            old_weights = np.median(old_weights, axis=0)

    if prior_weights is None:
        upper = np.full(nfreq, max_weight, dtype=DTYPE_R)
    else:
        prior_weights = np.asarray(prior_weights, dtype=DTYPE_R)
        if prior_weights.ndim == 2:
            upper = prior_weights[0, :]
        else:
            upper = prior_weights
        upper = np.clip(upper, min_weight, max_weight)

    # Spectral projection and complement
    # P = A (A^T A)^{-1} A^T projects onto the span of A
    # Q = I - P projects onto the orthogonal complement
    G = A.T @ A
    P = A @ np.linalg.pinv(G, rcond=1e-8) @ A.T
    Q = np.eye(nfreq, dtype=DTYPE_R) - P

    # Leverage: h_f = P_ff tells how much channel f is needed by the model
    # Residual DoF: (1 - h_f) tells how much independent residual DoF remains
    h = np.clip(np.diag(P), 0.0, 1.0).astype(DTYPE_R)
    residual_dof = np.maximum(1.0 - h, leverage_floor).astype(DTYPE_R)

    # High-pass spectral residual: component not supported by smooth basis
    # r_hp = Q @ residual (per time/baseline/frequency)
    r_hp = np.einsum("fg,tgb->tfb", Q, residual)

    # Whitened high-pass residual power per frequency (median over time)
    hp_power_tf = np.mean(np.abs(r_hp) ** 2, axis=2) * inv_noise_var
    hp_power_f = np.median(hp_power_tf, axis=0).astype(DTYPE_R)

    # DoF-normalized out-of-subspace statistic
    # z_f = hp_power_f / residual_dof_f
    z = hp_power_f / residual_dof
    z = z / np.maximum(np.median(z), 1e-30)

    # Conservative threshold: only act on clear outliers (z > 3 is ~5-sigma)
    excess = np.maximum(z - threshold, 0.0)
    target = 1.0 / (1.0 + excess / softness)

    # Information-cost correction: high-leverage channels are harder to downweight
    # Channels that contribute strongly to the model should require strong evidence
    # before being downweighted
    info_cost = h / np.maximum(np.median(h), 1e-6)
    target = 1.0 - (1.0 - target) / (1.0 + info_cost)

    target = np.clip(target, min_weight, upper).astype(DTYPE_R)

    # Asymmetric temporal smoothing
    # Downweighting is slow (requires repeated evidence)
    # Upweighting/recovery is fast (once the outlier goes away)
    alpha = np.where(target < old_weights, alpha_down, alpha_up)
    w = (1.0 - alpha) * old_weights + alpha * target

    # Limit how far a channel can fall in a single update
    w = np.maximum(w, old_weights - max_drop_per_update)
    w = np.clip(w, min_weight, upper).astype(DTYPE_R)

    # Broadcast to (ntime, nfreq)
    weights_tf = np.broadcast_to(w[None, :], (ntime, nfreq)).copy().astype(DTYPE_R)

    diagnostics = {
        "method": "dof_conservative_spectral_complement",
        "leverage_f": h,
        "residual_dof_f": residual_dof,
        "hp_power_f": hp_power_f,
        "z_f": z,
        "target_weights_f": target,
        "weights_f": w,
        "threshold": threshold,
        "softness": softness,
    }
    return weights_tf, diagnostics
