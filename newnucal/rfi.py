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

def fit_channel_weights_local_chi2_exponential(
    *,
    residual: np.ndarray,
    inv_noise_var: np.ndarray | None = None,
    old_weights: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    min_weight: float = 0.05,
    max_weight: float = 1.0,
    smooth_width_chans: int = 17,
    threshold_ratio: float = 3.0,
    gamma: float = 0.75,
    alpha_down: float = 0.20,
    alpha_up: float = 0.75,
    max_drop_per_update: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exponential soft channel weights from local chi-squared outliers.

    Detects channels with chi-squared significantly above their local neighbors
    in log space, then downweights exponentially. No spectral basis required.

    Parameters
    ----------
    residual : np.ndarray, shape (ntime, nfreq, nbls)
        Unweighted gain-calibrated residuals.
    inv_noise_var : np.ndarray, optional
        Inverse variance per channel, shape (nfreq,) or (ntime, nfreq).
    old_weights : np.ndarray, optional
        Current weights for asymmetric smoothing, shape (nfreq,).
    prior_weights : np.ndarray, optional
        Upper bound on weights, shape (nfreq,) or (ntime, nfreq).
    min_weight : float, default 0.05
        Minimum allowed weight.
    max_weight : float, default 1.0
        Maximum allowed weight.
    smooth_width_chans : int, default 17
        Width of median+Gaussian filter for local baseline (must be odd).
    threshold_ratio : float, default 3.0
        Threshold chi2_f / smooth_chi2_f above which downweighting begins.
    gamma : float, default 0.75
        Exponential decay rate: w = exp(-gamma * excess).
        Larger gamma = more aggressive downweighting.
    alpha_down : float, default 0.20
        Asymmetric smoothing: slow downweighting.
    alpha_up : float, default 0.75
        Asymmetric smoothing: fast upweighting/recovery.
    max_drop_per_update : float, default 0.20
        Maximum weight drop in a single update.

    Returns
    -------
    weights : np.ndarray, shape (ntime, nfreq)
        Fitted soft channel weights (broadcast across time).
    diagnostics : dict
        Contains chi2_f, log_chi2_f, smooth_log_chi2_f, ratio_f, target_weights_f.
    """
    from scipy.ndimage import median_filter, gaussian_filter1d

    residual = np.asarray(residual)
    if residual.ndim != 3:
        raise ValueError(
            f"Expected residual with shape (ntime, nfreq, nbls), got {residual.shape}"
        )

    ntime, nfreq, nbls = residual.shape

    if inv_noise_var is None:
        inv_noise_var_tf = np.ones((ntime, nfreq), dtype=DTYPE_R)
    else:
        inv_noise_var = np.asarray(inv_noise_var, dtype=DTYPE_R)
        if inv_noise_var.ndim == 1:
            inv_noise_var_tf = np.broadcast_to(inv_noise_var[None, :], (ntime, nfreq))
        elif inv_noise_var.shape == (ntime, nfreq):
            inv_noise_var_tf = inv_noise_var
        else:
            raise ValueError(
                f"inv_noise_var must have shape ({nfreq},) or ({ntime}, {nfreq})"
            )

    if old_weights is None:
        old_w = np.ones(nfreq, dtype=DTYPE_R)
    else:
        old_w = np.asarray(old_weights, dtype=DTYPE_R)
        if old_w.ndim == 2:
            old_w = np.median(old_w, axis=0).astype(DTYPE_R)

    if prior_weights is None:
        upper = np.full(nfreq, max_weight, dtype=DTYPE_R)
    else:
        upper = np.asarray(prior_weights, dtype=DTYPE_R)
        if upper.ndim == 2:
            upper = np.median(upper, axis=0).astype(DTYPE_R)
        upper = np.clip(upper, min_weight, max_weight)

    # Per-time/frequency whitened residual power (not yet weighted by channel_weights)
    chi2_tf = np.mean(np.abs(residual) ** 2, axis=2) * inv_noise_var_tf

    # Robust collapse over time (median is robust to outliers)
    chi2_f = np.median(chi2_tf, axis=0).astype(DTYPE_R)

    # Log chi-squared for outlier detection in log space
    eps = 1e-30
    log_chi2 = np.log(np.maximum(chi2_f, eps))

    # Robust local baseline: median filter rejects narrow RFI spikes,
    # Gaussian smoothing gives smooth trend
    width = int(smooth_width_chans)
    if width % 2 == 0:
        width += 1

    med_log = median_filter(log_chi2, size=width, mode="nearest")
    smooth_log = gaussian_filter1d(
        med_log,
        sigma=np.maximum(width / 4.0, 1.0),
        mode="nearest",
    ).astype(DTYPE_R)

    # Log-ratio: how much this channel exceeds local baseline
    log_ratio = log_chi2 - smooth_log
    ratio = np.exp(np.clip(log_ratio, -50, 50))

    # Exponential downweight above local-outlier threshold
    # Channels at or below threshold_ratio get weight ≈ 1
    # Channels above threshold decay exponentially in log-ratio space
    log_threshold = np.log(threshold_ratio)
    excess = np.maximum(log_ratio - log_threshold, 0.0)

    # w = exp(-gamma * excess)
    # Equivalent: w = (ratio / threshold_ratio) ** (-gamma) for ratio > threshold
    target = np.exp(-gamma * np.asarray(excess)).astype(DTYPE_R)
    target = np.clip(target, min_weight, upper)

    # Asymmetric temporal smoothing
    # Downweighting is slower (alpha_down ≈ 0.2) → requires repeated evidence
    # Upweighting/recovery is faster (alpha_up ≈ 0.75) → channels heal quickly
    alpha = np.where(target < old_w, alpha_down, alpha_up)
    w = (1.0 - alpha) * old_w + alpha * target

    # Prevent a single update from destroying a channel
    w = np.maximum(w, old_w - max_drop_per_update)
    w = np.clip(w, min_weight, upper).astype(DTYPE_R)

    # Broadcast to (ntime, nfreq)
    weights_tf = np.broadcast_to(w[None, :], (ntime, nfreq)).copy().astype(DTYPE_R)

    diagnostics = {
        "method": "local_chi2_exponential",
        "chi2_f": chi2_f,
        "log_chi2_f": log_chi2,
        "smooth_log_chi2_f": smooth_log,
        "log_ratio_f": log_ratio,
        "ratio_f": ratio,
        "excess_f": excess,
        "target_weights_f": target,
        "weights_f": w,
        "threshold_ratio": threshold_ratio,
        "gamma": gamma,
        "smooth_width_chans": smooth_width_chans,
    }

    return weights_tf, diagnostics
