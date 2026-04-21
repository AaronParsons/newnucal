"""RFI-aware channel weighting utilities for newnucal.

This module is built around a conservative workflow:

1. Start from an externally supplied per-channel RFI prior, typically from the
   HERA pipeline's DPSS / high-pass flagger.
2. Fit the newnucal model with those weights.
3. Update soft channel weights from whitened model residuals, after removing a
   smooth spectral trend so broad model mismatch is not immediately interpreted
   as narrowband RFI.
4. Refit and iterate.

The module intentionally does *not* attempt to model RFI itself.  Instead it
maintains soft reliability weights for time/frequency channels, suitable for
use in weighted losses and weighted dirty-map updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from .utils import DTYPE_R_NPY as DTYPE_R

@dataclass(slots=True)
class RFIConfig:
    """Configuration for residual-based channel reweighting.

    Parameters
    ----------
    flagged_weight : float
        Weight assigned to channels flagged by the external bootstrap mask.
    min_weight : float
        Minimum allowed soft weight.
    max_weight : float
        Maximum allowed soft weight.
    smooth_window : int
        Odd rolling-median window used to estimate a smooth spectral baseline.
    score_center : float
        Logistic half-power point for the residual excess score.
    score_slope : float
        Steepness of the logistic mapping from score to weight.
    prior_power : float
        Exponent applied to the external prior weights when combining with the
        residual-based weights.
    model_power : float
        Exponent applied to the residual-based weights when combining.
    blend : float
        Blend between current weights and newly inferred weights; 0 uses only
        the new weights, 1 keeps the current weights unchanged.
    time_reduce : {"median", "mean"}
        How to collapse per-time statistics to a per-frequency score.
    """

    flagged_weight: float = 0.05
    min_weight: float = 0.01
    max_weight: float = 1.0
    smooth_window: int = 9
    score_center: float = 3.0
    score_slope: float = 2.0
    prior_power: float = 1.0
    model_power: float = 1.0
    blend: float = 0.3
    time_reduce: str = "median"



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



def smooth_frequency_statistic(stat_f: np.ndarray, window: int = 9) -> np.ndarray:
    """Rolling-median spectral baseline for a 1-D channel statistic."""
    stat_f = np.asarray(stat_f, dtype=DTYPE_R)
    if stat_f.ndim != 1:
        raise ValueError(f"Expected 1-D statistic, got {stat_f.shape}")
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(stat_f, pad_width=pad, mode="edge")
    out = np.empty_like(stat_f)
    for i in range(stat_f.size):
        out[i] = np.median(padded[i:i + window])
    return out



def score_to_soft_weights(
    score: np.ndarray,
    *,
    center: float = 3.0,
    slope: float = 2.0,
    min_weight: float = 0.01,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Map a positive excess score to a soft weight in ``[min_weight, max_weight]``."""
    score = np.asarray(score, dtype=DTYPE_R)
    center = max(float(center), 1e-6)
    slope = max(float(slope), 1e-6)
    core = 1.0 / (1.0 + np.power(np.maximum(score, 0.0) / center, slope))
    return np.clip(core, min_weight, max_weight).astype(DTYPE_R)



def update_channel_weights_from_residuals(
    *,
    residual: np.ndarray,
    inv_noise_var: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    current_weights: np.ndarray | None = None,
    config: RFIConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer new soft channel weights from residuals and an external prior.

    Parameters
    ----------
    residual : array_like, shape (ntime, nfreq, nbls)
        Gain-calibrated model residuals.
    inv_noise_var : array_like, optional
        Inverse variance per time/frequency channel.
    prior_weights : array_like, optional
        External bootstrap prior from DPSS / high-pass flagging.
    current_weights : array_like, optional
        Current iteration's soft weights.
    config : RFIConfig, optional

    Returns
    -------
    new_weights : np.ndarray, shape (ntime, nfreq)
    diagnostics : dict
        Contains ``chi2_tf``, ``chi2_f``, ``smooth_f``, ``score_f``,
        ``model_weights_f`` and ``combined_weights``.
    """
    cfg = config or RFIConfig()
    resid = np.asarray(residual)
    ntime, nfreq, _ = resid.shape

    prior_tf = _as_tf(prior_weights, ntime, nfreq, fill_value=1.0)
    current_tf = _as_tf(current_weights, ntime, nfreq, fill_value=1.0)
    chi2_tf = channel_chi2_statistic(resid, inv_noise_var=inv_noise_var)

    if cfg.time_reduce == "mean":
        chi2_f = np.mean(chi2_tf, axis=0)
    else:
        chi2_f = np.median(chi2_tf, axis=0)

    smooth_f = smooth_frequency_statistic(chi2_f, window=cfg.smooth_window)
    score_f = np.maximum(chi2_f / np.maximum(smooth_f, 1e-6) - 1.0, 0.0)
    model_weights_f = score_to_soft_weights(
        score_f,
        center=cfg.score_center,
        slope=cfg.score_slope,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
    )
    model_tf = np.broadcast_to(model_weights_f[None, :], (ntime, nfreq)).astype(DTYPE_R)

    combined = np.power(np.clip(prior_tf, cfg.min_weight, cfg.max_weight), cfg.prior_power)
    combined *= np.power(np.clip(model_tf, cfg.min_weight, cfg.max_weight), cfg.model_power)
    combined = np.clip(combined, cfg.min_weight, cfg.max_weight)
    new_weights = cfg.blend * current_tf + (1.0 - cfg.blend) * combined
    new_weights = np.clip(new_weights, cfg.min_weight, cfg.max_weight).astype(DTYPE_R)

    diagnostics = {
        "chi2_tf": chi2_tf,
        "chi2_f": chi2_f,
        "smooth_f": smooth_f,
        "score_f": score_f,
        "model_weights_f": model_weights_f,
        "combined_weights": combined,
    }
    return new_weights, diagnostics
