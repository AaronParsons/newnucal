"""Multiresolution helpers for fitting across HEALPix resolutions."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import healpy

from .utils import DTYPE_R_JAX, DTYPE_R_NPY


def _ud_grade_spectra(spectra, nside_out: int, *, power: float = 0.0):
    spectra = np.asarray(spectra, dtype=DTYPE_R_NPY)
    if spectra.ndim != 2:
        raise ValueError(f"Expected spectra with shape (npix, nfreq), got {spectra.shape}")
    out = [
        healpy.ud_grade(spectra[:, fi], nside_out=nside_out, power=power)
        for fi in range(spectra.shape[1])
    ]
    return np.stack(out, axis=1).astype(DTYPE_R_NPY)


def _normalize_spectra_like_input(spectra_out, spectra_in, normalize):
    if normalize is None or normalize == "none":
        return spectra_out
    if normalize == "max":
        stat_in = np.max(np.abs(spectra_in), axis=0)
        stat_out = np.max(np.abs(spectra_out), axis=0)
    elif normalize == "mean":
        stat_in = np.mean(spectra_in, axis=0)
        stat_out = np.mean(spectra_out, axis=0)
    else:
        raise ValueError("normalize must be one of None, 'none', 'max', or 'mean'")
    scale = np.divide(
        stat_in,
        stat_out,
        out=np.ones_like(stat_in, dtype=DTYPE_R_NPY),
        where=np.abs(stat_out) > 1e-30,
    )
    return (spectra_out * scale[None, :]).astype(DTYPE_R_NPY)


def resample_sky_coeffs(coeffs_in, sky_model_in, sky_model_out, *, healpix_power: float = 0.0):
    """Resample sky coefficients between HEALPix resolutions.

    Coefficients are deprojected to frequency maps on the input sky,
    HEALPix-resampled per frequency, then projected into the output sky model's
    spectral basis. The resampling can increase or decrease NSIDE.
    """
    coeffs_in = np.asarray(coeffs_in, dtype=DTYPE_R_NPY)
    if coeffs_in.shape[0] != sky_model_in.npix:
        raise ValueError(
            f"sky coeff rows {coeffs_in.shape[0]} != input sky npix {sky_model_in.npix}"
        )
    spectra_in = sky_model_in.deproject(coeffs_in)
    spectra_out = _ud_grade_spectra(
        spectra_in, sky_model_out.nside, power=float(healpix_power)
    )
    return jnp.array(sky_model_out.project(spectra_out), dtype=DTYPE_R_JAX)


def resample_beam_coeffs(
    coeffs_in,
    beam_model_in,
    beam_model_out,
    *,
    healpix_power: float = 0.0,
    normalize: str | None = None,
):
    """Resample beam coefficients between HEALPix resolutions.

    The resampling can increase or decrease NSIDE.
    """
    coeffs_in = np.asarray(coeffs_in, dtype=DTYPE_R_NPY)
    if coeffs_in.shape[0] != beam_model_in.npix:
        raise ValueError(
            f"beam coeff rows {coeffs_in.shape[0]} != input beam npix {beam_model_in.npix}"
        )
    spectra_in = beam_model_in.deproject(coeffs_in)
    spectra_out = _ud_grade_spectra(
        spectra_in, beam_model_out.nside, power=float(healpix_power)
    )
    spectra_out = _normalize_spectra_like_input(spectra_out, spectra_in, normalize)
    return jnp.array(beam_model_out.project(spectra_out), dtype=DTYPE_R_JAX)


def resample_params(
    params_in,
    cal_in,
    cal_out,
    *,
    sky: bool = True,
    beam: bool = True,
    sky_healpix_power: float = 0.0,
    beam_healpix_power: float = 0.0,
    beam_normalize: str | None = None,
    params_input_full=None,
):
    """Transfer a parameter dict between calibrators at different resolutions.

    Sky and beam blocks are optionally resampled with ``healpy.ud_grade``, which
    supports both upsampling and downsampling. Gains and log-channel weights are
    copied unchanged. If a disabled sky or beam block is incompatible with the
    output model shape, it is initialized in the output model space.
    """
    params_full = cal_in._params_to_full_space(params_in, params_input_full)
    out = dict(params_full)

    if sky and params_full.get("sky_coeffs") is not None:
        out["sky_coeffs"] = resample_sky_coeffs(
            params_full["sky_coeffs"],
            cal_in.fwd.sky_model,
            cal_out.fwd.sky_model,
            healpix_power=sky_healpix_power,
        )
    elif params_full.get("sky_coeffs") is not None:
        if np.shape(params_full["sky_coeffs"])[0] == cal_out.fwd.sky_model.npix:
            out["sky_coeffs"] = jnp.array(params_full["sky_coeffs"], dtype=DTYPE_R_JAX)
        else:
            out["sky_coeffs"] = jnp.zeros(
                (cal_out.fwd.sky_model.npix, cal_out.fwd.sky_model.nmodes),
                dtype=DTYPE_R_JAX,
            )

    if beam and params_full.get("beam_coeffs") is not None:
        out["beam_coeffs"] = resample_beam_coeffs(
            params_full["beam_coeffs"],
            cal_in.beam_model,
            cal_out.beam_model,
            healpix_power=beam_healpix_power,
            normalize=beam_normalize,
        )
    elif params_full.get("beam_coeffs") is not None:
        if np.shape(params_full["beam_coeffs"])[0] == cal_out.beam_model.npix:
            out["beam_coeffs"] = cal_out._beam_coeffs_full(params_full["beam_coeffs"])
        else:
            out["beam_coeffs"] = jnp.array(cal_out.beam_model.coeffs, dtype=DTYPE_R_JAX)

    for key in cal_in._GAIN_PARAM_KEYS:
        if key in params_full:
            out[key] = params_full[key]
    if params_full.get("log_ch_weights") is not None:
        out["log_ch_weights"] = np.asarray(params_full["log_ch_weights"], dtype=DTYPE_R_NPY)
    else:
        out.pop("log_ch_weights", None)
    return out


def init_resampled_joint_state(
    cal_in,
    state_in,
    cal_out,
    *,
    resample_sky: bool = True,
    resample_beam: bool = True,
    sky_healpix_power: float = 0.0,
    beam_healpix_power: float = 0.0,
    beam_normalize: str | None = None,
    reset_anderson: bool = True,
    **fit_kwargs,
):
    """Initialize a joint state from another state's resampled parameters."""
    params_out = resample_params(
        state_in.params,
        cal_in,
        cal_out,
        sky=resample_sky,
        beam=resample_beam,
        sky_healpix_power=sky_healpix_power,
        beam_healpix_power=beam_healpix_power,
        beam_normalize=beam_normalize,
        params_input_full=state_in.settings.get("params_input_full"),
    )
    if "log_ch_weights" in params_out:
        cal_out.set_channel_weights(params_out["log_ch_weights"])

    settings = dict(fit_kwargs)
    if not reset_anderson and "joint_anderson_history" not in settings:
        settings["joint_anderson_history"] = state_in.joint_acc.history
    return cal_out.init_joint_sky_beam_dirty_state(params_out, **settings)
