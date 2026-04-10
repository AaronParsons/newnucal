"""
Per-frequency gain degeneracies.

After redcal has solved for per-antenna gains, four scalar DOF per
frequency remain unfixed (see arXiv:1712.07212):

  * log_amp[nfreq]   — log of overall amplitude  (real)
  * phase[nfreq]     — overall phase offset       (real, radians)
  * phi[2, nfreq]    — East/North phase gradient  (real, radians/metre)

Applied to a baseline (i→j) with separation bls[bl, :2] = (Δe, Δn):

  V_cal[t, freq, bl] = exp(log_amp[freq])
                       * exp(i * (phase[freq]
                                  + phi[0, freq] * bls[bl, 0]
                                  + phi[1, freq] * bls[bl, 1]))
                       * V_model[t, freq, bl]

These parameters are independent per frequency channel, allowing the
model to be brought into alignment with pre-calibrated data without
imposing spectral smoothness constraints on the gain itself.
"""

import numpy as np
import jax.numpy as jnp


def apply_gains(vis, log_amp, phase, phi, bls):
    """
    Apply gain degeneracies to model visibilities.

    Parameters
    ----------
    vis : jnp.array, shape (ntime, nfreq, nbls) or (nfreq, nbls)
        Model visibilities (complex).
    log_amp : jnp.array, shape (nfreq,)
        Log of overall amplitude correction.
    phase : jnp.array, shape (nfreq,)
        Overall phase correction in radians.
    phi : jnp.array, shape (2, nfreq)
        East/North phase gradient in radians per metre.
    bls : jnp.array, shape (nbls, 3) or (nbls, 2)
        Baseline vectors in metres.  Only the first two (East, North)
        components are used.

    Returns
    -------
    vis_cal : jnp.array, same shape as *vis*
    """
    # Baseline-dependent phase: (nfreq, nbls)
    phs_grad = jnp.einsum("xf,bx->fb", phi, bls[:, :2].astype(jnp.float32))
    # Total gain factor per (freq, baseline): amplitude × phase
    gain = jnp.exp(log_amp[:, None].astype(jnp.float32)) * jnp.exp(
        1j * (phase[:, None] + phs_grad).astype(jnp.float32)
    )  # (nfreq, nbls), complex64

    if vis.ndim == 3:
        # (ntime, nfreq, nbls) * (1, nfreq, nbls)
        return vis * gain[None, :, :]
    return vis * gain  # (nfreq, nbls)


def init_gain_params(nfreq: int):
    """
    Return a zero-initialised gain parameter dict.

    Parameters
    ----------
    nfreq : int

    Returns
    -------
    params : dict with keys 'log_amp', 'phase', 'phi'
    """
    return {
        "log_amp": jnp.zeros(nfreq, dtype=jnp.float32),
        "phase": jnp.zeros(nfreq, dtype=jnp.float32),
        "phi": jnp.zeros((2, nfreq), dtype=jnp.float32),
    }
