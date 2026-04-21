"""
Per-frequency, per-time gain degeneracies.

After redcal has solved for per-antenna gains, four scalar DOF per
frequency remain unfixed (see arXiv:1712.07212):

  * log_amp[ntime, nfreq]   — log of overall amplitude  (real)
  * phase[ntime, nfreq]     — overall phase offset       (real, radians)
  * phi[ntime, 2, nfreq]    — East/North phase gradient  (real, radians/metre)

Gains are solved independently at each time step so that slow temporal
variation (ionospheric drift, electronics) is captured, while the sky
model (equatorial HEALPix + DPSS) is shared across all times.

Applied to a baseline (i→j) with separation bls[bl, :2] = (Δe, Δn):

  V_cal[t, freq, bl] = exp(log_amp[t, freq])
                       * exp(i * (phase[t, freq]
                                  + phi[t, 0, freq] * bls[bl, 0]
                                  + phi[t, 1, freq] * bls[bl, 1]))
                       * V_model[t, freq, bl]
"""

import numpy as np
import jax.numpy as jnp
from .utils import DTYPE_R_JAX, DTYPE_C_JAX

def apply_gains(vis, log_amp, phase, phi, bls):
    """
    Apply gain degeneracies to model visibilities.

    Parameters
    ----------
    vis : jnp.array, shape (ntime, nfreq, nbls)
        Model visibilities (complex).
    log_amp : jnp.array, shape (ntime, nfreq)
        Log of overall amplitude correction per time and frequency.
    phase : jnp.array, shape (ntime, nfreq)
        Overall phase correction in radians per time and frequency.
    phi : jnp.array, shape (ntime, 2, nfreq)
        East/North phase gradient in radians per metre per time and frequency.
    bls : jnp.array, shape (nbls, 3) or (nbls, 2)
        Baseline vectors in metres.  Only the first two (East, North)
        components are used.

    Returns
    -------
    vis_cal : jnp.array, shape (ntime, nfreq, nbls)
    """
    # phi: (ntime, 2, nfreq), bls[:, :2]: (nbls, 2)
    # phs_grad[t, f, b] = sum_x phi[t, x, f] * bls[b, x]
    phs_grad = jnp.einsum("txf,bx->tfb", phi, bls[:, :2].astype(DTYPE_R_JAX))  # (ntime, nfreq, nbls)
    # Total gain factor per (time, freq, baseline)
    gain = jnp.exp(log_amp[:, :, None].astype(DTYPE_R_JAX)) * jnp.exp(
        1j * (phase[:, :, None] + phs_grad).astype(DTYPE_C_JAX)
    )  # (ntime, nfreq, nbls), complex64
    return vis * gain


def init_gain_params(ntime: int, nfreq: int):
    """
    Return a zero-initialised gain parameter dict.

    Parameters
    ----------
    ntime : int
    nfreq : int

    Returns
    -------
    params : dict with keys 'log_amp', 'phase', 'phi'
    """
    return {
        "log_amp": jnp.zeros((ntime, nfreq), dtype=DTYPE_R_JAX),
        "phase":   jnp.zeros((ntime, nfreq), dtype=DTYPE_R_JAX),
        "phi":     jnp.zeros((ntime, 2, nfreq), dtype=DTYPE_R_JAX),
    }
