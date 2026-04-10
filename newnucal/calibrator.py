"""
Calibrator — ties together ForwardModel, gain parameters, and optimisation.

Usage
-----
    cal = Calibrator(array, beam_model, sky_nside, sky_eta_max,
                     freqs, rot_matrices, data)
    params = cal.init_params()
    params, loss = cal.fit_lbfgs(params, maxiter=30)
    params, loss = cal.fit_optax(params, maxiter=60, lr=3e-2)

The parameter pytree has three entries:

    sky_coeffs  : (npix_sky, nmodes_sky)  float32 — sky DPSS coefficients
    log_amp     : (nfreq,)                float32 — gain amplitude
    phase       : (nfreq,)                float32 — gain overall phase
    phi         : (2, nfreq)              float32 — gain phase gradient

The loss is  mean |V_data - gain * V_model|^2  summed over time, frequency,
and baselines.
"""

import jax
import jax.numpy as jnp
import optax
from jaxopt import LBFGS

from .array import HERAArray
from .beam import BeamModel
from .dpss import dpss_matrix
from .simulate import ForwardModel
from .gains import apply_gains, init_gain_params

DTYPE_R = jnp.float32
DTYPE_C = jnp.complex64


class Calibrator:
    """
    End-to-end calibration object.

    Parameters
    ----------
    array : HERAArray
    beam_model : BeamModel
        Fixed beam model used throughout (can be updated later).
    sky_nside : int
        HEALPix resolution for the sky model.
    sky_eta_max : float
        DPSS delay half-width for the sky in seconds.
    freqs : array_like, shape (nfreq,)
        Frequency array in Hz.
    rot_matrices : array_like, shape (ntime, 3, 3)
        Pre-computed equatorial → topocentric rotation matrices.
    data : array_like, shape (ntime, nfreq, nbls), complex
        Observed (pre-calibrated) visibilities.
    sky_eigenval_cutoff : float
    eps : float
        NUFFT accuracy.
    """

    def __init__(
        self,
        array: HERAArray,
        beam_model: BeamModel,
        sky_nside: int,
        sky_eta_max: float,
        freqs,
        rot_matrices,
        data,
        sky_eigenval_cutoff: float = 1e-9,
        eps: float = 1e-6,
    ):
        self.freqs = jnp.array(freqs, dtype=DTYPE_R)
        self.rot_matrices = jnp.array(rot_matrices, dtype=DTYPE_R)
        self.data = jnp.array(data, dtype=DTYPE_C)
        self.bls = jnp.array(array.bls, dtype=DTYPE_R)

        # Sky DPSS matrix
        self.A_sky = dpss_matrix(
            freqs, sky_eta_max, eigenval_cutoff=sky_eigenval_cutoff
        )  # (nfreq, nmodes_sky)

        # Forward model
        self.fwd = ForwardModel(array, sky_nside, beam_model, freqs, eps=eps)
        self.fwd.set_sky_dpss(self.A_sky)

        # JIT-compiled loss
        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))

    # ------------------------------------------------------------------
    # Simulation + loss
    # ------------------------------------------------------------------

    def _loss(self, params):
        """Mean squared residual |data - gain * model|^2."""
        vis_model = self.fwd.simulate(params["sky_coeffs"], self.rot_matrices)
        vis_cal = apply_gains(
            vis_model,
            params["log_amp"],
            params["phase"],
            params["phi"],
            self.bls,
        )
        return jnp.mean(jnp.abs(self.data - vis_cal) ** 2)

    def calc_loss(self, params):
        return float(self._jit_loss(params))

    def simulate(self, params):
        """Return gain-applied model visibilities for the given params."""
        vis_model = self.fwd.simulate(params["sky_coeffs"], self.rot_matrices)
        return apply_gains(
            vis_model,
            params["log_amp"],
            params["phase"],
            params["phi"],
            self.bls,
        )

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def init_params(self, seed: int = 0):
        """
        Return a zero-initialised parameter dict.

        Sky coefficients are zero (no sky flux), gain parameters are
        zero (unity gain, zero phase).  The caller should set sky_coeffs
        from a prior simulation or sky model before fitting.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        npix_sky = self.fwd.npix_sky
        nmodes_sky = self.A_sky.shape[1]
        params = {
            "sky_coeffs": jnp.zeros((npix_sky, nmodes_sky), dtype=DTYPE_R),
            **init_gain_params(self.nfreq),
        }
        return params

    def init_sky_from_flux(self, flux):
        """
        Project a frequency-resolved flux map onto DPSS sky coefficients.

        Parameters
        ----------
        flux : array_like, shape (npix_sky, nfreq)
            Sky flux in Jy (or arbitrary units consistent with data).

        Returns
        -------
        sky_coeffs : jnp.array, shape (npix_sky, nmodes_sky)
        """
        from .dpss import dpss_project
        import numpy as np
        return jnp.array(
            dpss_project(np.asarray(flux), self.A_sky), dtype=DTYPE_R
        )

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------

    def fit_gains_only(self, sky_coeffs, gain_params, maxiter: int = 60, tol: float = 1e-9):
        """
        Optimise gain parameters with sky held fixed.

        Pre-computes model visibilities from *sky_coeffs*, then minimises
        only over {log_amp, phase, phi}.  This is faster and more stable
        than joint optimisation when the sky model is already good.

        Parameters
        ----------
        sky_coeffs : jnp.array, shape (npix_sky, nmodes_sky)
            Fixed sky model (not updated during optimisation).
        gain_params : dict
            Initial gain parameters with keys 'log_amp', 'phase', 'phi'.
        maxiter : int
        tol : float

        Returns
        -------
        gain_params : dict
        loss : float
        """
        # Pre-compute model visibilities once; only differentiate through gains.
        vis_model = jax.jit(self.fwd.simulate)(sky_coeffs, self.rot_matrices)

        def gain_loss(gp):
            vis_cal = apply_gains(vis_model, gp["log_amp"], gp["phase"], gp["phi"], self.bls)
            return jnp.mean(jnp.abs(self.data - vis_cal) ** 2)

        solver = LBFGS(
            fun=gain_loss,
            tol=tol,
            maxiter=maxiter,
            verbose=False,
            linesearch="backtracking",
            increase_factor=5,
            history_size=20,
            jit=True,
        )
        result = solver.run(gain_params)
        return result.params, float(result.state.value)

    def fit_lbfgs(self, params, maxiter: int = 30, tol: float = 1e-7):
        """
        L-BFGS optimisation via jaxopt.

        Returns
        -------
        params : dict
        loss : float
        """
        solver = LBFGS(
            fun=self._loss,
            tol=tol,
            maxiter=maxiter,
            verbose=False,
            linesearch="backtracking",
            increase_factor=5,
            history_size=10,
            jit=True,
        )
        result = solver.run(params)
        return result.params, float(result.state.value)

    def fit_optax(
        self,
        params,
        maxiter: int = 60,
        lr: float = 3e-2,
        opt_type=None,
        verbose: bool = False,
        **opt_kwargs,
    ):
        """
        Optax-based gradient descent.

        Parameters
        ----------
        opt_type : optax optimizer factory, optional
            Defaults to optax.adam.
        **opt_kwargs
            Passed to opt_type.

        Returns
        -------
        params : dict
        loss : float
        """
        if opt_type is None:
            opt_type = optax.adam
        optimizer = opt_type(learning_rate=lr, **opt_kwargs)
        opt_state = optimizer.init(params)

        best_params = params
        best_loss = jnp.inf

        for i in range(maxiter):
            loss, grads = self._jit_val_grad(params)
            if loss < best_loss:
                best_loss = loss
                best_params = params
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if verbose:
                print(f"  iter {i:04d}: loss={float(loss):.4e}")

        return best_params, float(best_loss)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nfreq(self) -> int:
        return int(self.freqs.shape[0])

    @property
    def ntime(self) -> int:
        return int(self.rot_matrices.shape[0])

    @property
    def nbls(self) -> int:
        return int(self.data.shape[2])
