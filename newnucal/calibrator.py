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

import signal as _signal

import jax
import jax.numpy as jnp
import optax
from jaxopt import LBFGS

from .array import HERAArray
from .beam import BeamModel
from .dpss import dpss_matrix, dpss_project
from .simulate import ForwardModel
from .gains import apply_gains, init_gain_params

DTYPE_R = jnp.float32
DTYPE_C = jnp.complex64


class _StopFitFlag:
    """
    SIGINT-driven graceful stop flag for fit loops.

    Install as a SIGINT handler: ``signal.signal(SIGINT, flag)``.
    After receiving SIGINT the flag is set and the current JAX computation
    is allowed to finish.  Fit loops check ``flag.stop`` after each
    complete iteration.

    Calling code in a non-main thread (e.g. pytest workers) cannot install
    signal handlers; the flag then acts as a plain False sentinel.
    """

    def __init__(self):
        self.stop = False

    def __call__(self, signum, frame):
        self.stop = True
        print(
            "\n  [stop requested — finishing current iteration and returning]",
            flush=True,
        )

    @staticmethod
    def install(flag):
        """
        Try to install *flag* as the SIGINT handler.

        Returns the previous handler, or None if installation failed
        (non-main thread, unsupported platform, etc.).
        """
        try:
            return _signal.signal(_signal.SIGINT, flag)
        except (ValueError, OSError):
            return None

    @staticmethod
    def restore(old_handler):
        """Restore a previous SIGINT handler returned by ``install``."""
        if old_handler is not None:
            try:
                _signal.signal(_signal.SIGINT, old_handler)
            except (ValueError, OSError):
                pass


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

        # Store constructor args needed for sub-Calibrators in fit_progressive
        self._array = array
        self._beam_model = beam_model
        self._sky_eta_max = sky_eta_max
        self._sky_eigenval_cutoff = sky_eigenval_cutoff
        self._eps = eps

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

    def fit_gains_linear(self, sky_coeffs):
        """
        Analytically solve for gain parameters with sky held fixed.

        The MSE loss is nonlinear in (log_amp, phase, phi) because gains enter
        as exp(...), so iterative optimisers like L-BFGS only approximate the
        solution.  This method is exact in one pass via two linear steps:

          1. Per-baseline optimal complex gain by matched-filter projection:
               g[f,b] = Σ_t data·conj(model) / Σ_t |model|²

          2. Log-domain weighted linear regression per frequency:
               log g[f,b] = log_amp[f]  +  i·(phase[f]
                             + phi_x[f]·bl_x[b] + phi_y[f]·bl_y[b])

        Parameters
        ----------
        sky_coeffs : jnp.array, shape (npix_sky, nmodes_sky)
            Fixed sky model.

        Returns
        -------
        gain_params : dict  with keys log_amp, phase, phi
        loss : float
        """
        import numpy as np

        vis_model = jax.jit(self.fwd.simulate)(sky_coeffs, self.rot_matrices)

        # Step 1: optimal unconstrained per-(freq, baseline) complex gain
        num = jnp.sum(self.data * jnp.conj(vis_model), axis=0)  # (nfreq, nbls)
        den = jnp.sum(jnp.abs(vis_model) ** 2, axis=0)          # (nfreq, nbls)
        g_opt = np.array(num / (den + 1e-30))   # (nfreq, nbls), complex
        w     = np.array(den)                   # model power, used as weights

        # Step 2: project onto 4-DOF gain model in the log domain
        #   Re(log g[f,b]) = log_amp[f]          (baseline-independent)
        #   Im(log g[f,b]) = phase[f] + phi_x[f]*bl_x[b] + phi_y[f]*bl_y[b]
        log_g = np.log(g_opt)           # (nfreq, nbls); principal log of complex
        bls   = np.array(self.bls)      # (nbls, 3)

        # log_amp: weighted mean of Re(log g) across baselines
        w_sum   = w.sum(axis=1) + 1e-30
        log_amp = (w * np.real(log_g)).sum(axis=1) / w_sum      # (nfreq,)

        # (phase, phi_x, phi_y): weighted lstsq of Im(log g) vs [1, bl_x, bl_y]
        X     = np.column_stack([np.ones(self.nbls), bls[:, 0], bls[:, 1]])
        theta = np.empty((3, self.nfreq), dtype=np.float64)
        for f in range(self.nfreq):
            sw         = np.sqrt(w[f] + 1e-30)
            theta[:, f] = np.linalg.lstsq(
                X * sw[:, None], np.imag(log_g[f]) * sw, rcond=None
            )[0]

        gain_params = {
            "log_amp": jnp.array(log_amp, dtype=jnp.float32),
            "phase":   jnp.array(theta[0], dtype=jnp.float32),
            "phi":     jnp.array(theta[1:], dtype=jnp.float32),
        }
        loss = float(self._jit_loss({"sky_coeffs": sky_coeffs, **gain_params}))
        return gain_params, loss

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

    def fit_sky_only(self, sky_coeffs, gain_params, maxiter: int = 30, tol: float = 1e-7):
        """
        Optimise sky coefficients with gains held fixed.

        Pre-applies the inverse gain to the data, then differentiates only
        through the sky NUFFT.  This halves the parameter space vs joint
        L-BFGS and avoids computing gain gradients (which are trivial anyway).

        Parameters
        ----------
        sky_coeffs : jnp.array, shape (npix_sky, nmodes_sky)
        gain_params : dict  with keys log_amp, phase, phi
        maxiter : int
        tol : float

        Returns
        -------
        sky_coeffs : jnp.array
        loss : float
        """
        # Pre-apply inverse gain to data so the sky loss is gain-free.
        inv_gain_params = {
            "log_amp": -gain_params["log_amp"],
            "phase":   -gain_params["phase"],
            "phi":     -gain_params["phi"],
        }
        data_cal = apply_gains(self.data, inv_gain_params["log_amp"],
                               inv_gain_params["phase"], inv_gain_params["phi"],
                               self.bls)

        def sky_loss(sc):
            vis_model = self.fwd.simulate(sc, self.rot_matrices)
            return jnp.mean(jnp.abs(data_cal - vis_model) ** 2)

        solver = LBFGS(
            fun=sky_loss,
            tol=tol,
            maxiter=maxiter,
            verbose=False,
            linesearch="backtracking",
            increase_factor=5,
            history_size=10,
            jit=True,
        )
        result = solver.run(sky_coeffs)
        sky_new = result.params
        loss = float(self._jit_loss({"sky_coeffs": sky_new, **gain_params}))
        return sky_new, loss

    def fit_alternating(
        self,
        sky_coeffs,
        gain_params,
        n_outer: int = 10,
        sky_maxiter: int = 10,
        sky_tol: float = 1e-7,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """
        Alternating minimisation: exact gain solve + sky gradient steps.

        Each outer iteration:
          1. Sky step  — L-BFGS on sky_coeffs only (gradient, NUFFT fwd+bwd)
          2. Gain step — analytic linear solve      (no gradient, NUFFT fwd only)

        This is the major/minor cycle structure from radio interferometry.
        Gain steps cost ~half a sky step (no NUFFT backward) and give the
        exact optimal gains for the current sky rather than a small nudge.

        Pressing Ctrl-C (SIGINT) sets a stop flag that is checked after each
        complete outer iteration, so the last full sky+gain state is always
        returned cleanly.

        Parameters
        ----------
        sky_coeffs : jnp.array
        gain_params : dict
        n_outer : int
            Number of alternating cycles.
        sky_maxiter : int
            L-BFGS iterations per sky step.
        verbose : bool
        _stop_flag : _StopFitFlag or None
            Internal: shared stop flag injected by ``fit_progressive``.
            When None a private flag is installed as the SIGINT handler.

        Returns
        -------
        sky_coeffs : jnp.array
        gain_params : dict
        loss : float
        """
        # Install SIGINT handler if we own it (standalone call)
        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None  # caller owns the handler

        try:
            loss = float(self._jit_loss({"sky_coeffs": sky_coeffs, **gain_params}))
            for i in range(n_outer):
                sky_coeffs, _ = self.fit_sky_only(sky_coeffs, gain_params, maxiter=sky_maxiter, tol=sky_tol)
                gain_params, loss = self.fit_gains_linear(sky_coeffs)
                if verbose:
                    print(f"  outer {i:03d}: loss={loss:.4e}")
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return sky_coeffs, gain_params, loss

    def fit_progressive(
        self,
        sky_coeffs,
        gain_params,
        nside_schedule=None,
        outer_per_level=None,
        sky_maxiter_per_level=None,
        eps_per_level=None,
        verbose: bool = False,
    ):
        """
        Multi-resolution coarse-to-fine alternating calibration.

        Runs ``fit_alternating`` at successively finer sky resolutions.
        Coarse levels are cheap (NUFFT cost ∝ npix_sky) and quickly drive
        the gain and large-scale sky parameters near the correct solution;
        finer levels then refine small-scale structure.

        Sky coefficients are transferred between levels by:
          1. Reconstruct full-frequency sky map from DPSS coefficients.
          2. Resample to new nside with ``healpy.ud_grade``.
          3. Re-project onto the new DPSS basis with ``dpss_project``.

        Gain parameters are dimensionless scalars per frequency and transfer
        directly without modification.

        **Interrupt handling**: pressing Ctrl-C (SIGINT) at any point sets a
        shared stop flag.  The running L-BFGS iteration completes, then the
        current sky is upscaled to the final nside (if stopped at a coarse
        level) and the method returns the best state so far.  A single Ctrl-C
        is enough regardless of which level is active.

        Parameters
        ----------
        sky_coeffs : jnp.array, shape (npix_sky_final, nmodes_sky)
            Initial sky at the *final* (finest) nside.  Downgraded internally
            for coarser levels.
        gain_params : dict  with keys log_amp, phase, phi
        nside_schedule : list[int], optional
            nside values to use, coarse to fine.  The last entry must equal
            ``self.fwd.sky_nside``.  Defaults to a geometric sequence from
            nside=8 up to the final nside (doubling each step).
        outer_per_level : list[int], optional
            Number of alternating outer iterations at each level.
            Default: 5 for coarse levels, 10 for the final level.
        sky_maxiter_per_level : list[int], optional
            L-BFGS iterations per sky step at each level.
            Default: 10 for all levels.
        eps_per_level : list[float], optional
            NUFFT accuracy at each level.  Coarser levels can tolerate lower
            accuracy (e.g., 1e-4) for extra speed.
            Default: ``self._eps`` at all levels.
        verbose : bool

        Returns
        -------
        sky_coeffs : jnp.array, shape (npix_sky_final, nmodes_sky)
        gain_params : dict
        loss : float
        """
        import healpy as hp
        import numpy as np

        final_nside = self.fwd.sky_nside
        A_sky_np = np.array(self.A_sky)

        # ---- build default schedule ----------------------------------------
        if nside_schedule is None:
            nside = 8
            nside_schedule = []
            while nside < final_nside:
                nside_schedule.append(nside)
                nside *= 2
            nside_schedule.append(final_nside)

        nlevels = len(nside_schedule)

        if outer_per_level is None:
            outer_per_level = [5] * (nlevels - 1) + [10]
        if sky_maxiter_per_level is None:
            sky_maxiter_per_level = [10] * nlevels
        if eps_per_level is None:
            eps_per_level = [self._eps] * nlevels

        assert nside_schedule[-1] == final_nside, (
            f"Last nside in schedule ({nside_schedule[-1]}) must equal "
            f"the calibrator's sky_nside ({final_nside})"
        )

        # ---- shared stop flag + SIGINT handler -----------------------------
        stop = _StopFitFlag()
        old_handler = _StopFitFlag.install(stop)

        # ---- reconstruct flux map at final nside from initial sky_coeffs ---
        # sky_coeffs: (npix_final, nmodes); A_sky: (nfreq, nmodes)
        # flux_map_final: (npix_final, nfreq)
        flux_map_final = np.array(sky_coeffs) @ A_sky_np.T  # (npix_final, nfreq)
        sky_coeffs_lvl = sky_coeffs   # initialise so it's always defined
        loss = float(self._jit_loss({"sky_coeffs": sky_coeffs, **gain_params}))

        try:
            # ---- iterate over levels ---------------------------------------
            for lvl, nside_lvl in enumerate(nside_schedule):
                npix_lvl    = hp.nside2npix(nside_lvl)
                n_outer     = outer_per_level[lvl]
                sky_maxiter = sky_maxiter_per_level[lvl]
                eps_lvl     = eps_per_level[lvl]
                is_final    = (nside_lvl == final_nside)

                if verbose:
                    print(f"[progressive] nside={nside_lvl:4d}  npix={npix_lvl:6d}  "
                          f"n_outer={n_outer}  sky_maxiter={sky_maxiter}  eps={eps_lvl:.0e}")

                # Downgrade flux map to current nside
                if is_final:
                    flux_map_lvl = flux_map_final       # (npix_final, nfreq)
                else:
                    flux_map_lvl = np.stack(
                        [hp.ud_grade(flux_map_final[:, f], nside_lvl)
                         for f in range(self.nfreq)],
                        axis=1,
                    )  # (npix_lvl, nfreq)

                # Project onto DPSS basis at this level (A_sky is frequency-only)
                sky_coeffs_lvl = jnp.array(
                    dpss_project(flux_map_lvl, A_sky_np), dtype=DTYPE_R
                )  # (npix_lvl, nmodes_sky)

                # Build sub-Calibrator at this nside/eps
                if is_final:
                    sub_cal = self
                else:
                    sub_cal = Calibrator(
                        self._array,
                        self._beam_model,
                        nside_lvl,
                        self._sky_eta_max,
                        np.array(self.freqs),
                        np.array(self.rot_matrices),
                        np.array(self.data),
                        sky_eigenval_cutoff=self._sky_eigenval_cutoff,
                        eps=eps_lvl,
                    )

                # Run alternating minimisation — shares the stop flag so one
                # Ctrl-C stops both the inner loop and this outer loop.
                sky_coeffs_lvl, gain_params, loss = sub_cal.fit_alternating(
                    sky_coeffs_lvl,
                    gain_params,
                    n_outer=n_outer,
                    sky_maxiter=sky_maxiter,
                    verbose=verbose,
                    _stop_flag=stop,
                )

                if verbose:
                    print(f"  → loss after nside={nside_lvl}: {loss:.4e}")

                # Upscale flux map for next level (also used on early exit)
                if not is_final:
                    flux_map_final = np.array(sky_coeffs_lvl) @ A_sky_np.T
                    flux_map_final = np.stack(
                        [hp.ud_grade(flux_map_final[:, f], final_nside)
                         for f in range(self.nfreq)],
                        axis=1,
                    )  # (npix_final, nfreq)

                if stop.stop:
                    break   # exit level loop; upscaling handled below

        finally:
            _StopFitFlag.restore(old_handler)

        # ---- if stopped at a coarse level, upscale sky to final nside ------
        if stop.stop and sky_coeffs_lvl.shape[0] != hp.nside2npix(final_nside):
            if verbose:
                print(f"[progressive] Upscaling sky from nside={nside_lvl} "
                      f"→ nside={final_nside} before returning.")
            sky_coeffs_lvl = jnp.array(
                dpss_project(flux_map_final, A_sky_np), dtype=DTYPE_R
            )
            loss = float(self._jit_loss({"sky_coeffs": sky_coeffs_lvl, **gain_params}))

        return sky_coeffs_lvl, gain_params, loss

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
