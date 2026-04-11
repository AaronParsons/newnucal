"""
Calibrator — ties together ForwardModel, gain parameters, and optimisation.
"""

import signal as _signal

import numpy as np
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


class Calibrator:
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
        eta_padding: float = 0.0,
    ):
        self.freqs = jnp.array(freqs, dtype=DTYPE_R)
        self.rot_matrices = jnp.array(rot_matrices, dtype=DTYPE_R)
        self.data = jnp.array(data, dtype=DTYPE_C)
        self.bls = jnp.array(array.bls, dtype=DTYPE_R)

        self.A_sky = dpss_matrix(freqs, sky_eta_max, eigenval_cutoff=sky_eigenval_cutoff)
        self.fwd = ForwardModel(
            array,
            sky_nside,
            beam_model,
            freqs,
            eps=eps,
            #eta_max=sky_eta_max + getattr(beam_model, 'eta_max', 0.0),
            eta_max=None,
            eta_padding=eta_padding,
        )
        self.fwd.set_sky_dpss(self.A_sky)
        self.fwd.precompute_time_geometry(self.rot_matrices)

        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))

    def _loss(self, params):
        vis_model = self.fwd.simulate(params['sky_coeffs'], self.rot_matrices)
        vis_cal = apply_gains(vis_model, params['log_amp'], params['phase'], params['phi'], self.bls)
        return jnp.mean(jnp.abs(self.data - vis_cal) ** 2)

    def calc_loss(self, params):
        return float(self._jit_loss(params))

    def simulate(self, params):
        vis_model = self.fwd.simulate(params['sky_coeffs'], self.rot_matrices)
        return apply_gains(vis_model, params['log_amp'], params['phase'], params['phi'], self.bls)

    def init_params(self):
        npix_sky = self.fwd.npix_sky
        nmodes_sky = self.A_sky.shape[1]
        return {
            'sky_coeffs':  jnp.zeros((npix_sky, nmodes_sky), dtype=DTYPE_R),
            'beam_coeffs': jnp.array(self.fwd.beam_coeffs),
            **init_gain_params(self.ntime, self.nfreq),
        }

    def init_sky_from_flux(self, flux):
        return jnp.array(dpss_project(np.asarray(flux), self.A_sky), dtype=DTYPE_R)

    @staticmethod
    def _invert_gains(gain_params):
        """Return gain parameters with negated log_amp, phase, and phi."""
        return {k: -gain_params[k] for k in ('log_amp', 'phase', 'phi')}

    def calibrated_residual(self, params):
        inv = self._invert_gains(params)
        data_cal = apply_gains(self.data, inv['log_amp'], inv['phase'], inv['phi'], self.bls)
        vis_model = self.fwd.simulate(params['sky_coeffs'], self.rot_matrices)
        return data_cal - vis_model

    # ------------------------------------------------------------------
    # Pixel-cut helpers
    # ------------------------------------------------------------------

    def apply_horizon_cut(self):
        """Remove sky pixels that are never above the horizon.

        Permanently reduces ``npix_sky``; call *before* initialising sky
        coefficients.  Returns the number of pixels retained.
        """
        mask = self.fwd.build_ever_visible_mask(self.rot_matrices)
        self.fwd.apply_pixel_mask(mask)
        self.fwd.precompute_time_geometry(self.rot_matrices)
        n_keep = int(mask.sum())
        n_drop = int((~mask).sum())
        print(f'  Horizon cut: {n_keep} pixels retained, {n_drop} removed.')
        return n_keep

    def apply_zenith_cut(self, scale=1.0):
        """Enable a per-time dynamic zenith cut at *scale* × beam FWHM.

        Pixels within ``scale × 1.22 λ_max / D`` of zenith are included
        at each time step; the rest are skipped in the NUFFT.  ``scale=1``
        keeps the full primary-beam footprint at the lowest frequency;
        smaller values restrict to the inner beam.

        Calling this re-runs ``precompute_time_geometry``; any previously
        applied horizon cut is preserved.
        """
        self.fwd.precompute_time_geometry(self.rot_matrices, zenith_cut_scale=scale)
        counts = [len(px) for px in self.fwd._zenith_px_idx_list]
        print(f'  Zenith cut (scale={scale}): '
              f'{min(counts)}–{max(counts)} pixels per time step '
              f'(mean {sum(counts)/len(counts):.0f}).')

    def disable_zenith_cut(self):
        """Remove the dynamic zenith cut and revert to the full pixel set."""
        self.fwd.precompute_time_geometry(self.rot_matrices, zenith_cut_scale=None)

    def fit_gains_linear(self, sky_coeffs):
        vis_model = jax.jit(self.fwd.simulate)(sky_coeffs, self.rot_matrices)
        # Per-time optimal complex gain: data · conj(model) / |model|²
        # Shape: (ntime, nfreq, nbls)
        den = np.array(jnp.abs(vis_model) ** 2)          # (ntime, nfreq, nbls), weights
        g_opt = np.array(self.data * jnp.conj(vis_model)) / (den + 1e-30)
        log_g = np.log(g_opt + 0j)                        # complex log, (ntime, nfreq, nbls)

        # log_amp: baseline-weighted mean of Re(log g) per (time, freq)
        w_sum = den.sum(axis=2) + 1e-30                   # (ntime, nfreq)
        log_amp = (den * np.real(log_g)).sum(axis=2) / w_sum  # (ntime, nfreq)

        # phase and phi: weighted lstsq over baselines, independently per (time, freq)
        # Design matrix X: (nbls, 3) — [1, bl_E, bl_N]
        bls = np.array(self.bls)
        X = np.column_stack([np.ones(self.nbls), bls[:, 0], bls[:, 1]])  # (nbls, 3)

        # Xy[t, f, k] = sum_b X[b, k] * den[t, f, b] * Im(log_g[t, f, b])
        Xy = np.einsum('bk,tfb->tfk', X, den * np.imag(log_g))   # (ntime, nfreq, 3)
        # XTX[t, f, k, l] = sum_b X[b, k] * den[t, f, b] * X[b, l]
        XTX = np.einsum('bk,tfb,bl->tfkl', X, den, X)             # (ntime, nfreq, 3, 3)
        XTX += 1e-10 * np.eye(3)                                   # regularise

        # Solve all (time, freq) systems at once via numpy broadcasting.
        # Add trailing dim so solve sees (..., 3, 1) not (..., 3) — required
        # by numpy ≥2.0 which no longer treats b.ndim == a.ndim-1 as a vector.
        theta = np.linalg.solve(XTX, Xy[..., None])[..., 0]       # (ntime, nfreq, 3)

        gain_params = {
            'log_amp': jnp.array(log_amp, dtype=jnp.float32),             # (ntime, nfreq)
            'phase':   jnp.array(theta[:, :, 0], dtype=jnp.float32),      # (ntime, nfreq)
            'phi':     jnp.array(                                           # (ntime, 2, nfreq)
                theta[:, :, 1:].transpose(0, 2, 1), dtype=jnp.float32
            ),
        }
        loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
        return gain_params, loss

    def _recompile_jit(self):
        """Recompile loss JIT functions after precomputed arrays change (e.g. beam update)."""
        self._jit_loss = jax.jit(self._loss)
        self._jit_val_grad = jax.jit(jax.value_and_grad(self._loss))

    def fit_beam_only(self, params, maxiter: int = 10, tol: float = 1e-7):
        """L-BFGS on beam_coeffs with sky and gains held fixed.

        Uses :meth:`~newnucal.simulate.ForwardModel.simulate_variable_beam` so
        the loss is differentiable w.r.t. *beam_coeffs* via JAX AD.  After the
        solve the ForwardModel's precomputed beam cache is updated and the
        calibrator's JIT functions are recompiled so subsequent sky/gain solves
        use the new beam.

        Parameters
        ----------
        params : dict
            Must include ``sky_coeffs``, ``beam_coeffs``, ``log_amp``, ``phase``,
            ``phi``.
        maxiter : int
        tol : float

        Returns
        -------
        params : dict — updated with new ``beam_coeffs``
        loss : float
        """
        sky_coeffs = params['sky_coeffs']
        gain_params = {k: params[k] for k in ('log_amp', 'phase', 'phi')}
        bls = self.bls

        def beam_loss(bc):
            vis_model = self.fwd.simulate_variable_beam(sky_coeffs, bc, self.rot_matrices)
            vis_cal = apply_gains(
                vis_model, gain_params['log_amp'], gain_params['phase'],
                gain_params['phi'], bls,
            )
            return jnp.mean(jnp.abs(self.data - vis_cal) ** 2)

        solver = LBFGS(
            fun=beam_loss, tol=tol, maxiter=maxiter, verbose=False,
            linesearch='backtracking', increase_factor=5, history_size=10, jit=True,
        )
        result = solver.run(params['beam_coeffs'])
        new_beam = result.params

        # Update the precomputed cache so sky/gain solves see the new beam.
        self.fwd.update_beam_cache(new_beam)
        self._recompile_jit()

        new_params = {**params, 'beam_coeffs': new_beam}
        loss = float(beam_loss(new_beam))
        return new_params, loss

    def fit_sky_only(self, sky_coeffs, gain_params, maxiter: int = 30, tol: float = 1e-7):
        inv_gain_params = {
            'log_amp': -gain_params['log_amp'],
            'phase': -gain_params['phase'],
            'phi': -gain_params['phi'],
        }
        data_cal = apply_gains(self.data, inv_gain_params['log_amp'], inv_gain_params['phase'], inv_gain_params['phi'], self.bls)

        def sky_loss(sc):
            vis_model = self.fwd.simulate(sc, self.rot_matrices)
            return jnp.mean(jnp.abs(data_cal - vis_model) ** 2)

        solver = LBFGS(fun=sky_loss, tol=tol, maxiter=maxiter, verbose=False, linesearch='backtracking', increase_factor=5, history_size=10, jit=True)
        result = solver.run(sky_coeffs)
        sky_new = result.params
        loss = float(self._jit_loss({'sky_coeffs': sky_new, **gain_params}))
        return sky_new, loss

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
        velocity = jnp.zeros_like(sky_coeffs)
        best_sky = sky_coeffs
        best_loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
        for i in range(n_iter):
            resid = self.calibrated_residual({'sky_coeffs': sky_coeffs, **gain_params})
            delta = self.fwd.accumulate_equatorial_sky_update(resid, step_size=step_size, beam_reg=beam_reg)
            velocity = momentum * velocity + delta
            sky_coeffs = sky_coeffs + velocity
            loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
            if loss < best_loss:
                best_loss = loss
                best_sky = sky_coeffs
            else:
                print('    WARNING: loss increased')
            if verbose:
                print(f'    dirty iter {i:03d}: loss={loss:.4e}')
        return best_sky, best_loss

    @staticmethod
    def _flatten_sky_coeffs(sky_coeffs):
        return np.asarray(sky_coeffs, dtype=np.float64).ravel()

    @staticmethod
    def _unflatten_sky_coeffs(vec, shape):
        return jnp.array(np.asarray(vec, dtype=np.float32).reshape(shape), dtype=DTYPE_R)

    @staticmethod
    def _anderson_coeffs(residual_rows, ridge: float = 1e-8):
        """Solve min ||sum_i beta_i f_i|| subject to sum_i beta_i = 1."""
        n = residual_rows.shape[0]
        gram = residual_rows @ residual_rows.T
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

    def fit_alternating_dirty(
        self,
        params,
        n_iter: int = 30,
        step_size: float = 0.5,
        beam_reg: float = 1e-3,
        momentum: float = 0.0,
        anderson_history: int = 0,
        aa_start: int = 2,
        aa_damping: float = 0.5,
        aa_ridge: float = 1e-8,
        aa_max_weight: float = 10.0,
        polish_maxiter: int = 5,
        zenith_cut_scale: float | None = None,
        solve_every: dict | None = None,
        beam_maxiter: int = 10,
        beam_tol: float = 1e-7,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Alternating dirty-map minimisation, optionally with Anderson acceleration.

        Parameters
        ----------
        params : dict with keys ``sky_coeffs``, ``beam_coeffs``, ``log_amp``,
            ``phase``, ``phi``
        n_iter : int
            Total number of iterations.  The sky receives one dirty-map update
            per iteration.
        solve_every : dict or None
            Controls how often non-sky components are updated.  Keys:

            ``'gains'`` (int, default 1)
                Solve gains every *N* iterations.  ``0`` disables gain
                solving entirely.
            ``'beam'`` (int, default 0)
                Solve beam every *N* iterations using
                :meth:`fit_beam_only`.  ``0`` (default) disables beam solving.
            ``'polish'`` (int, default 0)
                Run an L-BFGS sky polish (``polish_maxiter`` steps via
                :meth:`fit_sky_only`) every *N* iterations, followed by
                a gain re-solve.  ``0`` (default) disables polishing.

            Example: ``solve_every={'gains': 1, 'beam': 5, 'polish': 10}``.
        anderson_history : int
            Number of iterates to keep for Anderson acceleration on the
            sky fixed-point map.  ``0`` disables Anderson.
        zenith_cut_scale : float or None
            If given, enable a per-time dynamic zenith cut during this fit.
            Automatically removed on return, even if interrupted.

        Returns
        -------
        params : dict
            Updated parameters with keys ``sky_coeffs``, ``beam_coeffs``,
            ``log_amp``, ``phase``, ``phi``.
        loss : float
        """
        if solve_every is None:
            solve_every = {}
        gains_every  = solve_every.get('gains',  1)
        beam_every   = solve_every.get('beam',   0)
        polish_every = solve_every.get('polish', 0)

        sky_coeffs  = params['sky_coeffs']
        gain_params = {k: params[k] for k in ('log_amp', 'phase', 'phi')}
        beam_coeffs = params.get('beam_coeffs', self.fwd.beam_coeffs)

        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None
        if zenith_cut_scale is not None:
            self.fwd.precompute_time_geometry(self.rot_matrices, zenith_cut_scale=zenith_cut_scale)

        history_g = []
        history_f = []

        def _full_params():
            return {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}

        try:
            loss = float(self._jit_loss(_full_params()))
            for i in range(n_iter):
                # --- Gain solve (if scheduled) ---
                if gains_every > 0 and i % gains_every == 0:
                    gain_params, _gl = self.fit_gains_linear(sky_coeffs)
                    if verbose:
                        print(f'    [gains]: loss={_gl:.4e}')

                # --- Sky dirty update (every step) ---
                sky_plain, loss_plain = self.fit_sky_dirty(
                    sky_coeffs, gain_params,
                    n_iter=1, step_size=step_size,
                    beam_reg=beam_reg, momentum=momentum, verbose=verbose,
                )
                gain_plain = gain_params

                sky_next = sky_plain
                gain_next = gain_plain
                loss_next = loss_plain
                used_anderson = False

                if anderson_history > 0:
                    x_vec = self._flatten_sky_coeffs(sky_coeffs)
                    g_vec = self._flatten_sky_coeffs(sky_plain)
                    f_vec = g_vec - x_vec
                    history_g.append(g_vec.copy())
                    history_f.append(f_vec.copy())
                    if len(history_g) > anderson_history:
                        history_g.pop(0)
                        history_f.pop(0)

                    if i >= aa_start and len(history_f) >= 2:
                        F = np.stack(history_f, axis=0)
                        beta = self._anderson_coeffs(F, ridge=aa_ridge)
                        if np.all(np.isfinite(beta)) and float(np.max(np.abs(beta))) <= aa_max_weight:
                            g_mix = np.tensordot(beta, np.stack(history_g, axis=0), axes=1)
                            sky_aa = self._unflatten_sky_coeffs(g_mix, sky_coeffs.shape)
                            sky_cand = ((1.0 - aa_damping) * sky_plain + aa_damping * sky_aa).astype(DTYPE_R)
                            gain_cand, _ = self.fit_gains_linear(sky_cand)
                            loss_cand = float(self._jit_loss(
                                {'sky_coeffs': sky_cand, 'beam_coeffs': beam_coeffs, **gain_cand}
                            ))
                            if loss_cand < loss_plain:
                                sky_next = sky_cand
                                gain_next = gain_cand
                                loss_next = loss_cand
                                used_anderson = True

                sky_coeffs = sky_next
                gain_params = gain_next
                loss = loss_next

                # --- Polish sky with L-BFGS (if scheduled) ---
                if polish_every > 0 and (i + 1) % polish_every == 0:
                    if verbose:
                        print(f'    [polish]: starting L-BFGS sky polish ...')
                    sky_coeffs, loss = self.fit_sky_only(
                        sky_coeffs, gain_params, maxiter=polish_maxiter, tol=1e-7
                    )
                    gain_params, _ = self.fit_gains_linear(sky_coeffs)
                    if verbose:
                        print(f'    [polish]: loss={loss:.4e}')

                # --- Beam solve (if scheduled) ---
                if beam_every > 0 and (i + 1) % beam_every == 0:
                    if verbose:
                        print(f'    [beam]: starting L-BFGS beam solve ...')
                    bp, loss = self.fit_beam_only(
                        {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                        maxiter=beam_maxiter, tol=beam_tol,
                    )
                    beam_coeffs = bp['beam_coeffs']
                    if verbose:
                        print(f'    [beam]: loss={loss:.4e}')

                if verbose:
                    tag = 'AA' if used_anderson else 'plain'
                    print(f'  iter {i:04d} [{tag}]: loss={loss:.4e}')
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
            if zenith_cut_scale is not None:
                self.fwd.precompute_time_geometry(self.rot_matrices, zenith_cut_scale=None)
        return {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}, loss

    def fit_alternating(
        self,
        params,
        n_iter: int = 10,
        sky_maxiter: int = 10,
        sky_tol: float = 1e-7,
        zenith_cut_scale: float | None = None,
        solve_every: dict | None = None,
        beam_maxiter: int = 10,
        beam_tol: float = 1e-7,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Alternating L-BFGS sky / linear-gain minimisation.

        Parameters
        ----------
        params : dict with keys ``sky_coeffs``, ``beam_coeffs``, ``log_amp``,
            ``phase``, ``phi``
        n_iter : int
            Number of iterations.  The sky is updated (L-BFGS) every
            iteration.
        solve_every : dict or None
            Controls how often non-sky components are updated.  Keys:

            ``'gains'`` (int, default 1)
                Solve gains every *N* iterations.
            ``'beam'`` (int, default 0)
                Solve beam every *N* iterations using
                :meth:`fit_beam_only`.  ``0`` (default) disables beam solving.
        zenith_cut_scale : float or None
            If given, enable a per-time dynamic zenith cut during this fit.
            Automatically removed on return, even if interrupted.

        Returns
        -------
        params : dict
            Updated parameters with keys ``sky_coeffs``, ``beam_coeffs``,
            ``log_amp``, ``phase``, ``phi``.
        loss : float
        """
        if solve_every is None:
            solve_every = {}
        gains_every = solve_every.get('gains', 1)
        beam_every  = solve_every.get('beam',  0)

        sky_coeffs  = params['sky_coeffs']
        gain_params = {k: params[k] for k in ('log_amp', 'phase', 'phi')}
        beam_coeffs = params.get('beam_coeffs', self.fwd.beam_coeffs)

        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None
        if zenith_cut_scale is not None:
            self.fwd.precompute_time_geometry(self.rot_matrices, zenith_cut_scale=zenith_cut_scale)
        try:
            loss = float(self._jit_loss(params))
            for i in range(n_iter):
                # --- Sky L-BFGS update (every step) ---
                if verbose:
                    print(f'    [sky]: starting L-BFGS sky solve ...')
                sky_coeffs, _sl = self.fit_sky_only(
                    sky_coeffs, gain_params, maxiter=sky_maxiter, tol=sky_tol
                )
                if verbose:
                    print(f'    [sky]: loss={_sl:.4e}')

                # --- Gain solve (if scheduled) ---
                if gains_every > 0 and (i + 1) % gains_every == 0:
                    gain_params, loss = self.fit_gains_linear(sky_coeffs)
                    if verbose:
                        print(f'    [gains]: loss={loss:.4e}')
                else:
                    loss = float(self._jit_loss(
                        {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}
                    ))

                # --- Beam solve (if scheduled) ---
                if beam_every > 0 and (i + 1) % beam_every == 0:
                    if verbose:
                        print(f'    [beam]: starting L-BFGS beam solve ...')
                    bp, loss = self.fit_beam_only(
                        {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params},
                        maxiter=beam_maxiter, tol=beam_tol,
                    )
                    beam_coeffs = bp['beam_coeffs']
                    if verbose:
                        print(f'    [beam]: loss={loss:.4e}')

                if verbose:
                    print(f'  iter {i:04d}: loss={loss:.4e}')
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
            if zenith_cut_scale is not None:
                self.fwd.precompute_time_geometry(self.rot_matrices, zenith_cut_scale=None)
        return {'sky_coeffs': sky_coeffs, 'beam_coeffs': beam_coeffs, **gain_params}, loss

    def fit_lbfgs(self, params, maxiter: int = 30, tol: float = 1e-7):
        solver = LBFGS(fun=self._loss, tol=tol, maxiter=maxiter, verbose=False, linesearch='backtracking', increase_factor=5, history_size=10, jit=True)
        result = solver.run(params)
        return result.params, float(result.state.value)

    def fit_optax(self, params, maxiter: int = 60, lr: float = 3e-2, opt_type=None, verbose: bool = False, **opt_kwargs):
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
                print(f'  iter {i:04d}: loss={float(loss):.4e}')
        return best_params, float(best_loss)

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
