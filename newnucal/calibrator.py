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
from .dpss import dpss_matrix
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

    def init_params(self, seed: int = 0):
        npix_sky = self.fwd.npix_sky
        nmodes_sky = self.A_sky.shape[1]
        return {
            'sky_coeffs': jnp.zeros((npix_sky, nmodes_sky), dtype=DTYPE_R),
            **init_gain_params(self.nfreq),
        }

    def init_sky_from_flux(self, flux):
        from .dpss import dpss_project
        import numpy as np
        return jnp.array(dpss_project(np.asarray(flux), self.A_sky), dtype=DTYPE_R)

    def calibrated_residual(self, sky_coeffs, gain_params):
        inv_gain_params = {
            'log_amp': -gain_params['log_amp'],
            'phase': -gain_params['phase'],
            'phi': -gain_params['phi'],
        }
        data_cal = apply_gains(self.data, inv_gain_params['log_amp'], inv_gain_params['phase'], inv_gain_params['phi'], self.bls)
        vis_model = self.fwd.simulate(sky_coeffs, self.rot_matrices)
        return data_cal - vis_model

    def fit_gains_linear(self, sky_coeffs):
        import numpy as np

        vis_model = jax.jit(self.fwd.simulate)(sky_coeffs, self.rot_matrices)
        num = jnp.sum(self.data * jnp.conj(vis_model), axis=0)
        den = jnp.sum(jnp.abs(vis_model) ** 2, axis=0)
        g_opt = np.array(num / (den + 1e-30))
        w = np.array(den)
        log_g = np.log(g_opt)
        bls = np.array(self.bls)
        w_sum = w.sum(axis=1) + 1e-30
        log_amp = (w * np.real(log_g)).sum(axis=1) / w_sum
        X = np.column_stack([np.ones(self.nbls), bls[:, 0], bls[:, 1]])
        theta = np.empty((3, self.nfreq), dtype=np.float64)
        for f in range(self.nfreq):
            sw = np.sqrt(w[f] + 1e-30)
            theta[:, f] = np.linalg.lstsq(X * sw[:, None], np.imag(log_g[f]) * sw, rcond=None)[0]
        gain_params = {
            'log_amp': jnp.array(log_amp, dtype=jnp.float32),
            'phase': jnp.array(theta[0], dtype=jnp.float32),
            'phi': jnp.array(theta[1:], dtype=jnp.float32),
        }
        loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
        return gain_params, loss

    def fit_gains_only(self, sky_coeffs, gain_params, maxiter: int = 60, tol: float = 1e-9):
        vis_model = jax.jit(self.fwd.simulate)(sky_coeffs, self.rot_matrices)

        def gain_loss(gp):
            vis_cal = apply_gains(vis_model, gp['log_amp'], gp['phase'], gp['phi'], self.bls)
            return jnp.mean(jnp.abs(self.data - vis_cal) ** 2)

        solver = LBFGS(fun=gain_loss, tol=tol, maxiter=maxiter, verbose=False, linesearch='backtracking', increase_factor=5, history_size=20, jit=True)
        result = solver.run(gain_params)
        return result.params, float(result.state.value)

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
        n_minor: int = 3,
        step_size: float = 0.5,
        beam_reg: float = 1e-3,
        momentum: float = 0.0,
        verbose: bool = False,
    ):
        velocity = jnp.zeros_like(sky_coeffs)
        best_sky = sky_coeffs
        best_loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
        for i in range(n_minor):
            resid = self.calibrated_residual(sky_coeffs, gain_params)
            delta = self.fwd.accumulate_equatorial_sky_update(resid, sky_coeffs, step_size=step_size, beam_reg=beam_reg)
            velocity = momentum * velocity + delta
            sky_coeffs = sky_coeffs + velocity
            loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
            if loss < best_loss:
                best_loss = loss
                best_sky = sky_coeffs
            else:
                print('    WARNING: loss increased')
            if verbose:
                print(f'    dirty minor {i:03d}: loss={loss:.4e}')
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

    def _outer_dirty_map(
        self,
        sky_coeffs,
        n_minor: int,
        step_size: float,
        beam_reg: float,
        momentum: float,
        verbose: bool = False,
    ):
        gain_params, _ = self.fit_gains_linear(sky_coeffs)
        sky_new, loss = self.fit_sky_dirty(
            sky_coeffs,
            gain_params,
            n_minor=n_minor,
            step_size=step_size,
            beam_reg=beam_reg,
            momentum=momentum,
            verbose=verbose,
        )
        return sky_new, gain_params, loss

    def fit_alternating_dirty_anderson(
        self,
        sky_coeffs,
        gain_params,
        n_outer: int = 10,
        n_minor: int = 3,
        step_size: float = 0.5,
        beam_reg: float = 1e-3,
        momentum: float = 0.0,
        aa_history: int = 4,
        aa_start: int = 2,
        aa_damping: float = 0.5,
        aa_ridge: float = 1e-8,
        aa_max_weight: float = 10.0,
        polish_every: int = 0,
        polish_maxiter: int = 5,
        verbose: bool = False,
        _stop_flag=None,
    ):
        """Alternating dirty solve with Anderson acceleration on sky_coeffs.

        Anderson is applied to the reduced outer map
            sky -> fit_gains_linear(sky) -> fit_sky_dirty(...) -> new sky
        with acceptance based on the actual loss. If the accelerated candidate
        is not better than the plain outer step, the plain step is kept.
        """
        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None

        history_g = []
        history_f = []

        try:
            loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
            for i in range(n_outer):
                sky_plain, gain_plain, loss_plain = self._outer_dirty_map(
                    sky_coeffs,
                    n_minor=n_minor,
                    step_size=step_size,
                    beam_reg=beam_reg,
                    momentum=momentum,
                    verbose=verbose,
                )

                x_vec = self._flatten_sky_coeffs(sky_coeffs)
                g_vec = self._flatten_sky_coeffs(sky_plain)
                f_vec = g_vec - x_vec

                history_g.append(g_vec.copy())
                history_f.append(f_vec.copy())
                if len(history_g) > aa_history:
                    history_g.pop(0)
                    history_f.pop(0)

                sky_next = sky_plain
                gain_next = gain_plain
                loss_next = loss_plain
                used_anderson = False

                if i >= aa_start and len(history_f) >= 2:
                    F = np.stack(history_f, axis=0)
                    beta = self._anderson_coeffs(F, ridge=aa_ridge)
                    if np.all(np.isfinite(beta)) and float(np.max(np.abs(beta))) <= aa_max_weight:
                        g_mix = np.tensordot(beta, np.stack(history_g, axis=0), axes=1)
                        sky_aa = self._unflatten_sky_coeffs(g_mix, sky_coeffs.shape)
                        sky_cand = ((1.0 - aa_damping) * sky_plain + aa_damping * sky_aa).astype(DTYPE_R)
                        gain_cand, _ = self.fit_gains_linear(sky_cand)
                        loss_cand = float(self._jit_loss({'sky_coeffs': sky_cand, **gain_cand}))
                        if loss_cand < loss_plain:
                            sky_next = sky_cand
                            gain_next = gain_cand
                            loss_next = loss_cand
                            used_anderson = True

                sky_coeffs = sky_next
                gain_params = gain_next
                loss = loss_next

                if polish_every and ((i + 1) % polish_every == 0):
                    sky_coeffs, loss = self.fit_sky_only(
                        sky_coeffs, gain_params, maxiter=polish_maxiter, tol=1e-7
                    )
                    gain_params, _ = self.fit_gains_linear(sky_coeffs)

                if verbose:
                    tag = 'AA' if used_anderson else 'plain'
                    print(f'  outer {i:03d} [{tag}]: loss={loss:.4e}')
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return sky_coeffs, gain_params, loss

    def fit_alternating_dirty(
        self,
        sky_coeffs,
        gain_params,
        n_outer: int = 10,
        n_minor: int = 3,
        step_size: float = 0.5,
        beam_reg: float = 1e-3,
        momentum: float = 0.0,
        polish_every: int = 0,
        polish_maxiter: int = 5,
        verbose: bool = False,
        _stop_flag=None,
    ):
        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None
        try:
            loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
            for i in range(n_outer):
                gain_params, _ = self.fit_gains_linear(sky_coeffs)
                sky_coeffs, loss = self.fit_sky_dirty(
                    sky_coeffs,
                    gain_params,
                    n_minor=n_minor,
                    step_size=step_size,
                    beam_reg=beam_reg,
                    momentum=momentum,
                    verbose=verbose,
                )
                if polish_every and ((i + 1) % polish_every == 0):
                    sky_coeffs, loss = self.fit_sky_only(sky_coeffs, gain_params, maxiter=polish_maxiter, tol=1e-7)
                if verbose:
                    print(f'  outer {i:03d}: loss={loss:.4e}')
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return sky_coeffs, gain_params, loss

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
        if _stop_flag is None:
            stop = _StopFitFlag()
            old_handler = _StopFitFlag.install(stop)
        else:
            stop = _stop_flag
            old_handler = None
        try:
            loss = float(self._jit_loss({'sky_coeffs': sky_coeffs, **gain_params}))
            for i in range(n_outer):
                sky_coeffs, _ = self.fit_sky_only(sky_coeffs, gain_params, maxiter=sky_maxiter, tol=sky_tol)
                gain_params, loss = self.fit_gains_linear(sky_coeffs)
                if verbose:
                    print(f'  outer {i:03d}: loss={loss:.4e}')
                if stop.stop:
                    break
        finally:
            if _stop_flag is None:
                _StopFitFlag.restore(old_handler)
        return sky_coeffs, gain_params, loss

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
