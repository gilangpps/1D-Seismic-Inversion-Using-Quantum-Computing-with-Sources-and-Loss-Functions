"""
src/optimization/optimizer.py
──────────────────────────────────────────────────────────────────────────────
Adam Optimizer + SeismicOptimizer for 1-D Seismic Inversion

MATHEMATICAL FORMULATION (Adam, Kingma & Ba 2015):
    mₜ = β₁·m_{t-1} + (1−β₁)·gₜ         first moment
    vₜ = β₂·v_{t-1} + (1−β₂)·gₜ²        second moment
    m̂ₜ = mₜ / (1 − β₁ᵗ)                  bias-corrected mean
    v̂ₜ = vₜ / (1 − β₂ᵗ)                  bias-corrected variance
    θₜ₊₁ = θₜ − α · m̂ₜ / (√v̂ₜ + ε)     gradient DESCENT (Bug 2 fix)

BUG FIXES:
    Bug 2: + delta → ASCENT.  Fixed to − delta (descent).
    Bug 3: gradient computed twice per iteration.  Fixed: single call.
    Bug 4: gradient_norm logged from discarded first call.  Fixed.
──────────────────────────────────────────────────────────────────────────────
"""

import time
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

from .objective  import SeismicObjective
from .gradient   import FiniteDifferenceGradient
from .callbacks  import OptimizationLogger, LossHistoryCallback, ConvergenceReport


class AdamOptimizer:
    """
    Adam optimizer.

    Bug 2 fix: update rule is θ − δ (descent), not θ + δ (ascent).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1:  float = 0.9,
        beta2:  float = 0.999,
        epsilon: float = 1e-8,
        gradient_clip: Optional[float] = None,
        param_clip:    Optional[float] = None,
        weight_decay:  float = 0.0,
    ):
        self.alpha         = learning_rate
        self.beta1         = beta1
        self.beta2         = beta2
        self.epsilon       = epsilon
        self.gradient_clip = gradient_clip
        self.param_clip    = param_clip
        self.weight_decay  = weight_decay
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def step(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One Adam parameter update step (gradient descent)."""
        self.t += 1

        if self.m is None:
            self.m = np.zeros_like(theta, dtype=float)
            self.v = np.zeros_like(theta, dtype=float)

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        if self.gradient_clip is not None:
            gn = np.linalg.norm(m_hat)
            if gn > self.gradient_clip:
                m_hat = m_hat * (self.gradient_clip / (gn + 1e-30))

        delta = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        if self.param_clip is not None:
            pn = np.linalg.norm(delta)
            if pn > self.param_clip:
                delta = delta * (self.param_clip / (pn + 1e-30))

        # Bug 2 fix: SUBTRACT delta → gradient DESCENT
        new_theta = theta - delta

        if self.weight_decay > 0.0:
            new_theta -= self.alpha * self.weight_decay * theta

        return new_theta

    def state_dict(self) -> Dict:
        return {
            'alpha':   self.alpha,   'beta1':  self.beta1,
            'beta2':   self.beta2,   'epsilon': self.epsilon,
            'm':       self.m.copy() if self.m is not None else None,
            'v':       self.v.copy() if self.v is not None else None,
            't':       self.t,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.alpha   = state['alpha'];   self.beta1 = state['beta1']
        self.beta2   = state['beta2'];   self.epsilon = state['epsilon']
        self.m = state['m'].copy() if state['m'] is not None else None
        self.v = state['v'].copy() if state['v'] is not None else None
        self.t = state['t']


class SeismicOptimizer:
    """
    Iterative inversion driver: minimises J(μ) via Adam gradient descent.

    Loop:
        for each iteration:
            gradient, loss ← compute (Bug 3 fix: single call)
            μ ← Adam.step(μ, gradient)          (Bug 2 fix: descent)
            if new_loss < best_loss: save checkpoint
            check moving-average early stopping
    """

    def __init__(
        self,
        configs: Dict,
        objective: SeismicObjective,
        gradient: FiniteDifferenceGradient,
        loss_history_callback: LossHistoryCallback,
        logger: Optional[OptimizationLogger] = None,
        max_iterations: int = 200,
        convergence_tolerance: float = 1e-4,
        early_stopping_patience: int = 40,
        learning_rate: float = 5e7,
        use_deterministic: bool = True,
        reg_weight: float = 0.0,
        n_grad_avg: int = 1,
        ma_window: int = 5,
    ):
        self.configs      = configs
        self.objective    = objective
        self.gradient_obj = gradient
        self.loss_callback = loss_history_callback
        self.logger        = logger or OptimizationLogger()

        self.max_iterations          = max_iterations
        self.convergence_tolerance   = convergence_tolerance
        self.early_stopping_patience = early_stopping_patience
        self.use_deterministic       = use_deterministic
        self.reg_weight              = reg_weight
        self.n_grad_avg              = max(1, n_grad_avg)
        self.ma_window               = max(2, ma_window)

        self.mu_arr  = np.array(configs['mu'],  dtype=float)
        self.rho_arr = np.array(configs['rho'], dtype=float)
        self.u0      = np.array(configs['u0'],  dtype=float)
        self.v0      = np.array(configs.get('v0', np.zeros(len(configs['u0']))), dtype=float)

        # Physical bounds for elastic modulus constraint (Pa)
        # mu_true ranges ~1-4e10, so bounds: [0.01e10, 6e10] (more permissive)
        self.mu_min = 0.01e10
        self.mu_max = 6.0e10

        self.adam = AdamOptimizer(
            learning_rate=learning_rate,
            gradient_clip=1e13,
        )

        self.best_loss = float('inf')
        self.best_mu   = self.mu_arr.copy()
        self._ma_buf: deque = deque(maxlen=self.ma_window)

        self.iteration           = 0
        self.early_stop_counter  = 0
        self.convergence_reached = False

        self._sanitize_initial()

    def _sanitize_initial(self):
        if not np.all(np.isfinite(self.mu_arr)):
            self.mu_arr = np.where(np.isfinite(self.mu_arr), self.mu_arr, 1e10)
        if not np.all(np.isfinite(self.rho_arr)):
            self.rho_arr = np.where(np.isfinite(self.rho_arr), self.rho_arr, 2e3)

    def _clip_parameters(self) -> None:
        """
        Enforce physical constraints on μ after each Adam update:
            - μᵢ > 0 (no negative elastic modulus)
            - μₘᵢₙ ≤ μᵢ ≤ μₘₐₓ (within physical range)
        
        For seismic model: mu_min = 0.1e10 Pa, mu_max = 5.0e10 Pa.
        """
        self.mu_arr = np.clip(self.mu_arr, self.mu_min, self.mu_max)

    def _make_obj_fn(self):
        def fn(mu, rho, u0, v0):
            return self.objective.compute_loss_and_fields(
                mu, rho, u0, v0, self.use_deterministic
            )
        return fn

    def _compute_gradient(self) -> Tuple[np.ndarray, float]:
        """
        Compute gradient (single authoritative call, Bug 3 fix).
        Returns (gradient, current_loss).
        """
        self.gradient_obj.objective_fn = self._make_obj_fn()

        if self.n_grad_avg == 1:
            grad = self.gradient_obj.compute_with_regularization(
                self.mu_arr, self.rho_arr, self.u0, self.v0,
                reg_weight=self.reg_weight,
            )
        else:
            grads = [
                self.gradient_obj.compute_with_regularization(
                    self.mu_arr, self.rho_arr, self.u0, self.v0,
                    reg_weight=self.reg_weight,
                )
                for _ in range(self.n_grad_avg)
            ]
            grad = np.mean(grads, axis=0)

        # Bug 4 fix: current_loss from the SAME model that produced gradient
        current_loss, _ = self.objective.compute_loss_and_fields(
            self.mu_arr, self.rho_arr, self.u0, self.v0, self.use_deterministic
        )
        return grad, current_loss

    def _early_stop_check(self, loss: float) -> bool:
        """Moving-average early stopping."""
        self._ma_buf.append(loss)
        if len(self._ma_buf) < self.ma_window:
            return False
        half    = self.ma_window // 2
        ma_old  = np.mean(list(self._ma_buf)[:half])
        ma_new  = np.mean(list(self._ma_buf)[half:])
        improve = ma_old - ma_new
        if improve < self.convergence_tolerance:
            self.early_stop_counter += 1
        else:
            self.early_stop_counter = 0
        return self.early_stop_counter >= self.early_stopping_patience

    def _print_iter(self, it, loss, prev_loss, grad, update_norm, overlaps):
        gs     = FiniteDifferenceGradient.gradient_stats(grad)
        sign   = 'v' if loss < prev_loss else '^'
        ol     = overlaps[-1][1] if overlaps else float('nan')
        sep    = '-' * 68
        try:
            print(sep)
            print(f"  Iter {it:4d}  Loss: {loss:12.6e}  {sign}  dL: {loss-prev_loss:+.3e}")
            print(f"  Grad norm: {gs['norm']:.3e}  lr: {self.adam.alpha:.3e}  |dmu|: {update_norm:.3e}")
            print(f"  Best loss: {self.best_loss:.6e}  Overlap: {ol:.6f}")
            print(f"  mu  min={self.mu_arr.min():.3e}  max={self.mu_arr.max():.3e}  "
                  f"mean={self.mu_arr.mean():.3e}")
            print(f"  Grad stats: min={gs['min']:.3e}  max={gs['max']:.3e}  mean={gs['mean']:.3e}")
            print(sep)
        except UnicodeEncodeError:
            # Fallback for terminals with limited encoding (e.g. Windows cp1252)
            print(f"Iter {it:4d} Loss={loss:.6e} BestLoss={self.best_loss:.6e} GradNorm={gs['norm']:.3e}")


    def run_optimization(self) -> Dict:
        """Execute iterative inversion.  Returns results dict."""
        t_start = time.time()

        # Initial evaluation
        init_loss, init_fields = self.objective.compute_loss_and_fields(
            self.mu_arr, self.rho_arr, self.u0, self.v0, self.use_deterministic
        )
        init_overlaps = self.objective.compute_overlap_with_reference(init_fields)

        self.best_loss = init_loss
        self.best_mu   = self.mu_arr.copy()
        self._ma_buf.append(init_loss)
        self.logger.log_iteration(0, init_loss, 0.0, 0.0)

        history = {
            'iterations':      [0],
            'losses':          [init_loss],
            'mu_history':      [self.mu_arr.copy()],
            'fields_history':  [init_fields],
            'loss_time_series':[self.objective.compute_time_series_loss(init_fields)],
            'overlaps_history':[init_overlaps],
        }
        prev_loss = init_loss

        while self.iteration < self.max_iterations and not self.convergence_reached:
            self.iteration += 1

            # Gradient (Bug 3 + 4 fix: single canonical call)
            grad, _ = self._compute_gradient()
            grad_norm = float(np.linalg.norm(grad))

            # Early abort: grad_norm exactly 0 at iteration 1 means IC is in null space
            # of (H1-H2), making quantum evolution mu-independent. Don't waste compute.
            if self.iteration == 1 and grad_norm == 0.0:
                msg = (
                    f"Iter {self.iteration}: grad_norm = 0.0 (exact zero).\n"
                    "  IC is in the null space of (H1-H2) — quantum evolution is "
                    "mu-independent.\n"
                    "  Check BUG_REPORT_QUANTUM_INVERSION.md §290 for root cause.\n"
                    "  Aborting optimization to prevent wasted iterations."
                )
                self.logger.log_error(msg)
                print(f"\n  *** {msg} ***\n")
                break

            # AMPLIFICATION: If gradient is very small, amplify for optimization traction
            # This prevents stalling when dealing with small-gradient problems
            if 1e-30 < grad_norm < 1e-10:
                grad_amplification = min(1e8 / (grad_norm + 1e-30), 1e6)
                grad = grad * grad_amplification
                if self.iteration <= 20 or self.iteration % 50 == 0:
                    print(f"  [Iter {self.iteration}] Gradient amplified by {grad_amplification:.2e}")

            if not np.all(np.isfinite(grad)):
                self.logger.log_error(f"Iter {self.iteration}: gradient NaN/Inf — stopping")
                break

            # Adam step (Bug 2 fix: descent)
            mu_old      = self.mu_arr.copy()
            self.mu_arr = self.adam.step(self.mu_arr, grad)

            # Apply physical constraints: μ ∈ [mu_min, mu_max]
            self._clip_parameters()

            if not np.all(np.isfinite(self.mu_arr)):
                self.logger.log_error(f"Iter {self.iteration}: mu NaN/Inf -- reverting")
                self.mu_arr = mu_old.copy()
                break

            update_norm = float(np.linalg.norm(self.mu_arr - mu_old))

            # Evaluate new loss
            loss, fields = self.objective.compute_loss_and_fields(
                self.mu_arr, self.rho_arr, self.u0, self.v0, self.use_deterministic
            )
            overlaps = self.objective.compute_overlap_with_reference(fields)

            # Best-model checkpoint
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_mu   = self.mu_arr.copy()

            # Divergence guard
            if loss > 100.0 * self.best_loss and self.iteration > 5:
                self.logger.log_warning(
                    f"Iter {self.iteration}: divergence (loss {loss:.3e} >> "
                    f"best {self.best_loss:.3e}) — reverting"
                )
                self.mu_arr  = self.best_mu.copy()
                self.adam.m  = None
                self.adam.v  = None

            self._print_iter(self.iteration, loss, prev_loss, grad,
                             update_norm, overlaps)

            self.loss_callback(self.iteration, loss, self.mu_arr, fields)
            self.logger.log_iteration(self.iteration, loss, grad_norm, update_norm)

            if self._early_stop_check(loss):
                self.logger.log_convergence(self.iteration, loss)
                self.convergence_reached = True

            prev_loss = loss

            history['iterations'].append(self.iteration)
            history['losses'].append(loss)
            history['mu_history'].append(self.mu_arr.copy())
            history['fields_history'].append(fields)
            history['loss_time_series'].append(
                self.objective.compute_time_series_loss(fields))
            history['overlaps_history'].append(overlaps)

        report = ConvergenceReport(
            history['losses'], history['iterations'], self.configs
        ).generate_report()

        elapsed = time.time() - t_start
        self.logger.info(
            f"Done: {self.iteration} iters in {elapsed:.1f}s  "
            f"init_loss={history['losses'][0]:.4e}  "
            f"final_loss={history['losses'][-1]:.4e}  "
            f"best_loss={self.best_loss:.4e}"
        )

        return {
            'mu_final':           self.mu_arr,
            'mu_best':            self.best_mu,
            'loss_history':       history['losses'],
            'iteration_history':  history['iterations'],
            'mu_history':         history['mu_history'],
            'overlaps_history':   history['overlaps_history'],
            'convergence_report': report,
            'final_loss':         history['losses'][-1],
            'best_loss':          self.best_loss,
            'num_iterations':     self.iteration,
            'convergence_reached':self.convergence_reached,
        }


__all__ = ['AdamOptimizer', 'SeismicOptimizer']
