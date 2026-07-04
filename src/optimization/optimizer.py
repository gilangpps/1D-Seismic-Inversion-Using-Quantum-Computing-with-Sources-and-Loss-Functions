import time
import numpy as np
from typing import Dict, Optional, Callable, List, Tuple
from collections import deque

from .objective import SeismicObjective
from .gradient import FiniteDifferenceGradient
from .callbacks import (
    OptimizationLogger,
    LossHistoryCallback,
    ConvergenceReport,
)


class AdamOptimizer:
    """
    Adam optimizer — Adaptive Moment Estimation (Kingma & Ba, 2015).

    MATHEMATICAL FORMULATION:
    ─────────────────────────
        mₜ = β₁·m₍ₜ₋₁₎ + (1−β₁)·gₜ          first moment (mean)
        vₜ = β₂·v₍ₜ₋₁₎ + (1−β₂)·gₜ²         second moment (uncentred variance)
        m̂ₜ = mₜ / (1 − β₁ᵗ)                  bias-corrected first moment
        v̂ₜ = vₜ / (1 − β₂ᵗ)                  bias-corrected second moment
        θₜ₊₁ = θₜ − α · m̂ₜ / (√v̂ₜ + ε)      parameter update (DESCENT)

    Bug 2 (FIXED):
        The original code used:
            new_theta = theta + delta        ← GRADIENT ASCENT
        which maximises J instead of minimising it.

        Correct formulation:
            new_theta = theta - delta        ← GRADIENT DESCENT

        Sign error explanation:
            Adam computes delta = α·m̂/√v̂ > 0 when gradient > 0.
            Adding delta moves θ in the direction of increasing J.
            We must SUBTRACT to decrease J.

    WHY ADAM FOR SEISMIC INVERSION:
    ────────────────────────────────
        1. Handles stochastic/noisy gradients (quantum shot noise)
        2. Per-parameter adaptive learning rate handles the mixed-scale
           mu ~ 1e10 Pa problem gracefully
        3. Bias-corrected moments improve early-iteration stability
        4. Computationally cheap for the small parameter vectors here
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        lr_scheduler: Optional[Dict] = None,
        gradient_clip: Optional[float] = None,
        param_clip: Optional[float] = None,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            learning_rate:  Base learning rate α.
            beta1:          First-moment decay rate (default 0.9).
            beta2:          Second-moment decay rate (default 0.999).
            epsilon:        Numerical stability constant (default 1e-8).
            lr_scheduler:   Optional scheduler config dict (not yet implemented).
            gradient_clip:  Maximum L2-norm for the bias-corrected first moment
                            before scaling down.  None = no clipping.
            param_clip:     Maximum L2-norm of the parameter update Δθ.
                            None = no clipping.
            weight_decay:   Decoupled weight decay (AdamW style).  Adds
                            −λ·θ to the update (i.e. theta -= lr * wd * theta),
                            penalising large parameter magnitudes.
        """
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr_scheduler = lr_scheduler
        self.gradient_clip = gradient_clip
        self.param_clip = param_clip
        self.weight_decay = weight_decay

        # Optimizer state
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    # ------------------------------------------------------------------ #
    #  Core step                                                           #
    # ------------------------------------------------------------------ #

    def step(
        self,
        theta: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        One Adam update step.

        Args:
            theta:    Current parameters (μ array).
            gradient: Current gradient ∇J(θ).

        Returns:
            Updated parameters θₜ₊₁ = θₜ − Δθ.
        """
        self.t += 1

        # Initialise state on first call
        if self.m is None:
            self.m = np.zeros_like(theta, dtype=float)
            self.v = np.zeros_like(theta, dtype=float)

        # First moment update: biased mean
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient

        # Second moment update: biased uncentred variance
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient ** 2)

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # Optional gradient clipping (on bias-corrected moment)
        if self.gradient_clip is not None:
            grad_norm = np.linalg.norm(m_hat)
            if grad_norm > self.gradient_clip:
                m_hat = m_hat * (self.gradient_clip / (grad_norm + 1e-30))

        # Adaptive per-parameter update
        delta = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Optional parameter-update clipping
        if self.param_clip is not None:
            param_norm = np.linalg.norm(delta)
            if param_norm > self.param_clip:
                delta = delta * (self.param_clip / (param_norm + 1e-30))

        # Bug 2 fix: SUBTRACT delta (gradient descent, not ascent)
        new_theta = theta - delta

        # Decoupled weight decay (AdamW)
        if self.weight_decay > 0.0:
            new_theta = new_theta - self.alpha * self.weight_decay * theta

        return new_theta

    # ------------------------------------------------------------------ #
    #  State serialisation                                                 #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> Dict:
        return {
            'alpha':   self.alpha,
            'beta1':   self.beta1,
            'beta2':   self.beta2,
            'epsilon': self.epsilon,
            'm':       self.m.copy() if self.m is not None else None,
            'v':       self.v.copy() if self.v is not None else None,
            't':       self.t,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.alpha   = state['alpha']
        self.beta1   = state['beta1']
        self.beta2   = state['beta2']
        self.epsilon = state['epsilon']
        self.m       = state['m'].copy() if state['m'] is not None else None
        self.v       = state['v'].copy() if state['v'] is not None else None
        self.t       = state['t']


# ====================================================================== #
#  SeismicOptimizer                                                        #
# ====================================================================== #

class SeismicOptimizer:
    """
    Iterative inversion driver for 1-D seismic elastic-modulus recovery.

    OPTIMIZATION LOOP:
    ──────────────────
        for iteration in range(max_iterations):
            1. Forward simulation with current model μ
            2. Compute misfit J(μ) = (1/T) Σ_t ||u_fwd(t;μ) − u_ref(t)||²
            3. Compute gradient ∂J/∂μ via central finite differences
            4. Adam update:  μ ← μ − α·m̂/√v̂   (gradient DESCENT)
            5. Evaluate new loss for diagnostics
            6. Best-model checkpoint
            7. Moving-average early stopping
            8. Divergence detection

    BUG SUMMARY (all fixed in this version):
    ──────────────────────────────────────────
        Bug 2: Adam sign was +delta → ascent.  Fixed to -delta.
        Bug 3: gradient computed TWICE per iteration (compute() then
               compute_with_regularization()).  Second call overwrites first,
               wasting 2× the compute budget.  Now called ONCE via
               compute_with_regularization().
        Bug 4: gradient_norm was logged from the FIRST (discarded) gradient
               call.  Now logged from the SINGLE authoritative call.

    PHASE 9 IMPROVEMENTS:
    ──────────────────────
        • Best-model checkpoint:  saves μ with lowest observed loss.
        • Moving-average early stopping:  stops when MA-loss improvement
          over a patience window < tolerance (more robust than single-step).
        • Per-iteration diagnostics:  loss, Δloss, grad stats, overlap,
          μ stats, learning rate, param-update norm.
        • Gradient averaging over N_avg shots:  averages gradient over
          multiple stochastic forward passes to reduce quantum shot noise.
    """

    def __init__(
        self,
        configs: Dict,
        objective: SeismicObjective,
        gradient: FiniteDifferenceGradient,
        loss_history_callback: LossHistoryCallback,
        logger: Optional[OptimizationLogger] = None,
        max_iterations: int = 100,
        convergence_tolerance: float = 1e-6,
        early_stopping_patience: int = 20,
        learning_rate: float = 0.01,
        use_deterministic: bool = False,
        reg_weight: float = 0.0,
        n_grad_avg: int = 1,
        ma_window: int = 5,
    ):
        """
        Args:
            configs:                  Experiment configuration dict.
            objective:                SeismicObjective (with reference fields set).
            gradient:                 FiniteDifferenceGradient.
            loss_history_callback:    Callback for loss tracking.
            logger:                   Structured logger.
            max_iterations:           Maximum optimization iterations.
            convergence_tolerance:    Early-stop when MA-loss Δ < tol.
            early_stopping_patience:  Consecutive non-improving iterations.
            learning_rate:            Adam learning rate α.
            use_deterministic:        Use noiseless quantum reconstruction.
            reg_weight:               Tikhonov regularization weight λ.
            n_grad_avg:               Number of gradient evaluations to average
                                      (reduces quantum shot noise).
            ma_window:                Window length for moving-average early stop.
        """
        self.configs = configs
        self.objective = objective
        self.gradient = gradient
        self.loss_history_callback = loss_history_callback
        self.logger = logger or OptimizationLogger()

        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.early_stopping_patience = early_stopping_patience
        self.use_deterministic = use_deterministic
        self.reg_weight = reg_weight
        self.n_grad_avg = max(1, n_grad_avg)
        self.ma_window = max(2, ma_window)

        # Current state
        self.mu_arr  = np.array(configs['mu'],  dtype=float)
        self.rho_arr = np.array(configs['rho'], dtype=float)
        self.u0      = np.array(configs['u0'],  dtype=float)
        self.v0      = np.array(configs.get('v0', np.zeros_like(self.u0)),
                                dtype=float)

        # Adam optimizer — learning rate scaled to mu magnitude
        # (mu ~ 1e10 Pa, so lr = 1e7 gives relative update ~ 1e-3)
        self.optimizer = AdamOptimizer(
            learning_rate=learning_rate,
            gradient_clip=1e13,   # cap at ~1e3 × mu to allow real updates
            param_clip=None,      # let Adam handle scale via v_hat
        )

        # Best-model checkpoint (Phase 9)
        self.best_loss: float = float('inf')
        self.best_mu:   np.ndarray = self.mu_arr.copy()

        # Moving-average buffer for early stopping (Phase 9)
        self._loss_ma_buf: deque = deque(maxlen=self.ma_window)

        # Convergence state
        self.iteration: int = 0
        self.early_stopping_counter: int = 0
        self.convergence_reached: bool = False

        self._check_initialization()

    # ------------------------------------------------------------------ #
    #  Initialisation checks                                               #
    # ------------------------------------------------------------------ #

    def _check_initialization(self) -> None:
        if np.any(np.isnan(self.mu_arr)) or np.any(np.isinf(self.mu_arr)):
            self.logger.log_error("Initial μ contains NaN/Inf — replacing with 1e10")
            self.mu_arr = np.where(
                np.isfinite(self.mu_arr), self.mu_arr, 1e10
            )
        if np.any(np.isnan(self.rho_arr)) or np.any(np.isinf(self.rho_arr)):
            self.logger.log_error("Initial ρ contains NaN/Inf — replacing with 2e3")
            self.rho_arr = np.where(
                np.isfinite(self.rho_arr), self.rho_arr, 2e3
            )

    # ------------------------------------------------------------------ #
    #  Gradient with averaging (Phase 9)                                   #
    # ------------------------------------------------------------------ #

    def _compute_gradient(self) -> Tuple[np.ndarray, float]:
        """
        Compute gradient (optionally averaged over n_grad_avg evaluations).

        Multiple evaluations reduce stochastic quantum shot noise at the cost
        of n_grad_avg × more forward simulations per iteration.

        Returns:
            (gradient, current_loss) tuple.
        """
        # Closure passed to gradient computer each call
        def obj_fn(mu, rho, u0, v0):
            return self.objective.compute_loss_and_fields(
                mu, rho, u0, v0, self.use_deterministic
            )

        self.gradient.objective_fn = obj_fn

        if self.n_grad_avg == 1:
            # Bug 3 fix: single authoritative call to compute_with_regularization
            # (previously called compute() first, then overwrote with
            # compute_with_regularization() — 2× waste, wrong gradient used)
            gradient = self.gradient.compute_with_regularization(
                self.mu_arr, self.rho_arr, self.u0, self.v0,
                reg_weight=self.reg_weight,
            )
        else:
            # Average over n_grad_avg independent evaluations
            grads = []
            for _ in range(self.n_grad_avg):
                g = self.gradient.compute_with_regularization(
                    self.mu_arr, self.rho_arr, self.u0, self.v0,
                    reg_weight=self.reg_weight,
                )
                grads.append(g)
            gradient = np.mean(grads, axis=0)

        # Bug 4 fix: gradient_norm is computed from the SAME gradient that
        # will be used for the Adam step (previously logged from the first
        # discarded call)
        current_loss, _ = self.objective.compute_loss_and_fields(
            self.mu_arr, self.rho_arr, self.u0, self.v0, self.use_deterministic
        )

        return gradient, current_loss

    # ------------------------------------------------------------------ #
    #  Moving-average early stopping (Phase 9)                            #
    # ------------------------------------------------------------------ #

    def _check_early_stopping(self, loss: float) -> bool:
        """
        Moving-average early stopping.

        Stops when the improvement in the moving average of the loss over the
        last ma_window iterations is smaller than convergence_tolerance.

        This is more robust than single-step improvement because it is not
        confused by random up/down fluctuations from shot noise.

        Returns:
            True if optimization should stop.
        """
        self._loss_ma_buf.append(loss)

        if len(self._loss_ma_buf) < self.ma_window:
            return False  # Not enough history yet

        # MA over first half vs second half of window
        half = self.ma_window // 2
        ma_old = np.mean(list(self._loss_ma_buf)[:half])
        ma_new = np.mean(list(self._loss_ma_buf)[half:])

        improvement = ma_old - ma_new  # positive = improving

        if improvement < self.convergence_tolerance:
            self.early_stopping_counter += 1
        else:
            self.early_stopping_counter = 0

        return self.early_stopping_counter >= self.early_stopping_patience

    # ------------------------------------------------------------------ #
    #  Per-iteration diagnostics (Phase 8)                                 #
    # ------------------------------------------------------------------ #

    def _print_diagnostics(
        self,
        iteration: int,
        current_loss: float,
        prev_loss: float,
        gradient: np.ndarray,
        param_update_norm: float,
        overlaps: list,
    ) -> None:
        """
        Print rich per-iteration diagnostics.

        Covers all Phase 8 requirements:
            - Iteration, Current/Previous Loss, Loss Change
            - Gradient Norm/Max/Min/Std
            - Learning Rate (effective = α / √v̂ per parameter)
            - Parameter Update Norm
            - Current/Best Overlap
            - μ statistics
        """
        grad_stats = FiniteDifferenceGradient.gradient_stats(gradient)
        loss_change = current_loss - prev_loss
        sign = "▼" if loss_change < 0 else "▲"

        current_overlap = overlaps[-1][1] if overlaps else float('nan')

        sep = "─" * 70
        print(sep)
        print(f"  Iter {iteration:4d}  |  Loss: {current_loss:12.6e}  {sign} {abs(loss_change):.3e}")
        print(f"  Prev loss:   {prev_loss:12.6e}   |  Δloss: {loss_change:+.3e}")
        print(f"  Grad norm:   {grad_stats['norm']:12.6e}   |  lr:     {self.optimizer.alpha:.3e}")
        print(f"  Grad min:    {grad_stats['min']:12.6e}   |  Grad max: {grad_stats['max']:.3e}")
        print(f"  Grad mean:   {grad_stats['mean']:12.6e}   |  Grad std: {grad_stats['std']:.3e}")
        print(f"  |Δμ|₂:       {param_update_norm:12.6e}   |  Best loss: {self.best_loss:.6e}")
        print(f"  Overlap:     {current_overlap:12.6f}   |  Best μ checkpoint: {'yes' if current_loss <= self.best_loss else 'no'}")
        print(f"  μ min: {self.mu_arr.min():.4e}  μ max: {self.mu_arr.max():.4e}  "
              f"μ mean: {self.mu_arr.mean():.4e}  μ std: {self.mu_arr.std():.4e}")
        print(sep)

    # ------------------------------------------------------------------ #
    #  Main optimization loop                                              #
    # ------------------------------------------------------------------ #

    def run_optimization(self) -> Dict:
        """
        Execute the iterative optimization.

        Returns:
            Dict with keys:
                mu_final, mu_best, loss_history, iteration_history,
                mu_history, overlaps_history, convergence_report,
                final_loss, best_loss, num_iterations, convergence_reached.
        """
        start_time = time.time()

        # ── Initial evaluation ─────────────────────────────────────────
        initial_loss, initial_fields = self.objective.compute_loss_and_fields(
            self.mu_arr, self.rho_arr, self.u0, self.v0, self.use_deterministic
        )
        initial_overlaps = self.objective.compute_overlap_with_reference(initial_fields)

        self.best_loss = initial_loss
        self.best_mu   = self.mu_arr.copy()
        self._loss_ma_buf.append(initial_loss)

        self.logger.log_iteration(self.iteration, initial_loss, 0.0, 0.0)

        history: Dict = {
            'iterations':      [self.iteration],
            'losses':          [initial_loss],
            'mu_history':      [self.mu_arr.copy()],
            'fields_history':  [initial_fields],
            'loss_time_series':[self.objective.compute_time_series_loss(initial_fields)],
            'overlaps_history':[initial_overlaps],
        }

        prev_loss = initial_loss

        # ── Main loop ──────────────────────────────────────────────────
        while self.iteration < self.max_iterations and not self.convergence_reached:
            self.iteration += 1

            # ── Gradient computation (Bug 3 + Bug 4 fixed) ────────────
            gradient, _ = self._compute_gradient()

            # Bug 4 fix: gradient_norm is from the SINGLE correct gradient
            gradient_norm = np.linalg.norm(gradient)

            # Guard against degenerate gradients
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                self.logger.log_error(
                    f"Iter {self.iteration}: gradient contains NaN/Inf — stopping"
                )
                break

            # ── Parameter update (Bug 2 fixed: descent not ascent) ────
            old_mu = self.mu_arr.copy()
            self.mu_arr = self.optimizer.step(self.mu_arr, gradient)

            # Guard against parameter divergence
            if not np.all(np.isfinite(self.mu_arr)):
                self.logger.log_error(
                    f"Iter {self.iteration}: μ contains NaN/Inf after update — reverting"
                )
                self.mu_arr = old_mu.copy()
                break

            param_update_norm = np.linalg.norm(self.mu_arr - old_mu)

            # ── Evaluate new loss with updated model ──────────────────
            loss, fields = self.objective.compute_loss_and_fields(
                self.mu_arr, self.rho_arr, self.u0, self.v0, self.use_deterministic
            )

            # ── Overlap (Phase 8 diagnostic) ──────────────────────────
            overlaps = self.objective.compute_overlap_with_reference(fields)

            # ── Best-model checkpoint (Phase 9) ───────────────────────
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_mu   = self.mu_arr.copy()

            # ── Divergence detection ──────────────────────────────────
            if loss > 100.0 * self.best_loss and self.iteration > 5:
                self.logger.log_warning(
                    f"Iter {self.iteration}: loss {loss:.3e} >> best {self.best_loss:.3e} "
                    "(divergence?) — reverting to best model"
                )
                self.mu_arr = self.best_mu.copy()
                # Reset Adam state to avoid carrying bad momentum
                self.optimizer.m = None
                self.optimizer.v = None

            # ── Phase 8 diagnostics ───────────────────────────────────
            self._print_diagnostics(
                self.iteration, loss, prev_loss, gradient,
                param_update_norm, overlaps
            )

            # ── Callbacks ─────────────────────────────────────────────
            self.loss_history_callback(self.iteration, loss, self.mu_arr, fields)
            self.logger.log_iteration(
                self.iteration, loss, gradient_norm, param_update_norm
            )

            # ── Moving-average early stopping (Phase 9) ───────────────
            if self._check_early_stopping(loss):
                self.logger.log_convergence(self.iteration, loss)
                self.convergence_reached = True

            prev_loss = loss

            # ── Store history ─────────────────────────────────────────
            history['iterations'].append(self.iteration)
            history['losses'].append(loss)
            history['mu_history'].append(self.mu_arr.copy())
            history['fields_history'].append(fields)
            history['loss_time_series'].append(
                self.objective.compute_time_series_loss(fields)
            )
            history['overlaps_history'].append(overlaps)

        # ── Final summary ──────────────────────────────────────────────
        report = ConvergenceReport(
            history['losses'], history['iterations'], self.configs
        ).generate_report()

        elapsed = time.time() - start_time
        self.logger.info(
            f"Optimization complete: {self.iteration} iterations in {elapsed:.2f}s. "
            f"Initial loss={history['losses'][0]:.4e}  "
            f"Final loss={history['losses'][-1]:.4e}  "
            f"Best loss={self.best_loss:.4e}"
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
