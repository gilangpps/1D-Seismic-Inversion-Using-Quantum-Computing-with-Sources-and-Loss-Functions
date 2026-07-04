import numpy as np
from typing import Optional, Callable


class FiniteDifferenceGradient:
    """
    Central finite-difference gradient for seismic inversion.

    SCIENTIFIC BACKGROUND:
    ──────────────────────
    The objective function involves:
        1. Wave-equation PDE solver      (non-trivial chain rule)
        2. Quantum amplitude encoding    (non-differentiable due to shots)
        3. Projective measurement        (stochastic)

    Analytical (adjoint-state) gradients would require deriving the discrete
    adjoint of the leapfrog scheme plus the chain rule through the quantum
    encoding.  For this exploratory code, central finite differences are used:

        ∂J/∂μᵢ ≈ [J(μ + δᵢ eᵢ) − J(μ − δᵢ eᵢ)] / (2 δᵢ)

    STEP SIZE SELECTION (Bug 7 fixed):
    ────────────────────────────────────
    The original code used:
        delta = max(delta_scale * |μᵢ|,  epsilon)
        with epsilon = 1e-8

    For μ ≈ 1e10 Pa and delta_scale = 1e-4, the adaptive term gives:
        delta ≈ 1e-4 × 1e10 = 1e6  ← physically meaningful

    But the absolute floor epsilon = 1e-8 is never reached for mu ~ 1e10,
    so it is harmless but misleadingly small.  The real issue was that
    delta_scale itself was fine (1e-4) but the floor was kept at 1e-8
    to protect against near-zero mu.

    The corrected default:
        epsilon = 1.0   (1 Pa absolute minimum step)

    This prevents division-by-zero for near-zero mu while keeping the
    adaptive term dominant for the realistic mu ~ 1e10 Pa regime.

    Numerical precision note:
        For mu ~ 1e10 and delta = 1e6,
            J(mu+delta) − J(mu−delta) involves differences at the ~1e-5 loss
            scale. The relative step δ/μ = 1e-4 is in the sweet-spot between
            truncation error (too large δ) and cancellation error (too small δ).
            Reducing epsilon below this does not help and can cause cancellation.
    """

    def __init__(
        self,
        objective_fn: Optional[Callable] = None,
        delta_scale: float = 1e-4,
        epsilon: float = 1.0,
    ):
        """
        Args:
            objective_fn:  Callable (mu, rho, u0, v0) → (loss, fields).
                           Injected by SeismicOptimizer at each iteration.
            delta_scale:   Relative step size.  delta_i = delta_scale * |mu_i|.
                           Typical value: 1e-4.  For mu ~ 1e10, delta ~ 1e6.
            epsilon:       Absolute minimum step (Pa).  Protects against
                           near-zero mu while staying physically meaningful.
                           Default 1.0 Pa (was 1e-8 — too small).
        """
        self.objective_fn = objective_fn
        self.delta_scale = delta_scale
        self.epsilon = epsilon

    # ------------------------------------------------------------------ #
    #  Core gradient computation                                           #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> np.ndarray:
        """
        Central finite-difference gradient of J w.r.t. μ.

            ∂J/∂μᵢ = [J(μ + δᵢ eᵢ) − J(μ − δᵢ eᵢ)] / (2 δᵢ)

        Step size:
            δᵢ = max(delta_scale × |μᵢ|,  epsilon)

        For μᵢ ~ 1e10, delta_scale = 1e-4:
            δᵢ ≈ 1e6  (relative error O(δ²) ≈ 1e-8, well above float64 noise)

        Args:
            mu_arr:  Current elastic modulus (length nx+1 or nx+2)
            rho_arr: Density (held fixed during gradient eval)
            u0:      Initial displacement
            v0:      Initial velocity

        Returns:
            Gradient array, same shape as mu_arr.
        """
        n_params = len(mu_arr)
        gradient = np.zeros(n_params)

        for i in range(n_params):
            # Adaptive step: relative to parameter magnitude, with floor
            delta = max(self.delta_scale * abs(mu_arr[i]), self.epsilon)

            mu_plus = mu_arr.copy()
            mu_plus[i] += delta
            loss_plus, _ = self.objective_fn(mu_plus, rho_arr, u0, v0)

            mu_minus = mu_arr.copy()
            mu_minus[i] -= delta
            loss_minus, _ = self.objective_fn(mu_minus, rho_arr, u0, v0)

            gradient[i] = (loss_plus - loss_minus) / (2.0 * delta)

        return gradient

    def compute_with_regularization(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        reg_weight: float = 0.0,
        mu_prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Gradient with optional Tikhonov (L2) regularization.

        Regularized objective:
            J_reg(μ) = J(μ) + λ ||μ − μ_prior||²

        Regularized gradient:
            ∂J_reg/∂μ = ∂J/∂μ + 2λ (μ − μ_prior)

        The regularization discourages large deviations from the prior,
        stabilizing the ill-posed inversion.

        Args:
            mu_arr:     Current model parameters.
            rho_arr:    Density (fixed).
            u0:         Initial displacement.
            v0:         Initial velocity.
            reg_weight: Regularization weight λ ≥ 0.
            mu_prior:   Reference model for regularization.  If None, uses
                        the zero vector (Tikhonov penalty on parameter magnitude).

        Returns:
            Regularized gradient array.
        """
        # Single gradient computation (Bug 3 fix: do NOT call self.compute()
        # separately first — compute_with_regularization IS the canonical call)
        gradient = self.compute(mu_arr, rho_arr, u0, v0)

        if reg_weight > 0.0:
            if mu_prior is None:
                # Penalize magnitude (equivalent to L2 on mu)
                reg_term = 2.0 * reg_weight * mu_arr
            else:
                reg_term = 2.0 * reg_weight * (mu_arr - mu_prior)
            gradient = gradient + reg_term

        return gradient

    # ------------------------------------------------------------------ #
    #  Diagnostic utilities                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def gradient_stats(gradient: np.ndarray) -> dict:
        """
        Compute diagnostic statistics for a gradient vector.

        Returns:
            Dictionary with norm, min, max, mean, std.
        """
        return {
            'norm': float(np.linalg.norm(gradient)),
            'min':  float(np.min(gradient)),
            'max':  float(np.max(gradient)),
            'mean': float(np.mean(gradient)),
            'std':  float(np.std(gradient)),
        }


def compute_gradient_adjoint(
    mu_arr: np.ndarray,
    rho_arr: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    fields: list,
    dx: float,
    dt: float,
) -> np.ndarray:
    """
    Placeholder for future adjoint-state gradient.

    The adjoint-state method computes gradients in O(N) cost independent of
    the number of parameters, versus O(N_params) for finite differences.
    Implementation requires:
        1. Discrete adjoint of the leapfrog wave solver
        2. Chain rule through amplitude encoding
        3. Treatment of stochastic measurement

    Returns zeros until implemented.
    """
    return np.zeros_like(mu_arr)
