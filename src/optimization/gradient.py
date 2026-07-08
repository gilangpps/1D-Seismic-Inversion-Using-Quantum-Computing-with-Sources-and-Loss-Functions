"""
src/optimization/gradient.py
──────────────────────────────────────────────────────────────────────────────
Central Finite-Difference Gradient for Seismic Inversion

The objective involves a PDE solver + quantum encoding, making analytical
(adjoint-state) gradients research-level.  Central FD is used:

    ∂J/∂μᵢ ≈ [J(μ + δᵢ eᵢ) − J(μ − δᵢ eᵢ)] / (2δᵢ)

Step size (Bug 7 fix):
    δᵢ = max(delta_scale × |μᵢ|,  epsilon_abs)

    For μ ~ 1e10 Pa and delta_scale = 1e-4:  δ ~ 1e6 Pa
    epsilon_abs = 1.0 Pa  (floor for near-zero μ)
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Optional, Callable


class FiniteDifferenceGradient:
    """
    Central finite-difference gradient with adaptive step size.
    """

    def __init__(
        self,
        objective_fn: Optional[Callable] = None,
        delta_scale: float = 1e-4,
        epsilon: float = 1.0,
    ):
        """
        Parameters
        ----------
        objective_fn : callable (mu, rho, u0, v0) → (loss, fields)
            Injected by SeismicOptimizer before each call.
        delta_scale : float
            Relative step:  δ_i = delta_scale × |μ_i|.
        epsilon : float
            Absolute minimum step [Pa].  Protects against near-zero μ.
            Default 1.0 Pa (was 1e-8 — caused catastrophic cancellation).
        """
        self.objective_fn = objective_fn
        self.delta_scale  = delta_scale
        self.epsilon      = epsilon

    def compute(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> np.ndarray:
        """
        Central FD gradient ∂J/∂μ.

        Returns
        -------
        np.ndarray, same shape as mu_arr.
        """
        n_params = len(mu_arr)
        gradient = np.zeros(n_params, dtype=float)

        for i in range(n_params):
            delta = max(self.delta_scale * abs(mu_arr[i]), self.epsilon)

            mu_plus        = mu_arr.copy()
            mu_plus[i]    += delta
            loss_plus, _   = self.objective_fn(mu_plus, rho_arr, u0, v0)

            mu_minus       = mu_arr.copy()
            mu_minus[i]   -= delta
            loss_minus, _  = self.objective_fn(mu_minus, rho_arr, u0, v0)

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
        FD gradient with optional Tikhonov (L2) regularization.

        J_reg(μ) = J(μ) + λ ‖μ − μ_prior‖²
        ∂J_reg/∂μ = ∂J/∂μ + 2λ(μ − μ_prior)

        Bug 3 fix: this is the SINGLE canonical gradient call.
        Do not call self.compute() separately before this method.
        """
        # Single authoritative gradient computation
        gradient = self.compute(mu_arr, rho_arr, u0, v0)

        if reg_weight > 0.0:
            ref = mu_prior if mu_prior is not None else np.zeros_like(mu_arr)
            gradient = gradient + 2.0 * reg_weight * (mu_arr - ref)

        return gradient

    @staticmethod
    def gradient_stats(gradient: np.ndarray) -> dict:
        """Diagnostic statistics for a gradient vector."""
        return {
            'norm': float(np.linalg.norm(gradient)),
            'min':  float(np.min(gradient)),
            'max':  float(np.max(gradient)),
            'mean': float(np.mean(gradient)),
            'std':  float(np.std(gradient)),
        }


def compute_gradient_adjoint(mu_arr, rho_arr, u0, v0, fields,
                              dx, dt) -> np.ndarray:
    """
    Placeholder for future adjoint-state gradient.

    Adjoint-state method computes gradient in O(N) cost regardless of
    parameter count.  Requires discrete adjoint of leapfrog + chain rule
    through quantum encoding.  Returns zeros until implemented.
    """
    return np.zeros_like(mu_arr)
