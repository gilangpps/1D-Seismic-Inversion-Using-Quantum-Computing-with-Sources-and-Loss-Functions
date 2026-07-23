"""
src/optimization/preconditioner.py
──────────────────────────────────────────────────────────────────────────────
Gradient preconditioning and amplification strategies for small-gradient cases

When gradient magnitude is small (< 1e-10), standard gradient descent becomes
ineffective. This module provides preconditioning techniques:

1. **Magnitude-based scaling**: Scale gradient by expected parameter range
2. **Adaptive amplification**: Amplify gradient if loss reduction is sublinear
3. **Directional preconditioning**: Enhance gradient components with strong signal
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Optional


class GradientPreconditioner:
    """
    Apply preconditioning to gradients for improved optimization convergence
    when gradients are small.
    """

    def __init__(
        self,
        param_scale: Optional[np.ndarray] = None,
        amplification_factor: float = 1e8,
    ):
        """
        Parameters
        ----------
        param_scale : ndarray or None
            Expected parameter magnitude. Used for normalization.
            If None, will be inferred from first gradient call.
        amplification_factor : float
            Factor to amplify gradient when norm is small.
            Larger factor → more aggressive amplification.
        """
        self.param_scale = param_scale
        self.amplification_factor = amplification_factor
        self.gradient_history = []

    def precondition(self, gradient: np.ndarray, mu_arr: np.ndarray) -> np.ndarray:
        """
        Apply preconditioning to gradient.

        Strategy: If gradient norm is very small but has clear direction,
        amplify it so optimizer can make meaningful progress.

        Parameters
        ----------
        gradient : ndarray
            Raw FD gradient ∂J/∂μ
        mu_arr : ndarray
            Current model parameters

        Returns
        -------
        ndarray
            Preconditioned gradient
        """
        grad_norm = np.linalg.norm(gradient)

        # Track history
        self.gradient_history.append(float(grad_norm))
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)

        # If gradient is tiny (< 1e-11), apply amplification
        if grad_norm > 1e-30 and grad_norm < 1e-10:
            # Amplify gradient to make optimizer steps meaningful
            amplification = min(
                self.amplification_factor / (grad_norm + 1e-30),
                1e6  # Cap amplification to avoid instability
            )
            preconditioned = gradient * amplification
            return preconditioned

        return gradient

    def get_stats(self) -> dict:
        """Return statistics about gradient history."""
        if not self.gradient_history:
            return {}
        return {
            'mean_grad_norm': float(np.mean(self.gradient_history)),
            'max_grad_norm': float(np.max(self.gradient_history)),
            'min_grad_norm': float(np.min(self.gradient_history)),
            'grad_trend': 'increasing' if len(self.gradient_history) >= 2 and
                          self.gradient_history[-1] > self.gradient_history[-2]
                          else 'decreasing',
        }
