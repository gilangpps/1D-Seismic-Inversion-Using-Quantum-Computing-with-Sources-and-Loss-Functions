import numpy as np
from typing import Optional, Callable


class FiniteDifferenceGradient:
    """
    Finite-difference gradient computation for seismic inversion.
    
    Since the objective function involves:
    1. Wave equation PDE solver (non-trivial derivatives)
    2. Quantum state encoding (discrete, non-differentiable due to shots)
    3. Amplitude-to-probability measurement (non-differentiable)
    
    Analytical gradients would require:
    - Adjoint-state method for wave equation
    - Chain rule through quantum encoding
    - Special treatment for stochastic measurement
    
    This is extremely complex, so we use finite-difference approximation:
    
        ∂J/∂mᵢ ≈ [J(m + ε·eᵢ) - J(m - ε·eᵢ)] / (2ε)
    
    where eᵢ is the unit vector in the i-th parameter direction.
    
    Reference: Since the quantum reconstruction uses stochastic sampling,
    we use a relative step size for numerical stability.
    """
    
    def __init__(
        self,
        objective_fn: Callable,
        delta_scale: float = 1e-4,
        epsilon: float = 1e-8,
    ):
        """
        Initialize finite-difference gradient.
        
        Args:
            objective_fn: Function that takes model parameters and returns loss
            delta_scale: Scaling factor for step size (relative to parameter value)
            epsilon: Absolute minimum step size for near-zero parameters
        """
        self.objective_fn = objective_fn
        self.delta_scale = delta_scale
        self.epsilon = epsilon
    
    def compute(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient of loss with respect to mu using central finite difference.
        
        Mathematical formulation:
            ∂L/∂μᵢ = [L(μ + δ·μᵢ·eᵢ) - L(μ - δ·μᵢ·eᵢ)] / (2·δ·μᵢ)
            
        where δ = delta_scale is a small constant (typically 1e-4 to 1e-6).
        
        Args:
            mu_arr: Elastic modulus parameters
            rho_arr: Density parameters (kept fixed)
            u0: Initial displacement
            v0: Initial velocity
            
        Returns:
            Gradient array with same shape as mu_arr
        """
        n_params = len(mu_arr)
        gradient = np.zeros(n_params)
        
        for i in range(n_params):
            delta = max(self.delta_scale * abs(mu_arr[i]), self.epsilon)
            
            mu_plus = mu_arr.copy()
            mu_plus[i] += delta
            loss_plus, _ = self.objective_fn(mu_plus, rho_arr, u0, v0)
            
            mu_minus = mu_arr.copy()
            mu_minus[i] -= delta
            loss_minus, _ = self.objective_fn(mu_minus, rho_arr, u0, v0)
            
            gradient[i] = (loss_plus - loss_minus) / (2 * delta)
        
        return gradient
    
    def compute_with_regularization(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        reg_weight: float = 0.0,
    ) -> np.ndarray:
        """
        Compute gradient with optional Tikhonov regularization.
        
        Regularization helps stabilize the inversion by penalizing
        large model variations:
            J_reg(m) = J(m) + λ||m - m_prior||²
            
        The gradient becomes:
            ∂J_reg/∂m = ∂J/∂m + 2λ(m - m_prior)
            
        Args:
            mu_arr: Model parameters
            reg_weight: Regularization weight λ (default 0 = no regularization)
            
        Returns:
            Regularized gradient
        """
        gradient = self.compute(mu_arr, rho_arr, u0, v0)
        
        if reg_weight > 0:
            gradient += 2 * reg_weight * mu_arr
        
        return gradient


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
    Placeholder for future adjoint-state gradient implementation.
    
    The adjoint-state method would compute gradients efficiently with
    O(N) cost independent of number of model parameters. However, this
    requires:
    1. Deriving the discrete adjoint of the finite-difference wave solver
    2. Accounting for the quantum reconstruction in the adjoint
    3. Handling the stochastic measurement component
    
    This is a research-level implementation that would require significant
    additional work. For now, finite-difference gradients are used.
    
    Returns:
        Gradient array (placeholder, returns zeros)
    """
    return np.zeros_like(mu_arr)