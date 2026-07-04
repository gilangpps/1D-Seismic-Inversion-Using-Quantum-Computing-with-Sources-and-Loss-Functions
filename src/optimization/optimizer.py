import time
import numpy as np
from typing import Dict, Optional, Callable, List, Tuple

from .objective import SeismicObjective
from .gradient import FiniteDifferenceGradient
from .callbacks import (
    OptimizationLogger,
    LossHistoryCallback,
    ConvergenceReport,
)


class AdamOptimizer:
    """
    Adam optimizer for seismic inversion.
    
    Adam (Adaptive Moment Estimation) combines advantages of RMSprop and
    momentum optimization methods.
    
    Mathematical formulation:
        mₜ = β₁·m₍ₜ₋₁₎ + (1-β₁)·gₜ
        vₜ = β₂·v₍ₜ₋₁₎ + (1-β₂)·gₜ²
        m̂ₜ = mₜ / (1-β₁ᵗ)
        v̂ₜ = vₜ / (1-β₂ᵗ)
        θₜ₊₁ = θₜ - α·m̂ₜ / (√v̂ₜ + ε)
        
    where:
        θ: model parameters (mu)
        g: gradient
        m: first moment vector
        v: second moment vector
        α: learning rate
        β₁, β₂: exponential decay rates
        ε: numerical stability constant
    
    Adam is chosen because:
    1. Handles noisy gradients well (from finite-difference and quantum simulation)
    2. Adaptively adjusts learning rate per parameter
    3. Often converges faster than simple gradient descent
    4. Computationally efficient for this problem size
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
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Base learning rate α
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment  
            epsilon: Small constant for numerical stability
            lr_scheduler: Learning rate scheduler configuration
            gradient_clip: Max norm for gradient clipping
            param_clip: Max norm for parameter clipping
        """
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr_scheduler = lr_scheduler
        self.gradient_clip = gradient_clip
        self.param_clip = param_clip
        
        # Initialize state
        self.m = None
        self.v = None
        self.t = 0
        
    def step(
        self,
        theta: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one optimization step.
        
        Args:
            theta: Current parameters (mu)
            gradient: Current gradient ∇J(θ)
            
        Returns:
            Updated parameters θ₊₁
        """
        self.t += 1
        
        # Initialize state on first call
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second raw moment estimate  
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Apply gradient clipping if specified
        if self.gradient_clip is not None:
            grad_norm = np.linalg.norm(m_hat)
            if grad_norm > self.gradient_clip:
                m_hat = m_hat * self.gradient_clip / (grad_norm + self.epsilon)
        
        # Compute parameter update
        delta = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Apply parameter clipping if specified
        if self.param_clip is not None:
            param_norm = np.linalg.norm(delta)
            if param_norm > self.param_clip:
                delta = delta * self.param_clip / (param_norm + self.epsilon)
        
        # Update parameters
        new_theta = theta + delta
        
        return new_theta
    
    def _update_learning_rate(self):
        """Update learning rate according to scheduler."""
        # Implement scheduler logic here
        # Common schedulers: exponential, step, cosine annealing, etc.
        pass
    
    def state_dict(self) -> Dict:
        """Return optimizer state."""
        return {
            'alpha': self.alpha,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'm': self.m.copy() if self.m is not None else None,
            'v': self.v.copy() if self.v is not None else None,
            't': self.t,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state."""
        self.alpha = state_dict['alpha']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.epsilon = state_dict['epsilon']
        self.m = state_dict['m'].copy() if state_dict['m'] is not None else None
        self.v = state_dict['v'].copy() if state_dict['v'] is not None else None
        self.t = state_dict['t']


class SeismicOptimizer:
    """
    Main optimization driver for seismic inversion.
    
    This class orchestrates the iterative optimization process using the
    equation structure specified in the requirements:
    
    for iteration:
        Forward simulation ↓
        Quantum reconstruction ↓
        Compute loss ↓
        Compute model gradient ↓
        Update model ↓
        Re-run forward simulation ↓
        Repeat until convergence
    """
    
    def __init__(
        self,
        configs: Dict,
        objective: SeismicObjective,
        gradient: FiniteDifferenceGradient,
        loss_history_callback: LossHistoryCallback,
        logger: Optional[OptimizationLogger] = None,
        max_iterations: int = 100,
        convergence_tolerance: float = 1e-5,
        early_stopping_patience: int = 20,
        learning_rate: float = 0.01,
        use_deterministic: bool = False,
    ):
        """
        Initialize optimization driver.
        
        Args:
            configs: Experiment configuration dictionary
            objective: Objective function handler
            gradient: Gradient computation handler
            loss_history_callback: Callback for tracking loss history
            logger: Structured logger for iteration messages
            max_iterations: Maximum number of optimization iterations
            convergence_tolerance: Tolerance for convergence stopping
            early_stopping_patience: Patience for early stopping
            learning_rate: Initial learning rate
            use_deterministic: Use deterministic reconstruction (no shots)
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
        
        # Initialize state
        self.mu_arr = np.array(configs['mu'])
        self.rho_arr = np.array(configs['rho'])
        self.u0 = np.array(configs['u0'])
        self.v0 = np.array(configs['v0'])
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(
            learning_rate=learning_rate,
            gradient_clip=1e6,  # Larger clip to allow meaningful updates
            param_clip=1e6,      # Larger clip to allow meaningful updates
        )
        
        # Optimization tracking
        self.iteration = 0
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.convergence_reached = False
        
        # Check for NaN or divergence in initial parameters
        self._check_initialization()
    
    def _check_initialization(self):
        """Check for NaN or invalid values in initial parameters."""
        if np.any(np.isnan(self.mu_arr)):
            self.logger.log_error("Initial mu contains NaN values")
            self.mu_arr = self._safe_parameter_update(self.mu_arr, 1e6)
        
        if np.any(np.isnan(self.rho_arr)):
            self.logger.log_error("Initial rho contains NaN values")
            self.rho_arr = self._safe_parameter_update(self.rho_arr, 1e3)
    
    def _safe_parameter_update(self, params: np.ndarray, magnitude: float) -> np.ndarray:
        """Apply safe updates to parameters (replace NaN/inf with reasonable values)."""
        params = np.where(np.isnan(params), np.random.randn(*params.shape) * magnitude, params)
        params = np.where(np.isinf(params), np.sign(params) * magnitude, params)
        return params
    
    def run_optimization(self) -> Dict:
        """
        Execute the iterative optimization process.
        
        Returns:
            Dictionary containing optimization results and metadata
        """
        start_time = time.time()
        
        # Run initial forward simulation
        initial_loss, initial_fields = self.objective.compute_loss_and_fields(
            self.mu_arr, self.rho_arr, self.u0, self.v0,
            self.use_deterministic
        )
        
        self.logger.log_iteration(
            self.iteration, initial_loss, 0.0, 0.0
        )
        
        # Store initial state
        history = {
            'iterations': [self.iteration],
            'losses': [initial_loss],
            'mu_history': [self.mu_arr.copy()],
            'fields_history': [initial_fields],
            'loss_time_series': [self.objective.compute_time_series_loss(initial_fields)],
        }
        
        # Main optimization loop
        while self.iteration < self.max_iterations and not self.convergence_reached:
            self.iteration += 1
            
            # Prepare gradient function that includes current state
            def objective_fn(mu_arr, rho_arr, u0, v0):
                loss, fields = self.objective.compute_loss_and_fields(
                    mu_arr, rho_arr, u0, v0,
                    self.use_deterministic
                )
                return loss, fields
            
            # Update gradient function to use current objective
            self.gradient.objective_fn = objective_fn
            
            # Compute gradient using finite differences
            gradient = self.gradient.compute(
                self.mu_arr, self.rho_arr, self.u0, self.v0
            )
            
            # Update gradient with regularization
            gradient = self.gradient.compute_with_regularization(
                self.mu_arr, self.rho_arr, self.u0, self.v0,
                reg_weight=0.01
            )
            
            # Compute gradient norm for logging
            gradient_norm = np.linalg.norm(gradient)
            
            # Check for numerical issues
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                self.logger.log_error(
                    f"Iteration {self.iteration}: Gradient contains NaN/inf values"
                )
                break
            
            # Store parameter change before update
            old_mu = self.mu_arr.copy()
            
            # Update parameters using Adam optimizer
            self.mu_arr = self.optimizer.step(
                self.mu_arr, gradient
            )
            
            # Check for parameter divergence
            if np.any(np.isnan(self.mu_arr)) or np.any(np.isinf(self.mu_arr)):
                self.logger.log_error(
                    f"Iteration {self.iteration}: Model parameters contain NaN/inf values"
                )
                self.mu_arr = self._safe_parameter_update(self.mu_arr, 1e10)
                break
            
            # Compute parameter change for logging
            param_change = np.linalg.norm(self.mu_arr - old_mu)
            
            # Update model fields using new parameters
            loss, fields = self.objective.compute_loss_and_fields(
                self.mu_arr, self.rho_arr, self.u0, self.v0,
                self.use_deterministic
            )
            
            # Update convergence tracking
            self.loss_history_callback(self.iteration, loss, self.mu_arr, fields)
            
            # Check for convergence
            loss_improvement = abs(self.best_loss - loss)
            if loss_improvement < self.convergence_tolerance:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0
                self.best_loss = min(self.best_loss, loss)
            
            # Check for divergence
            if loss > 10 * self.best_loss and self.iteration > 10:
                self.logger.log_warning(
                    f"Iteration {self.iteration}: Loss increased dramatically"
                )
            
            # Log iteration
            self.logger.log_iteration(
                self.iteration, loss, gradient_norm, param_change
            )
            
            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.log_convergence(
                    self.iteration, loss
                )
                self.convergence_reached = True
                break
            
            # Store iteration data
            history['iterations'].append(self.iteration)
            history['losses'].append(loss)
            history['mu_history'].append(self.mu_arr.copy())
            history['fields_history'].append(fields)
            history['loss_time_series'].append(self.objective.compute_time_series_loss(fields))
        
        # Generate convergence report
        report = ConvergenceReport(
            history['losses'], history['iterations'], self.configs
        ).generate_report()
        
        # Compile final results
        results = {
            'mu_final': self.mu_arr,
            'loss_history': history['losses'],
            'iteration_history': history['iterations'],
            'mu_history': history['mu_history'],
            'convergence_report': report,
            'final_loss': history['losses'][-1],
            'num_iterations': self.iteration,
            'convergence_reached': self.convergence_reached,
        }
        
        execution_time = time.time() - start_time
        self.logger.info(f"Optimization completed in {execution_time:.2f} seconds")
        
        return results


# Export public API
__all__ = ['AdamOptimizer', 'SeismicOptimizer']