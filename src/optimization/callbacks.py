import numpy as np
from typing import Optional


class OptimizationLogger:
    """Structured logging for optimization iterations."""
    
    def __init__(self, log_file: str = "optimization.log"):
        import logging
        self.log_file = log_file
        self.logger = logging.getLogger("optimization")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler  
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_iteration(self, iteration: int, loss: float, 
                     gradient_norm: float, param_change: float):
        """Log details for a single optimization iteration."""
        self.logger.info(
            f"Iteration {iteration:3d} | Loss: {loss:12.6e} | "
            f"Grad norm: {gradient_norm:10.6e} | "
            f"Delta param: {param_change:10.6e}"
        )
    
    def log_convergence(self, iteration: int, loss: float):
        """Log convergence message."""
        self.logger.info(
            f"Converged at iteration {iteration} with loss {loss:.6e}"
        )
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
    
    def info(self, message: str):
        """Info log method for compatibility."""
        self.logger.info(message)


class LossHistoryCallback:
    """Callback to track loss history for analysis and plotting."""
    
    def __init__(self):
        self.loss_history = []
        self.loss_time_series = []
        self.iterations = []
    
    def __call__(self, iteration: int, loss: float, 
                 mu_arr: np.ndarray, fields: list):
        """Store loss and related data each iteration."""
        self.iterations.append(iteration)
        self.loss_history.append(loss)
        
        # Compute reconstruction error time series (placeholder - should use actual objective)
        loss_ts = np.zeros(len(fields) - 1)
        if len(fields) > 1:
            # Simple reconstruction error - replace with actual quantum reconstruction
            for i in range(1, len(fields)):
                u_classical = fields[i][1:-1]
                # Use classical field for now - would need quantum_reconstruct
                loss_ts[i - 1] = np.mean((u_classical - u_classical) ** 2)
        
        self.loss_time_series.append(loss_ts)


class ConvergenceReport:
    """Generate convergence report and analysis."""
    
    def __init__(
        self,
        loss_history: list,
        iterations: list,
        configs: dict,
        output_file: str = "convergence_report.txt"
    ):
        self.loss_history = loss_history
        self.iterations = iterations
        self.configs = configs
        self.output_file = output_file
    
    def generate_report(self) -> dict:
        """Generate convergence statistics and write report."""
        if not self.loss_history:
            return {}
        
        loss_array = np.array(self.loss_history)
        iterations_array = np.array(self.iterations)
        
        # Basic statistics
        initial_loss = loss_array[0]
        final_loss = loss_array[-1]
        min_loss_idx = np.argmin(loss_array)
        min_loss = loss_array[min_loss_idx]
        
        # Convergence metrics
        if initial_loss > 0:
            loss_reduction = (initial_loss - final_loss) / initial_loss
        else:
            loss_reduction = 0.0
        
        avg_loss_reduction_per_iter = loss_reduction / max(1, len(loss_array) - 1)
        
        # Convergence rate (log-scale)
        if final_loss > 0:
            log_init = np.log10(max(initial_loss, 1e-16))
            log_final = np.log10(final_loss)
            iterations_converged = max(0, min_loss_idx - 1)
            convergence_rate = (log_final - log_init) / max(1, iterations_converged)
        else:
            convergence_rate = np.nan
        
        report = {
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'minimum_loss': float(min_loss),
            'minimum_loss_iteration': int(min_loss_idx),
            'loss_reduction_ratio': float(loss_reduction),
            'average_loss_reduction_per_iteration': float(avg_loss_reduction_per_iter),
            'convergence_rate_per_iteration': float(convergence_rate),
            'total_iterations': len(loss_array),
            'converged_normally': loss_reduction > 0.01,
        }
        
        # Write detailed report
        with open(self.output_file, 'w') as f:
            f.write("Seismic Inversion Convergence Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Initial loss: {initial_loss:.6e}\n")
            f.write(f"Final loss: {final_loss:.6e}\n")
            f.write(f"Minimum loss: {min_loss:.6e} (at iteration {min_loss_idx})\n")
            f.write(f"Loss reduction: {loss_reduction*100:.2f}%\n")
            f.write(f"Average reduction per iteration: {avg_loss_reduction_per_iter*100:.4f}%\n")
            f.write(f"Iterations completed: {len(loss_array)}\n")
            f.write(f"Config: {self.configs}\n")
        
        return report