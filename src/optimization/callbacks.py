import numpy as np
from typing import Optional
from src.encoding import quantum_reconstruct


class OptimizationLogger:
    """Structured logging for optimization iterations."""

    def __init__(self, log_file: str = "optimization.log"):
        import logging
        self.log_file = log_file
        self.logger = logging.getLogger("optimization")

        # Guard: only add handlers if none exist yet (prevents duplicate log
        # entries when the module is imported multiple times in a session)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_iteration(self, iteration: int, loss: float,
                      gradient_norm: float, param_change: float) -> None:
        self.logger.info(
            f"Iteration {iteration:3d} | Loss: {loss:12.6e} | "
            f"Grad norm: {gradient_norm:10.6e} | "
            f"Delta param: {param_change:10.6e}"
        )

    def log_convergence(self, iteration: int, loss: float) -> None:
        self.logger.info(
            f"Converged at iteration {iteration} with loss {loss:.6e}"
        )

    def log_warning(self, message: str) -> None:
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        self.logger.error(message)

    def info(self, message: str) -> None:
        self.logger.info(message)


class LossHistoryCallback:
    """
    Callback to track per-iteration loss history.

    Bug 5 (FIXED):
    ──────────────
    The original code computed:
        loss_ts[i] = np.mean((u_classical - u_classical) ** 2)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This is identically ZERO because a field is always equal to itself.
    The callback was recording zeros for every time step at every iteration,
    producing a flat "loss" plot that conveyed no information.

    Fix: Store the loss value passed in from the optimizer (which is the
    true misfit J(μ) computed by SeismicObjective.compute_loss).  For the
    time-series breakdown we call quantum_reconstruct to get the actual
    per-timestep reconstruction diagnostic (kept as a secondary metric
    for visualisation, not used to drive the optimizer).
    """

    def __init__(self):
        self.loss_history:      list = []
        self.loss_time_series:  list = []
        self.iterations:        list = []

    def __call__(
        self,
        iteration: int,
        loss: float,
        mu_arr: np.ndarray,
        fields: list,
    ) -> None:
        """
        Store the optimizer's true misfit loss and a per-timestep breakdown.

        Args:
            iteration: Current iteration index.
            loss:      Misfit J(μ) from SeismicObjective (the correct value).
            mu_arr:    Current elastic modulus (stored for post-analysis).
            fields:    Forward wavefield snapshots for this iteration.
        """
        self.iterations.append(iteration)
        self.loss_history.append(loss)   # Bug 5 fix: use the correct loss

        # Per-timestep reconstruction diagnostic (quantum vs classical).
        # NOTE: this is a secondary diagnostic metric, NOT the inversion
        # objective.  It measures how faithfully the quantum circuit
        # reconstructs each snapshot.
        loss_ts = np.zeros(max(0, len(fields) - 1))
        np.random.seed(iteration)   # reproducible shot noise for diagnostics
        for i in range(1, len(fields)):
            qr = quantum_reconstruct(fields[i], shots=1000)
            u_classical = fields[i][1:-1]
            u_quantum   = qr[1:-1]
            # Bug 5 fix: compare against quantum reconstruction, not self
            loss_ts[i - 1] = np.mean((u_classical - u_quantum) ** 2)

        self.loss_time_series.append(loss_ts)


class ConvergenceReport:
    """Generate convergence statistics and write a text report."""

    def __init__(
        self,
        loss_history: list,
        iterations: list,
        configs: dict,
        output_file: str = "convergence_report.txt",
    ):
        self.loss_history = loss_history
        self.iterations   = iterations
        self.configs      = configs
        self.output_file  = output_file

    def generate_report(self) -> dict:
        """Compute statistics and write convergence report file."""
        if not self.loss_history:
            return {}

        loss_array = np.array(self.loss_history)

        initial_loss    = float(loss_array[0])
        final_loss      = float(loss_array[-1])
        min_loss_idx    = int(np.argmin(loss_array))
        min_loss        = float(loss_array[min_loss_idx])

        loss_reduction = (
            (initial_loss - final_loss) / initial_loss
            if initial_loss > 0 else 0.0
        )

        avg_loss_reduction_per_iter = (
            loss_reduction / max(1, len(loss_array) - 1)
        )

        if final_loss > 0 and initial_loss > 0:
            log_init  = np.log10(max(initial_loss, 1e-30))
            log_final = np.log10(max(final_loss,   1e-30))
            convergence_rate = (log_final - log_init) / max(1, len(loss_array) - 1)
        else:
            convergence_rate = float('nan')

        report = {
            'initial_loss':                       initial_loss,
            'final_loss':                         final_loss,
            'minimum_loss':                       min_loss,
            'minimum_loss_iteration':             min_loss_idx,
            'loss_reduction_ratio':               loss_reduction,
            'average_loss_reduction_per_iteration': avg_loss_reduction_per_iter,
            'convergence_rate_per_iteration':     convergence_rate,
            'total_iterations':                   len(loss_array),
            'converged_normally':                 loss_reduction > 0.01,
        }

        with open(self.output_file, 'w') as f:
            f.write("Seismic Inversion Convergence Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Initial loss:                {initial_loss:.6e}\n")
            f.write(f"Final loss:                  {final_loss:.6e}\n")
            f.write(f"Minimum loss:                {min_loss:.6e}"
                    f" (at iteration {min_loss_idx})\n")
            f.write(f"Loss reduction:              {loss_reduction*100:.2f}%\n")
            f.write(f"Avg reduction / iteration:   "
                    f"{avg_loss_reduction_per_iter*100:.4f}%\n")
            f.write(f"Convergence rate (log10/iter):{convergence_rate:.4f}\n")
            f.write(f"Iterations completed:        {len(loss_array)}\n")
            f.write(f"\nConfig: {self.configs}\n")

        return report
