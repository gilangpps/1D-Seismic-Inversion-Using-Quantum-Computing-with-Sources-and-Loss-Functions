"""
src/optimization/callbacks.py
──────────────────────────────────────────────────────────────────────────────
Optimization callbacks: logging, loss history, convergence report.

Bug 5 fix:
    Original code:
        loss_ts[i] = np.mean((u_classical - u_classical) ** 2)  # always 0
    Fixed: store the actual misfit loss passed from the optimizer.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Optional

from src.encoding import quantum_reconstruct


class OptimizationLogger:
    """Structured file + console logger for optimization iterations."""

    def __init__(self, log_file: str = "optimization.log"):
        import logging
        self.log_file = log_file
        self.logger   = logging.getLogger("seismic_optimization")

        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fmt = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)

            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)   # only warnings+ to console
            ch.setFormatter(fmt)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_iteration(self, iteration, loss, gradient_norm, param_change):
        self.logger.info(
            f"Iter {iteration:4d} | Loss: {loss:12.6e} | "
            f"Grad: {gradient_norm:10.4e} | ΔParam: {param_change:10.4e}"
        )

    def log_convergence(self, iteration, loss):
        self.logger.info(f"CONVERGED at iter {iteration}, loss={loss:.6e}")

    def log_warning(self, msg):
        self.logger.warning(msg)

    def log_error(self, msg):
        self.logger.error(msg)

    def info(self, msg):
        self.logger.info(msg)


class LossHistoryCallback:
    """
    Tracks per-iteration loss history.

    Bug 5 fix: stores the true misfit J(μ) from the optimizer, not a
    zero-valued self-comparison artifact.
    """

    def __init__(self):
        self.iterations:       list = []
        self.loss_history:     list = []
        self.loss_time_series: list = []

    def __call__(
        self,
        iteration: int,
        loss: float,
        mu_arr: np.ndarray,
        fields: list,
    ) -> None:
        """
        Parameters
        ----------
        iteration : int
        loss : float
            True misfit J(μ) from SeismicObjective (Bug 5 fix: use this value).
        mu_arr : np.ndarray
            Current elastic modulus.
        fields : list
            Forward wavefield snapshots for diagnostic plotting.
        """
        self.iterations.append(iteration)
        self.loss_history.append(loss)    # Bug 5 fix: correct loss

        # Per-timestep quantum reconstruction diagnostic
        # (NOT the inversion objective; for visualization only)
        n = max(0, len(fields) - 1)
        loss_ts = np.zeros(n)
        np.random.seed(iteration % 2**31)
        for i in range(1, len(fields)):
            qr  = quantum_reconstruct(fields[i], shots=1000)
            u_c = fields[i][1:-1]
            u_q = qr[1:-1]
            loss_ts[i - 1] = float(np.mean((u_c - u_q) ** 2))
        self.loss_time_series.append(loss_ts)


class ConvergenceReport:
    """Generate convergence statistics and write a text report file."""

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
        if not self.loss_history:
            return {}

        arr          = np.array(self.loss_history, dtype=float)
        init_loss    = float(arr[0])
        final_loss   = float(arr[-1])
        min_idx      = int(np.argmin(arr))
        min_loss     = float(arr[min_idx])
        loss_red     = (init_loss - final_loss) / init_loss if init_loss > 0 else 0.0

        if init_loss > 0 and final_loss > 0:
            conv_rate = (np.log10(max(final_loss, 1e-30))
                       - np.log10(max(init_loss,  1e-30))) / max(1, len(arr) - 1)
        else:
            conv_rate = float('nan')

        report = {
            'initial_loss':              init_loss,
            'final_loss':                final_loss,
            'minimum_loss':              min_loss,
            'minimum_loss_iteration':    min_idx,
            'loss_reduction_ratio':      loss_red,
            'convergence_rate_per_iter': conv_rate,
            'total_iterations':          len(arr),
            'converged_normally':        loss_red > 0.01,
        }

        try:
            with open(self.output_file, 'w') as f:
                f.write("Seismic Inversion Convergence Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Initial loss:            {init_loss:.6e}\n")
                f.write(f"Final loss:              {final_loss:.6e}\n")
                f.write(f"Minimum loss:            {min_loss:.6e} (iter {min_idx})\n")
                f.write(f"Loss reduction:          {loss_red*100:.2f}%\n")
                f.write(f"Conv rate (log10/iter):  {conv_rate:.4f}\n")
                f.write(f"Iterations:              {len(arr)}\n")
                f.write(f"\nConfig: {self.configs}\n")
        except OSError:
            pass

        return report
