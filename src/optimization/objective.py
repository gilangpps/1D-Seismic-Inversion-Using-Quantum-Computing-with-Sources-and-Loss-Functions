import numpy as np
from typing import Dict, Optional, Callable

from src.wave import evolve_1d_wave
from src.encoding import quantum_reconstruct


class SeismicObjective:
    """
    Objective function for seismic inversion.
    
    Computes the reconstruction loss between classical wavefield and 
    quantum reconstruction. This represents the misfit that the inversion
    algorithm attempts to minimize.
    
    Mathematical formulation:
        J(m) = (1/N) * Σₜ ||u_classical(t; m) - u_quantum(t; m)||²
        
    where m is the elastic parameter (mu), u_classical is the forward
    wavefield solution, and u_quantum is the quantum reconstruction.
    """
    
    def __init__(
        self,
        nx: int,
        dx: float,
        dt: float,
        steps: int,
        measure_every: int = 4,
        shots: int = 1000,
        bc: str = 'dirichlet',
        seed: int = 42,
    ):
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.steps = steps
        self.measure_every = measure_every
        self.shots = shots
        self.bc = bc
        self.seed = seed
    
    def forward_simulate(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """
        Compute forward wavefield simulation.
        
        Args:
            mu_arr: Elastic modulus at grid points (length nx+1)
            rho_arr: Density at grid points (length nx)
            u0: Initial displacement field
            v0: Initial velocity field
            
        Returns:
            List of wavefield snapshots at each time step
        """
        nx = self.nx
        
        u0_bc = np.zeros(nx + 2)
        u0_bc[1:-1] = u0
        u1_bc = u0_bc.copy()
        
        rho_bc = np.zeros(nx + 2)
        rho_bc[1:-1] = rho_arr
        rho_bc[0] = rho_arr[0]
        rho_bc[-1] = rho_arr[-1]
        
        mu_bc = np.zeros(nx + 2)
        mu_bc[1:min(len(mu_arr) + 1, nx + 2)] = mu_arr[:min(len(mu_arr), nx + 1)]
        mu_bc[0] = mu_arr[0]
        mu_bc[-1] = mu_arr[-1]
        
        fields = evolve_1d_wave(
            u0_bc, u1_bc, dx=self.dx, dt=self.dt,
            mu=mu_bc, rho=rho_bc,
            source_func=None, steps=self.steps, bc=self.bc
        )
        return fields
    
    def compute_loss(
        self,
        fields: list,
        use_deterministic: bool = False,
    ) -> float:
        """
        Compute reconstruction loss between classical and quantum wavefields.
        
        The loss is the mean squared error (MSE) integrated over all time steps:
            L = (1/T) Σₜ (1/N) ||u_classical(t) - u_quantum(t)||²
        
        Args:
            fields: List of wavefield snapshots from forward simulation
            use_deterministic: If True, use deterministic reconstruction (no shots)
            
        Returns:
            Average reconstruction loss (scalar)
        """
        loss_arr = []
        
        for i in range(1, len(fields)):
            qr = quantum_reconstruct(fields[i], shots=None if use_deterministic else self.shots)
            u_classical = fields[i][1:-1]
            u_quantum = qr[1:-1]
            loss_val = np.mean((u_classical - u_quantum) ** 2)
            loss_arr.append(loss_val)
        
        return float(np.mean(loss_arr))
    
    def compute_loss_and_fields(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        use_deterministic: bool = False,
    ) -> tuple:
        """
        Convenience method: run forward simulation and compute loss in one call.
        
        Returns:
            Tuple of (loss, fields)
        """
        fields = self.forward_simulate(mu_arr, rho_arr, u0, v0)
        loss = self.compute_loss(fields, use_deterministic)
        return loss, fields
    
    def compute_time_series_loss(
        self,
        fields: list,
    ) -> np.ndarray:
        """
        Compute loss at each time step (not averaged).
        
        Useful for debugging and visualization.
        """
        loss_arr = []
        for i in range(1, len(fields)):
            qr = quantum_reconstruct(fields[i], shots=None)
            u_classical = fields[i][1:-1]
            u_quantum = qr[1:-1]
            loss_val = np.mean((u_classical - u_quantum) ** 2)
            loss_arr.append(loss_val)
        return np.array(loss_arr)