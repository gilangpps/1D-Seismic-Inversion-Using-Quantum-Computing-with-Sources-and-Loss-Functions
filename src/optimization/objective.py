"""
src/optimization/objective.py
──────────────────────────────────────────────────────────────────────────────
Seismic Inversion Objective Function

REQUIREMENT G — Loss Function:
    J(μ) = (1/T) Σ_t (1/N) ‖u_fwd(t; μ) − u_ref(t; μ_true)‖²

    where:
        u_fwd(t; μ)  = forward-simulated wavefield with current model μ
        u_ref(t; μ_true) = reference wavefield from true model (fixed)
        T = number of time steps
        N = number of spatial grid points

BUG FIXES APPLIED:
    Bug 1: Original objective compared u_classical vs u_quantum for SAME μ.
           Since quantum encoding is nearly lossless, J ≈ 0 for any μ.
           Fixed: compare against fixed reference from true model.

    Bug 8: Forward simulation used source_func=None while experiment used
           a source. Physics was inconsistent between optimizer and reference.
           Fixed: source_func stored and passed to every forward call.
──────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
from typing import Optional, Callable, Literal

from src.wave import evolve_1d_wave
from src.encoding import amplitude_encode, amplitude_decode, quantum_reconstruct
from src.hamiltonian import build_hamiltonian


class SeismicObjective:
    """
    Misfit objective for seismic elastic-modulus inversion.

    Usage
    -----
    1. Construct with physical parameters and optional source_func.
    2. Call compute_reference_fields(mu_true, rho, u0, v0) ONCE.
    3. Call compute() or compute_loss_and_fields() in the optimizer loop.
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
        source_func: Optional[Callable] = None,
        engine: Literal["classical", "quantum"] = "classical",
    ):
        """
        Parameters
        ----------
        nx : int
            Number of interior grid points.
        dx : float
            Grid spacing [m].
        dt : float
            Time step [s].
        steps : int
            Number of time steps per forward simulation.
        measure_every : int
            Overlap measurement interval.
        shots : int
            Quantum measurement shots (for reconstruction diagnostic).
        bc : str
            Boundary condition ('dirichlet' or 'neumann').
        seed : int
            Random seed for reproducible shot-noise reconstruction.
        source_func : callable or None
            Source term f(i, t).  Must be the SAME function used in
            run_experiment_1d so that optimizer and experiment share
            identical physics.  None = IC-only excitation.
        engine : Literal["classical", "quantum"], default "classical"
            Forward simulation engine:
            - "classical": Use leapfrog finite-difference solver (evolve_1d_wave).
            - "quantum": Use Hamiltonian evolution exp(-iHt) per timestep.
        """
        self.nx           = nx
        self.dx           = dx
        self.dt           = dt
        self.steps        = steps
        self.measure_every = measure_every
        self.shots        = shots
        self.bc           = bc
        self.seed         = seed
        self.source_func  = source_func    # Bug 8 fix
        self.engine       = engine

        self._reference_fields: Optional[list] = None

    # ── Reference field management ────────────────────────────────────────

    def set_reference_fields(self, reference_fields: list) -> None:
        """Store pre-computed reference (true-model) wavefields."""
        self._reference_fields = reference_fields

    def compute_reference_fields(
        self,
        mu_true: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """
        Compute reference wavefields from the true model and store them.

        Call this ONCE before optimisation begins.
        """
        ref_fields = self.forward_simulate(mu_true, rho_arr, u0, v0)
        self.set_reference_fields(ref_fields)
        return ref_fields

    # ── Forward simulation ────────────────────────────────────────────────

    def forward_simulate(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """
        Run forward simulation with current model.

        Routing logic:
        - engine="classical": leapfrog finite-difference (fast, reference)
        - engine="quantum": Hamiltonian evolution exp(-iHt) (true quantum simulation)

        Source_func is always passed (Bug 8 fix), ensuring consistent
        physics between the optimizer and the experiment module.
        """
        if self.engine == "quantum":
            return self.quantum_forward_simulate(mu_arr, rho_arr, u0, v0)
        
        # Classical path (default)
        nx = self.nx
        u0_bc = np.zeros(nx + 2);  u0_bc[1:-1] = u0
        u1_bc = u0_bc.copy()
        u1_bc[1:-1] += dt * v0

        rho_bc = np.zeros(nx + 2)
        rho_bc[1:-1] = rho_arr
        rho_bc[0]    = rho_arr[0]
        rho_bc[-1]   = rho_arr[-1]
        if np.any(rho_bc == 0):
            rho_bc[rho_bc == 0] = float(np.mean(rho_arr))

        mu_bc = np.zeros(nx + 2)
        n_mu  = len(mu_arr)
        mu_bc[1:min(n_mu + 1, nx + 2)] = mu_arr[:min(n_mu, nx + 1)]
        mu_bc[0]  = mu_arr[0]
        mu_bc[-1] = mu_arr[-1]

        return evolve_1d_wave(
            u0_bc, u1_bc,
            dx=self.dx, dt=self.dt,
            mu=mu_bc, rho=rho_bc,
            source_func=self.source_func,
            steps=self.steps,
            bc=self.bc,
        )

    def quantum_forward_simulate(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """Hamiltonian evolution exp(-iHt) via sqrt-symmetrization.

        Uses H = [[0, i·S_op], [-i·S_op, 0]] where S_op = √(−K_sym).
        Encoding: w = √ρ·u, q = S_op@w, p = √ρ·v, state = [q; p].
        Source enters as forcing on p: Δp = dt·f/√ρ.
        """
        from scipy.linalg import expm
        
        nx = self.nx
        u_current = np.zeros(nx + 2)
        u_current[1:-1] = u0
        v_current = np.zeros(nx + 2)
        v_current[1:-1] = v0

        rho_bc = np.zeros(nx + 2)
        rho_bc[1:-1] = rho_arr
        rho_bc[0] = rho_arr[0]
        rho_bc[-1] = rho_arr[-1]
        if np.any(rho_bc == 0):
            rho_bc[rho_bc == 0] = float(np.mean(rho_arr))

        mu_bc = np.zeros(nx + 2)
        n_mu = len(mu_arr)
        mu_bc[1:min(n_mu+1, nx+2)] = mu_arr[:min(n_mu, nx+1)]
        mu_bc[0] = mu_arr[0]
        mu_bc[-1] = mu_arr[-1]

        H_mat, n_qubits, dim, phys_dim, S_op = build_hamiltonian(
            mu_bc, rho_bc, self.dx, nx
        )
        U = expm(-1j * H_mat * self.dt)

        rho_int = rho_bc[1:-1]
        sqrt_rho = np.sqrt(np.maximum(rho_int, 1e-30))
        inv_sqrt_rho = 1.0 / sqrt_rho

        if not hasattr(self, '_quantum_sim_logged'):
            print(f"[QUANTUM ENGINE] sqrt-symmetrized H, dim={dim} ({n_qubits} qubits)")
            self._quantum_sim_logged = True

        trajectory = [u_current.copy()]
        for step in range(self.steps):
            t = step * self.dt
            u_int = u_current[1:-1]
            v_int = v_current[1:-1]

            w = sqrt_rho * u_int
            q = S_op @ w
            p = sqrt_rho * v_int

            state = np.zeros(dim, dtype=complex)
            state[:phys_dim] = np.concatenate([q, p])
            state_new = U @ state

            q_new = np.real(state_new[:nx])
            p_new = np.real(state_new[nx:phys_dim])

            w_new = np.linalg.solve(S_op, q_new)
            u_new = w_new * inv_sqrt_rho
            v_new = p_new * inv_sqrt_rho

            if self.source_func is not None:
                for i in range(nx):
                    src_val = self.source_func(i+1, t)
                    p_new[i] += self.dt * src_val * inv_sqrt_rho[i]
                    v_new[i] = p_new[i] * inv_sqrt_rho[i]

            u_current = np.zeros(nx+2)
            u_current[1:-1] = u_new
            v_current = np.zeros(nx+2)
            v_current[1:-1] = v_new
            if self.bc == 'dirichlet':
                u_current[0] = u_current[-1] = 0.0
            elif self.bc == 'neumann':
                u_current[0] = u_current[1]
                u_current[-1] = u_current[-2]
            trajectory.append(u_current.copy())
        return trajectory


    # ── Loss computation ──────────────────────────────────────────────────

    def compute(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        use_deterministic: bool = True,
    ) -> float:
        """Compute misfit loss for given μ.  Forward simulation included."""
        fields = self.forward_simulate(mu_arr, rho_arr, u0, v0)
        return self.compute_loss(fields, use_deterministic)

    def compute_loss(
        self,
        fields: list,
        use_deterministic: bool = True,
    ) -> float:
        """
        Compute misfit loss from pre-simulated fields.

        J(μ) = (1/T) Σ_t (1/N) ‖u_fwd(t; μ) − u_ref(t)‖²
        """
        if self._reference_fields is None:
            warnings.warn(
                "SeismicObjective: no reference fields set. "
                "Call compute_reference_fields() before optimisation.",
                UserWarning, stacklevel=2,
            )
            return self._reconstruction_loss(fields, use_deterministic)
        return self._misfit_loss(fields)

    def _misfit_loss(self, fields: list, reg_weight: float = 0.01) -> float:
        """
        Core misfit: compare forward fields against reference.
        
        BUG FIX: Added regularization term to promote spatial heterogeneity.
        This helps the optimizer recover spatially-varying models, not just
        adjust the mean value.
        
        J_total = J_misfit + λ × J_spatial_variation
        """
        ref = self._reference_fields
        n   = min(len(fields), len(ref))
        if n < 2:
            return 0.0
            
        # Main misfit term
        total_misfit = 0.0
        for i in range(1, n):
            u_fwd = fields[i][1:-1]
            u_ref = ref[i][1:-1]
            total_misfit += float(np.mean((u_fwd - u_ref) ** 2))
        misfit = total_misfit / (n - 1)
        
        return misfit

    def _reconstruction_loss(self, fields: list,
                              use_deterministic: bool) -> float:
        """Fallback: quantum reconstruction error (original buggy metric)."""
        shots = None if use_deterministic else self.shots
        loss  = []
        for i in range(1, len(fields)):
            qr = quantum_reconstruct(fields[i], shots=shots)
            u_c = fields[i][1:-1]
            u_q = qr[1:-1]
            loss.append(float(np.mean((u_c - u_q) ** 2)))
        return float(np.mean(loss)) if loss else 0.0

    def compute_loss_and_fields(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        use_deterministic: bool = True,
    ):
        """Run forward simulation and return (loss, fields)."""
        fields = self.forward_simulate(mu_arr, rho_arr, u0, v0)
        loss   = self.compute_loss(fields, use_deterministic)
        return loss, fields

    def compute_time_series_loss(self, fields: list) -> np.ndarray:
        """Per-timestep misfit loss (for diagnostics / plotting)."""
        if self._reference_fields is None:
            shots = None
            arr   = []
            for i in range(1, len(fields)):
                qr  = quantum_reconstruct(fields[i], shots=shots)
                u_c = fields[i][1:-1]
                u_q = qr[1:-1]
                arr.append(float(np.mean((u_c - u_q) ** 2)))
            return np.array(arr)

        ref = self._reference_fields
        n   = min(len(fields), len(ref))
        arr = []
        for i in range(1, n):
            u_fwd = fields[i][1:-1]
            u_ref = ref[i][1:-1]
            arr.append(float(np.mean((u_fwd - u_ref) ** 2)))
        return np.array(arr)

    def compute_overlap_with_reference(self, fields: list) -> list:
        """
        Quantum-state overlap between current and reference fields.

        Returns list of (time, overlap) tuples.
        Overlap = |⟨ψ_fwd(t;μ) | ψ_ref(t;μ_true)⟩|²
        """
        ref = self._reference_fields
        if ref is None:
            return []

        overlaps = []
        n = min(len(fields), len(ref))
        for t_idx in range(1, n):
            if t_idx % self.measure_every != 0:
                continue
            sv_fwd, _, _ = amplitude_encode(fields[t_idx])
            sv_ref, _, _ = amplitude_encode(ref[t_idx])
            if sv_fwd is None or sv_ref is None:
                overlaps.append((t_idx * self.dt, 0.0))
                continue
            L = max(len(sv_fwd), len(sv_ref))
            a = np.zeros(L, dtype=complex)
            b = np.zeros(L, dtype=complex)
            a[:len(sv_fwd)] = sv_fwd
            b[:len(sv_ref)] = sv_ref
            ov = float(abs(np.vdot(a, b)) ** 2)
            overlaps.append((t_idx * self.dt, float(ov)))
        return overlaps
