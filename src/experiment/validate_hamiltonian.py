"""
src/experiment/validate_hamiltonian.py
──────────────────────────────────────────────────────────────────────────────
Independent Hamiltonian Validation Experiment

PURPOSE:
    Validate that the Hamiltonian H = i(A − Aᵀ)/2 from Schrödingerisation
    actually captures the elastic wave physics by comparing:

    • Quantum trajectory: pure exp(-iHt) evolution from (u0, v0)
    • Classical trajectory: leapfrog finite-difference from (u0, v0)

    Both use IDENTICAL μ, ρ, dx, dt, initial conditions.
    No classical reference is used to guide the quantum evolution.

DISTINCTION FROM EXISTING OVERLAP METRICS:
    • overlap.png / Overlaps sheet:
        Compares u_fwd(t; μ_rec) vs u_ref(t; μ_true) during inversion.
        Measures how well the recovered model fits the true model.
        This is an INVERSION QUALITY metric.

    • error.png / QuantumReconLoss sheet:
        Compares quantum_reconstruct(u_classical) vs u_classical.
        Measures encoding-decoding fidelity (shot noise, hardware noise).
        This is an ENCODING QUALITY diagnostic.

    • THIS EXPERIMENT (new):
        Compares quantum evolution trajectory vs classical trajectory
        for the SAME μ and ρ, starting from SAME initial conditions.
        Measures whether Hamiltonian physics matches PDE physics.
        This is a HAMILTONIAN VALIDATION metric.

METRICS:
    1. Trajectory L2 error: ‖u_quantum(t) − u_classical(t)‖₂ / ‖u_classical(t)‖₂
    2. State overlap: |⟨ψ_quantum(t) | ψ_classical(t)⟩|²
    3. Energy conservation check (classical)
    4. Unitarity check (quantum)

OUTPUT:
    • Excel sheet: HamiltonianValidation (time, L2_error, overlap, energy_classical, norm_quantum)
    • Plot: hamiltonian_validation.png (4 subplots: trajectories, L2 error, overlap, norms)

REFERENCE MODELS 2 (RM2) JUSTIFICATION:
    If the Hamiltonian correctly encodes the wave equation, then:
        exp(-iHt)|ψ(0)⟩ ≈ leapfrog(u0, v0)
    for small t and moderate heterogeneity.

    High overlap (>0.95) provides evidence that the Schrödingerisation
    approach (Schade et al. 2024) successfully maps the elastic wave PDE
    into a quantum evolution that can be run on a quantum computer.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Optional, Callable, Tuple
from scipy.linalg import expm

from src.hamiltonian import build_hamiltonian
from src.wave import evolve_1d_wave
from src.encoding import amplitude_encode, amplitude_decode


def run_hamiltonian_validation(
    mu_arr: np.ndarray,
    rho_arr: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    dx: float,
    dt: float,
    steps: int,
    nx: int,
    bc: str = 'dirichlet',
    source_func: Optional[Callable] = None,
) -> dict:
    """
    Run independent Hamiltonian validation: quantum vs classical evolution.

    Parameters
    ----------
    mu_arr : np.ndarray
        Elastic modulus [Pa], shape (nx,) interior points.
    rho_arr : np.ndarray
        Density [kg/m³], shape (nx,).
    u0 : np.ndarray
        Initial displacement [m], shape (nx,).
    v0 : np.ndarray
        Initial velocity [m/s], shape (nx,).
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].
    steps : int
        Number of time steps.
    nx : int
        Number of interior grid points.
    bc : str
        Boundary condition ('dirichlet' or 'neumann').
    source_func : callable or None
        Source term f(i, t). If None, IC-only excitation.

    Returns
    -------
    dict with keys:
        classical_trajectory : list of np.ndarray
            Classical leapfrog fields (with BC padding).
        quantum_trajectory : list of np.ndarray
            Quantum exp(-iHt) fields (with BC padding).
        time : np.ndarray
            Time array [s].
        l2_error : np.ndarray
            Per-timestep relative L2 error.
        overlap : np.ndarray
            Per-timestep quantum state overlap |⟨ψ_q|ψ_c⟩|².
        energy_classical : np.ndarray
            Classical PDE energy [J/m].
        norm_quantum : np.ndarray
            Quantum state norm (should be ~1.0 if unitary).
        mean_l2_error : float
            Mean L2 error over all timesteps.
        mean_overlap : float
            Mean overlap over all timesteps.
    """
    print("\n" + "="*78)
    print("HAMILTONIAN VALIDATION EXPERIMENT")
    print("="*78)
    print("Comparing quantum exp(-iHt) evolution vs classical leapfrog PDE.")
    print(f"Parameters: nx={nx}, dx={dx:.1f}m, dt={dt:.6f}s, steps={steps}")
    print(f"Boundary: {bc}, Source: {'Yes' if source_func else 'IC-only'}")
    print()

    # ── 1. Run classical simulation ──────────────────────────────────────
    print("[1/3] Running classical leapfrog solver...")
    
    # Prepare BC-padded arrays
    u0_bc = np.zeros(nx + 2)
    u0_bc[1:-1] = u0
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

    classical_trajectory = evolve_1d_wave(
        u0_bc, u1_bc,
        dx=dx, dt=dt,
        mu=mu_bc, rho=rho_bc,
        source_func=source_func,
        steps=steps,
        bc=bc,
    )
    print(f"  [OK] Classical trajectory: {len(classical_trajectory)} timesteps")

    # ── 2. Run quantum evolution via sqrt-symmetrization ──────────────
    print("[2/3] Running quantum exp(-iHt) evolution (sqrt-symmetrized H)...")
    
    H_mat, n_qubits, dim, phys_dim, S_op = build_hamiltonian(mu_bc, rho_bc, dx, nx)
    U = expm(-1j * H_mat * dt)
    
    rho_int = rho_bc[1:-1]
    sqrt_rho = np.sqrt(np.maximum(rho_int, 1e-30))
    inv_sqrt_rho = 1.0 / sqrt_rho
    
    print(f"  • Sqrt-symmetrized H: {dim}×{dim} ({n_qubits} qubits), phys_dim={phys_dim}")
    print(f"  • S_op condition number: {np.linalg.cond(S_op):.3e}")
    
    u_current = np.zeros(nx + 2)
    u_current[1:-1] = u0
    v_current = np.zeros(nx + 2)
    v_current[1:-1] = v0

    quantum_trajectory = [u_current.copy()]

    # Time evolution loop
    for step in range(steps):
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
        
        if source_func is not None:
            for i in range(nx):
                src_val = source_func(i+1, step * dt)
                p_new[i] += dt * src_val * inv_sqrt_rho[i]
                v_new[i] = p_new[i] * inv_sqrt_rho[i]
        
        u_current = np.zeros(nx+2)
        u_current[1:-1] = u_new
        v_current = np.zeros(nx+2)
        v_current[1:-1] = v_new
        
        if bc == 'dirichlet':
            u_current[0] = u_current[-1] = 0.0
        elif bc == 'neumann':
            u_current[0] = u_current[1]
            u_current[-1] = u_current[-2]
        
        quantum_trajectory.append(u_current.copy())

    print(f"  [OK] Quantum trajectory: {len(quantum_trajectory)} timesteps")

    # ── 3. Compute comparison metrics ────────────────────────────────────
    print("[3/3] Computing validation metrics...")
    
    n_steps = min(len(classical_trajectory), len(quantum_trajectory))
    time_arr = np.arange(n_steps) * dt
    
    l2_error = np.zeros(n_steps)
    overlap = np.zeros(n_steps)
    energy_classical = np.zeros(n_steps)
    norm_quantum = np.zeros(n_steps)

    for i in range(n_steps):
        u_c = classical_trajectory[i][1:-1]
        u_q = quantum_trajectory[i][1:-1]

        # L2 error
        norm_c = np.linalg.norm(u_c)
        if norm_c > 1e-15:
            l2_error[i] = np.linalg.norm(u_q - u_c) / norm_c
        else:
            l2_error[i] = 0.0

        # Quantum overlap
        sv_c, _, _ = amplitude_encode(classical_trajectory[i])
        sv_q, _, _ = amplitude_encode(quantum_trajectory[i])
        
        if sv_c is not None and sv_q is not None:
            L = max(len(sv_c), len(sv_q))
            a = np.zeros(L, dtype=complex)
            b = np.zeros(L, dtype=complex)
            a[:len(sv_c)] = sv_c
            b[:len(sv_q)] = sv_q
            overlap[i] = float(abs(np.vdot(a, b)) ** 2)
        else:
            overlap[i] = 0.0

        # Classical energy (kinetic + potential, approximate)
        if i > 0:
            u_prev = classical_trajectory[i-1][1:-1]
            v_approx = (u_c - u_prev) / dt
            KE = 0.5 * np.sum(rho_arr * v_approx**2) * dx
            
            # Potential energy: ½ ∫ μ (∂u/∂x)² dx
            du_dx = np.diff(u_c) / dx
            mu_mid = 0.5 * (mu_arr[:-1] + mu_arr[1:]) if len(mu_arr) > 1 else mu_arr
            PE = 0.5 * np.sum(mu_mid[:len(du_dx)] * du_dx**2) * dx
            
            energy_classical[i] = KE + PE
        else:
            energy_classical[i] = 0.0

        # Quantum norm (should be ~1.0)
        sv_q_full, _, norm_q = amplitude_encode(quantum_trajectory[i])
        norm_quantum[i] = norm_q if sv_q_full is not None else 0.0

    mean_l2 = float(np.mean(l2_error[1:]))  # Skip t=0 (identical IC)
    mean_ov = float(np.mean(overlap[1:]))

    print(f"\n  Results:")
    print(f"    Mean L2 error: {mean_l2:.6f}")
    print(f"    Mean overlap:  {mean_ov:.6f}")
    print(f"    Energy range (classical): [{np.min(energy_classical):.3e}, {np.max(energy_classical):.3e}] J/m")
    print(f"    Norm range (quantum):     [{np.min(norm_quantum):.3e}, {np.max(norm_quantum):.3e}]")
    print()

    # Interpretation
    if mean_ov > 0.95:
        print("  [PASS] VALIDATION PASSED: High overlap confirms Hamiltonian captures wave physics.")
    elif mean_ov > 0.85:
        print("  [PARTIAL] VALIDATION PARTIAL: Moderate overlap. Check for heterogeneity effects.")
    else:
        print("  [FAIL] VALIDATION FAILED: Low overlap. Hamiltonian may not accurately represent PDE.")

    print("="*78 + "\n")

    return {
        'classical_trajectory': classical_trajectory,
        'quantum_trajectory': quantum_trajectory,
        'time': time_arr,
        'l2_error': l2_error,
        'overlap': overlap,
        'energy_classical': energy_classical,
        'norm_quantum': norm_quantum,
        'mean_l2_error': mean_l2,
        'mean_overlap': mean_ov,
    }


def format_validation_for_excel(validation: dict) -> dict:
    """
    Format validation results for Excel export.

    Returns dict suitable for pandas DataFrame or openpyxl sheet.
    """
    return {
        'Time [s]': validation['time'],
        'L2 Error (quantum vs classical)': validation['l2_error'],
        'State Overlap |⟨ψ_q|ψ_c⟩|²': validation['overlap'],
        'Classical Energy [J/m]': validation['energy_classical'],
        'Quantum State Norm': validation['norm_quantum'],
    }
