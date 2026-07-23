"""
Regression test: raw matrix A matches leapfrog evolution.

This test validates that the first-order system matrix A (before any
Hermitianization) correctly represents the elastic wave PDE by comparing
direct expm(A·t) evolution against the leapfrog finite-difference solver.

Purpose:
    Detect regressions in K (stiffness operator) or A (system matrix)
    construction BEFORE they propagate to quantum Hamiltonian issues.

Expected behavior:
    Mean L2 relative error < 0.2 (allows for leapfrog truncation error)

If this test fails:
    The problem is in the K or A construction itself, NOT the Hermitianization.
"""

import numpy as np
from scipy.linalg import expm

from src.distributions import raised_cosine
from src.wave import evolve_1d_wave


def build_raw_A(mu, rho, dx: float, nx: int):
    """
    Build raw first-order system matrix A (no Hermitianization).
    
    Returns A such that d/dt [u, v] = A [u, v]
    """
    mu_arr  = np.asarray(mu,  dtype=float)
    rho_arr = np.asarray(rho, dtype=float)

    # Build K (elastic stiffness operator)
    K = np.zeros((nx, nx), dtype=float)
    for i in range(nx):
        mu_r = 0.5 * (mu_arr[min(i,     len(mu_arr) - 1)]
                    + mu_arr[min(i + 1, len(mu_arr) - 1)])
        mu_l = 0.5 * (mu_arr[max(i - 1, 0)]
                    + mu_arr[min(i,     len(mu_arr) - 1)])
        rho_i = max(rho_arr[min(i, len(rho_arr) - 1)], 1e-30)

        K[i, i] = -(mu_r + mu_l) / (rho_i * dx * dx)
        if i + 1 < nx:
            K[i, i + 1] =  mu_r / (rho_i * dx * dx)
        if i - 1 >= 0:
            K[i, i - 1] =  mu_l / (rho_i * dx * dx)

    # Assemble first-order system A
    A = np.zeros((2 * nx, 2 * nx), dtype=float)
    A[:nx, nx:]   = np.eye(nx)   # ∂u/∂t = v
    A[nx:, :nx]   = K            # ∂v/∂t = K·u

    return A


def evolve_via_matrix_exponential(u0, v0, A, dt: float, steps: int):
    """
    Evolve [u, v] via direct matrix exponential: psi(t) = expm(A·t) @ psi(0).
    
    Returns:
        u_history: (steps+1, nx)
        v_history: (steps+1, nx)
    """
    nx = len(u0)
    psi = np.concatenate([u0, v0])  # [u, v]
    
    u_hist = [u0.copy()]
    v_hist = [v0.copy()]
    
    # Precompute exp(A·dt)
    exp_A_dt = expm(A * dt)
    
    for _ in range(steps):
        psi = exp_A_dt @ psi
        u_hist.append(psi[:nx].copy())
        v_hist.append(psi[nx:].copy())
    
    return np.array(u_hist), np.array(v_hist)


def test_raw_A_matches_leapfrog():
    """
    Test that raw matrix A (before Hermitianization) matches leapfrog PDE solver.
    
    Uses same parameters as main simulation:
        nx=7, dx=63m, dt=0.005s, steps=40
        μ: raised-cosine 1e10 to 4e10 Pa
        ρ: raised-cosine 2e3 to 4e3 kg/m³
    """
    # Setup parameters (match main.py)
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 40
    
    # Build heterogeneous medium (match main.py parameters)
    mu_min, mu_max = 1.0e10, 4.0e10
    rho_min, rho_max = 2.0e3, 4.0e3
    
    mu_true = raised_cosine(
        value=(mu_max - mu_min),
        length=nx + 1,
        position=nx,
        sigma=6,
        offset=mu_min
    )
    rho = raised_cosine(
        value=(rho_max - rho_min),
        length=nx,
        position=nx - 1,
        sigma=6,
        offset=rho_min
    )
    
    # Initial conditions: Gaussian pulse in displacement
    x = np.arange(nx) * dx
    x_center = x[nx // 2]
    sigma_x = 2.0 * dx
    u0 = np.exp(-0.5 * ((x - x_center) / sigma_x) ** 2)
    v0 = np.zeros(nx)
    
    # Method 1: Classical leapfrog (ground truth)
    def dummy_source(x, t):
        return 0.0 * x
    
    # Add boundary padding
    u0_bc = np.zeros(nx + 2)
    u0_bc[1:-1] = u0
    u1_bc = u0_bc.copy()  # u1=u0 implies v0=0
    
    # Add boundary padding for medium properties
    mu_bc = np.zeros(nx + 2)
    mu_bc[1:nx+2] = mu_true
    mu_bc[0] = mu_true[0]
    mu_bc[-1] = mu_true[-1]
    
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    u_leapfrog_bc = evolve_1d_wave(
        u0=u0_bc,
        u1=u1_bc,
        dx=dx,
        dt=dt,
        mu=mu_bc,
        rho=rho_bc,
        source_func=dummy_source,
        steps=steps
    )
    
    # Extract interior points (remove BC padding)
    u_leapfrog = np.array([u[1:-1] for u in u_leapfrog_bc])
    
    # Method 2: Direct matrix exponential evolution
    # Use interior points only (no BC) to match the state vector size
    A = build_raw_A(mu_true, rho, dx, nx)
    u_expm, v_expm = evolve_via_matrix_exponential(u0, v0, A, dt, steps)
    
    # Compare trajectories
    l2_errors = []
    for t in range(steps + 1):
        u_leap = u_leapfrog[t]
        u_exp  = u_expm[t]
        
        diff = u_exp - u_leap
        err_norm = np.linalg.norm(diff)
        sig_norm = np.linalg.norm(u_leap) + 1e-30
        rel_err = err_norm / sig_norm
        l2_errors.append(rel_err)
    
    mean_error = np.mean(l2_errors)
    max_error  = np.max(l2_errors)
    
    # Report
    print(f"\n{'='*60}")
    print("RAW MATRIX A vs LEAPFROG VALIDATION")
    print(f"{'='*60}")
    print(f"Mean L2 relative error: {mean_error:.6f}")
    print(f"Max  L2 relative error: {max_error:.6f}")
    print(f"{'='*60}\n")
    
    # Assertion: mean error should be small (< 0.2)
    # This allows for finite-difference truncation error in leapfrog
    # but catches major bugs in K or A construction
    assert mean_error < 0.2, (
        f"Raw matrix A does not match leapfrog evolution!\n"
        f"Mean L2 error = {mean_error:.4f} (threshold: 0.2)\n"
        f"This indicates a bug in K or A construction."
    )
    
    print("[PASS] Raw matrix A correctly represents the elastic wave PDE")


if __name__ == "__main__":
    test_raw_A_matches_leapfrog()
