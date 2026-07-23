"""
Test IC Fix: Verify new initial conditions produce mu-dependent quantum evolution
"""
import numpy as np
from src.optimization.objective import SeismicObjective

def test_ic_fix():
    """
    Test that multi-mode IC produces different trajectories for different mu,
    unlike the old Gaussian IC which was an eigenmode.
    """
    print("="*80)
    print("TEST: IC Fix Verification")
    print("="*80)
    print()
    
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 10
    
    # Two different models
    mu1 = np.ones(nx + 1) * 1e10  # Soft
    mu2 = np.ones(nx + 1) * 4e10  # Stiff (4x difference)
    rho = np.ones(nx) * 2e3
    
    print(f"Test parameters:")
    print(f"  nx={nx}, dx={dx}, dt={dt}, steps={steps}")
    print(f"  mu1={mu1[0]:.2e} Pa, mu2={mu2[0]:.2e} Pa (ratio={mu2[0]/mu1[0]:.1f}x)")
    print()
    
    # === OLD IC (BUGGY) ===
    print("Test 1: OLD Gaussian IC (eigenmode bug)")
    print("-"*80)
    
    x_grid = np.arange(nx) * dx
    x_center = x_grid[nx // 2]
    sigma_x = 2.0 * dx
    u0_old = np.exp(-0.5 * ((x_grid - x_center) / sigma_x) ** 2)
    v0_old = np.zeros(nx)
    
    print(f"  u0 (Gaussian): {u0_old}")
    print(f"  v0: {v0_old}")
    print()
    
    obj_old = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    
    print("  Running quantum forward with mu1...")
    fields1_old = obj_old.quantum_forward_simulate(mu1, rho, u0_old, v0_old)
    u1_old_final = fields1_old[-1][1:-1]
    
    print("  Running quantum forward with mu2...")
    fields2_old = obj_old.quantum_forward_simulate(mu2, rho, u0_old, v0_old)
    u2_old_final = fields2_old[-1][1:-1]
    
    diff_old = np.linalg.norm(u1_old_final - u2_old_final)
    rel_diff_old = diff_old / (np.linalg.norm(u1_old_final) + 1e-30)
    
    print(f"  u1_final: {u1_old_final}")
    print(f"  u2_final: {u2_old_final}")
    print(f"  ||u1 - u2||: {diff_old:.6e}")
    print(f"  Relative: {rel_diff_old:.6e}")
    
    if diff_old < 1e-8:
        print("  [FAIL] OLD IC produces identical trajectories (as expected - this is the bug)")
    else:
        print("  [PASS] OLD IC produces different trajectories (unexpected!)")
    print()
    
    # === NEW IC (FIXED) ===
    print("Test 2: NEW Multi-mode IC (eigenmode fix)")
    print("-"*80)
    
    x_norm = (x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0] + 1e-30)
    u0_new = (0.5 * np.sin(2 * np.pi * x_norm) + 
              0.3 * np.sin(4 * np.pi * x_norm) + 
              0.2 * np.sin(6 * np.pi * x_norm))
    u0_new = u0_new / (np.max(np.abs(u0_new)) + 1e-30)
    
    c_typical = np.sqrt(np.mean(mu1) / np.mean(rho))
    v0_new = 0.1 * c_typical * np.cos(2 * np.pi * x_norm)
    
    print(f"  u0 (multi-mode): {u0_new}")
    print(f"  v0 (non-zero): {v0_new}")
    print()
    
    obj_new = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    
    print("  Running quantum forward with mu1...")
    fields1_new = obj_new.quantum_forward_simulate(mu1, rho, u0_new, v0_new)
    u1_new_final = fields1_new[-1][1:-1]
    
    print("  Running quantum forward with mu2...")
    fields2_new = obj_new.quantum_forward_simulate(mu2, rho, u0_new, v0_new)
    u2_new_final = fields2_new[-1][1:-1]
    
    diff_new = np.linalg.norm(u1_new_final - u2_new_final)
    rel_diff_new = diff_new / (np.linalg.norm(u1_new_final) + 1e-30)
    
    print(f"  u1_final: {u1_new_final}")
    print(f"  u2_final: {u2_new_final}")
    print(f"  ||u1 - u2||: {diff_new:.6e}")
    print(f"  Relative: {rel_diff_new:.6e}")
    
    if diff_new > 1e-6:
        print("  [PASS] NEW IC produces different trajectories!")
    else:
        print("  [FAIL] NEW IC still produces identical trajectories")
    print()
    
    # === COMPARISON ===
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"OLD IC (Gaussian):    ||diff|| = {diff_old:.6e} (relative: {rel_diff_old:.6e})")
    print(f"NEW IC (Multi-mode):  ||diff|| = {diff_new:.6e} (relative: {rel_diff_new:.6e})")
    print()
    
    improvement_factor = diff_new / (diff_old + 1e-30)
    print(f"Improvement factor: {improvement_factor:.2e}x")
    print()
    
    # Acceptance criteria
    success = diff_new > 1e-6
    if success:
        print("[SUCCESS] Fix verified! Multi-mode IC creates mu-dependent evolution.")
    else:
        print("[FAILURE] Fix did not resolve the bug. Further investigation needed.")
    
    return success


if __name__ == '__main__':
    success = test_ic_fix()
    exit(0 if success else 1)
