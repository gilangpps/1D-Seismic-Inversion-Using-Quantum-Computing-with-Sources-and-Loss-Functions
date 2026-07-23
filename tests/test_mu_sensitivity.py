"""
Test: Sensitivity of quantum forward to mu changes
"""

import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian


def test_mu_sensitivity():
    """Test: Seberapa sensitif evolution terhadap perubahan mu?"""
    print("Test: Sensitivity of quantum evolution to mu changes")
    print("="*80)
    
    nx = 3
    dx = 63.0
    dt = 0.005
    steps = 10
    
    # Two models: soft vs stiff
    mu_soft = np.ones(nx + 2) * 1e10
    mu_stiff = np.ones(nx + 2) * 4e10
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    # Initial conditions
    u0 = np.array([0.5, 1.0, 0.5])
    v0 = np.zeros(nx)
    
    print(f"Models:")
    print(f"  mu_soft: {mu_soft[0]:.2e} Pa (homogeneous)")
    print(f"  mu_stiff: {mu_stiff[0]:.2e} Pa (homogeneous)")
    print(f"  Ratio: {mu_stiff[0]/mu_soft[0]:.1f}x")
    print()
    
    # Function to evolve
    def evolve(mu):
        H_mat, _, dim, phys_dim, _ = build_hamiltonian(mu, rho_bc, dx, nx)
        U = expm(-1j * H_mat * dt)
        
        sqrt_rho = np.sqrt(rho)
        inv_sqrt_rho = 1.0 / sqrt_rho
        
        u_current = u0.copy()
        v_current = v0.copy()
        
        for step in range(steps):
            u_weighted = sqrt_rho * u_current
            v_weighted = inv_sqrt_rho * v_current
            state_vec = np.concatenate([u_weighted, v_weighted])
            
            psi = np.zeros(dim, dtype=complex)
            psi[:phys_dim] = state_vec
            psi_norm = np.linalg.norm(psi)
            if psi_norm > 1e-15:
                psi = psi / psi_norm
            else:
                psi_norm = 1.0
            
            psi_evolved = U @ psi
            state_vec_evolved = psi_evolved[:phys_dim]
            
            u_weighted_evolved = np.real(state_vec_evolved[:nx]) * psi_norm
            u_current = u_weighted_evolved / (sqrt_rho + 1e-30)
            
            v_weighted_evolved = np.real(state_vec_evolved[nx:phys_dim]) * psi_norm
            v_current = v_weighted_evolved * (sqrt_rho + 1e-30)
        
        return u_current
    
    print(f"Evolving for {steps} steps with dt={dt}s (total time = {steps*dt}s)...")
    u_soft = evolve(mu_soft)
    u_stiff = evolve(mu_stiff)
    
    print(f"\nResults:")
    print(f"  u0 (initial): {u0}")
    print(f"  u_soft  (final): {u_soft}")
    print(f"  u_stiff (final): {u_stiff}")
    print()
    
    diff_soft_ic = np.linalg.norm(u_soft - u0)
    diff_stiff_ic = np.linalg.norm(u_stiff - u0)
    diff_soft_stiff = np.linalg.norm(u_soft - u_stiff)
    
    print(f"Changes:")
    print(f"  ||u_soft - u0||: {diff_soft_ic:.6e}")
    print(f"  ||u_stiff - u0||: {diff_stiff_ic:.6e}")
    print(f"  ||u_soft - u_stiff||: {diff_soft_stiff:.6e}")
    print()
    
    # Relative sensitivity
    rel_change_soft = diff_soft_ic / (np.linalg.norm(u0) + 1e-30)
    rel_change_stiff = diff_stiff_ic / (np.linalg.norm(u0) + 1e-30)
    rel_diff = diff_soft_stiff / (np.linalg.norm(u0) + 1e-30)
    
    print(f"Relative changes (normalized by ||u0||):")
    print(f"  soft:  {rel_change_soft:.6e}")
    print(f"  stiff: {rel_change_stiff:.6e}")
    print(f"  diff:  {rel_diff:.6e}")
    print()
    
    # Diagnosis
    if diff_soft_stiff < 1e-10:
        print("[BUG] Trajectories identical for 4x different mu!")
        print("  -> Evolution doesn't depend on mu")
    elif diff_soft_stiff < 1e-6:
        print("[WARNING] Very small sensitivity to mu changes")
        print(f"  -> Only {rel_diff:.2e} relative difference for 4x mu change")
        print(f"  -> This will cause near-zero gradients in optimization")
    else:
        print("[OK] Trajectories differ, sensitivity exists")
    
    # Check if evolution is too weak
    if rel_change_soft < 1e-4:
        print("\n[WARNING] Evolution is very weak!")
        print(f"  -> After {steps} steps, state only changed by {rel_change_soft:.2e}")
        print(f"  -> May need: longer time, larger dt, or stronger initial conditions")


if __name__ == '__main__':
    test_mu_sensitivity()
