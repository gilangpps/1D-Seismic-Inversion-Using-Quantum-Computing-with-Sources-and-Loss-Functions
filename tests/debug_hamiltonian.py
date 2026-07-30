"""
Debug: Print Hamiltonian structure to see what's wrong
"""

import numpy as np
from src.hamiltonian import build_hamiltonian


def debug_hamiltonian():
    """Inspect Hamiltonian structure"""
    print("Debug: Hamiltonian structure")
    print("="*80)
    
    nx = 3
    dx = 63.0
    
    mu = np.ones(nx + 2) * 2e10
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    H, n_qubits, dim, phys_dim, _ = build_hamiltonian(mu, rho_bc, dx, nx)
    
    print(f"Dimensions:")
    print(f"  nx = {nx}")
    print(f"  phys_dim = {phys_dim} (should be 2*nx = {2*nx})")
    print(f"  dim = {dim} (2^{n_qubits})")
    print()
    
    print(f"Hamiltonian H ({dim}x{dim}):")
    print(f"  ||H||_F = {np.linalg.norm(H, 'fro'):.6e}")
    print(f"  max|H| = {np.max(np.abs(H)):.6e}")
    print()
    
    # Check structure: should be [[0, A], [A†, 0]]
    # Physical subspace: first phys_dim rows/cols
    # Dilated has 2*phys_dim before padding to 2^n
    actual_dilated_dim = 2 * phys_dim
    
    print(f"Expected dilated structure before padding:")
    print(f"  Upper-left (phys_dim x phys_dim): should be 0")
    print(f"  Upper-right (phys_dim x phys_dim): should be A")
    print(f"  Lower-left (phys_dim x phys_dim): should be A†")
    print(f"  Lower-right (phys_dim x phys_dim): should be 0")
    print()
    
    if actual_dilated_dim <= dim:
        H_ul = H[:phys_dim, :phys_dim]
        H_ur = H[:phys_dim, phys_dim:actual_dilated_dim]
        H_ll = H[phys_dim:actual_dilated_dim, :phys_dim]
        H_lr = H[phys_dim:actual_dilated_dim, phys_dim:actual_dilated_dim]
        
        print(f"Actual structure:")
        print(f"  Upper-left block (should be ~0):")
        print(f"    ||UL||_F = {np.linalg.norm(H_ul, 'fro'):.6e}")
        print(f"  Upper-right block (should be A):")
        print(f"    ||UR||_F = {np.linalg.norm(H_ur, 'fro'):.6e}")
        print(f"  Lower-left block (should be A†):")
        print(f"    ||LL||_F = {np.linalg.norm(H_ll, 'fro'):.6e}")
        print(f"  Lower-right block (should be ~0):")
        print(f"    ||LR||_F = {np.linalg.norm(H_lr, 'fro'):.6e}")
        print()
        
        # Check if UR and LL are non-zero and conjugate transpose
        if np.linalg.norm(H_ur, 'fro') < 1e-10:
            print("[BUG] Upper-right block (A) is zero!")
            print("  -> First-order system matrix A is not being constructed")
        
        if np.linalg.norm(H_ll, 'fro') < 1e-10:
            print("[BUG] Lower-left block (A†) is zero!")
        
        if np.linalg.norm(H_ur, 'fro') > 1e-10 and np.linalg.norm(H_ll, 'fro') > 1e-10:
            diff_hermitian = np.linalg.norm(H_ur - H_ll.conj().T, 'fro')
            print(f"  Hermiticity check: ||UR - LL†||_F = {diff_hermitian:.6e}")
            if diff_hermitian < 1e-10:
                print("    -> UR = LL† ✓ (correct Hermitian dilation)")
            else:
                print("    -> UR != LL† ✗ (Hermiticity broken)")
        
        # Print actual A matrix (UR block) for inspection
        print(f"\nA matrix (upper-right {phys_dim}x{phys_dim} block):")
        print(H_ur)
    
    # Test if H acts correctly on physical state
    print("\n" + "="*80)
    print("Test H action on physical state:")
    u = np.array([0.5, 1.0, 0.5])
    v = np.zeros(nx)
    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    u_w = sqrt_rho * u
    v_w = inv_sqrt_rho * v
    state = np.concatenate([u_w, v_w])
    
    psi = np.zeros(dim, dtype=complex)
    psi[:phys_dim] = state
    
    H_psi = H @ psi
    
    print(f"  |psi>[:phys_dim]: {psi[:phys_dim]}")
    print(f"  H|psi>[:phys_dim]: {H_psi[:phys_dim]}")
    print(f"  ||H|psi>||: {np.linalg.norm(H_psi):.6e}")
    
    if np.linalg.norm(H_psi) < 1e-10:
        print("\n[CRITICAL BUG] H|psi> = 0 for all physical states!")
        print("  -> Hamiltonian does nothing")
        print("  -> Check build_hamiltonian implementation")


if __name__ == '__main__':
    debug_hamiltonian()
