"""
Minimal test: Apakah Hamiltonian evolution mengubah state?
"""

import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian


def test_evolution_changes_state():
    """Test minimal: apakah exp(-iHt) mengubah state?"""
    print("Test: Does exp(-iHt) evolution change the state?")
    print("="*80)
    
    nx = 3  # Very small
    dx = 63.0
    dt = 0.005
    
    # Simple homogeneous model
    mu = np.ones(nx + 2) * 2e10
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    # Build Hamiltonian
    print(f"Building Hamiltonian for nx={nx}...")
    H_mat, n_qubits, dim, phys_dim, _ = build_hamiltonian(mu, rho_bc, dx, nx)
    print(f"  H shape: {H_mat.shape}")
    print(f"  Physical dim: {phys_dim} (2*nx = {2*nx})")
    print(f"  Dilated dim: {dim}")
    print()
    
    # Evolution operator
    U = expm(-1j * H_mat * dt)
    
    # Initial state: simple displacement
    u0 = np.array([0.5, 1.0, 0.5])
    v0 = np.zeros(nx)
    
    # Mass-weighted state (as in quantum_forward_simulate)
    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    
    u_weighted = sqrt_rho * u0
    v_weighted = inv_sqrt_rho * v0
    state_vec = np.concatenate([u_weighted, v_weighted])  # Length 2*nx = 6
    
    print(f"Initial state (physical, 2*nx={2*nx}):")
    print(f"  u0: {u0}")
    print(f"  v0: {v0}")
    print(f"  state_vec (weighted): {state_vec}")
    print()
    
    # Embed into dilated space
    psi = np.zeros(dim, dtype=complex)
    psi[:phys_dim] = state_vec
    psi_norm = np.linalg.norm(psi)
    if psi_norm > 1e-15:
        psi = psi / psi_norm
    
    print(f"Dilated state (dim={dim}):")
    print(f"  ||psi||: {np.linalg.norm(psi):.6f}")
    print(f"  psi[:phys_dim]: {psi[:phys_dim]}")
    print()
    
    # Apply evolution
    psi_evolved = U @ psi
    
    print(f"After evolution:")
    print(f"  ||psi_evolved||: {np.linalg.norm(psi_evolved):.6f} (should be ~1.0)")
    print(f"  psi_evolved[:phys_dim]: {psi_evolved[:phys_dim]}")
    print()
    
    # Back-project
    state_vec_evolved = psi_evolved[:phys_dim]
    u_weighted_evolved = np.real(state_vec_evolved[:nx]) * psi_norm
    u_decoded = u_weighted_evolved / (sqrt_rho + 1e-30)
    
    print(f"Decoded:")
    print(f"  u_decoded: {u_decoded}")
    print(f"  u0 (initial): {u0}")
    print()
    
    # Check if state changed
    diff = np.linalg.norm(u_decoded - u0)
    print(f"Change: ||u_decoded - u0|| = {diff:.6e}")
    
    if diff < 1e-10:
        print("[BUG FOUND] State tidak berubah! Evolution tidak bekerja.")
        print("  Possible causes:")
        print("    1. H = 0 (Hamiltonian is zero)")
        print("    2. U = I (Evolution operator is identity)")
        print("    3. Back-projection error")
        
        # Debug H
        H_norm = np.linalg.norm(H_mat, 'fro')
        print(f"\n  ||H||_F = {H_norm:.6e}")
        if H_norm < 1e-10:
            print("    -> H is essentially zero!")
        
        # Debug U
        U_minus_I = U - np.eye(dim)
        U_diff = np.linalg.norm(U_minus_I, 'fro')
        print(f"  ||U - I||_F = {U_diff:.6e}")
        if U_diff < 1e-10:
            print("    -> U is identity! exp(-iHt) didn't do anything")
    else:
        print("[OK] State changed, evolution works")


if __name__ == '__main__':
    test_evolution_changes_state()
