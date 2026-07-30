"""
Test hypothesis: Bug ada di normalisasi yang membatalkan efek evolution
"""

import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian


def test_normalization_bug():
    """Test: Apakah normalization membatalkan efek evolution?"""
    print("Test: Normalization bug hypothesis")
    print("="*80)
    
    nx = 3
    dx = 63.0
    dt = 0.005
    
    mu = np.ones(nx + 2) * 2e10
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    H_mat, _, dim, phys_dim, _ = build_hamiltonian(mu, rho_bc, dx, nx)
    U = expm(-1j * H_mat * dt)
    
    u0 = np.array([0.5, 1.0, 0.5])
    v0 = np.zeros(nx)
    
    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    
    # ── Method 1: Current (buggy) implementation ──
    print("Method 1: Current implementation (normalize, evolve, scale back)")
    u_weighted = sqrt_rho * u0
    v_weighted = inv_sqrt_rho * v0
    state_vec = np.concatenate([u_weighted, v_weighted])
    
    psi = np.zeros(dim, dtype=complex)
    psi[:phys_dim] = state_vec
    psi_norm_OLD = np.linalg.norm(psi)  # Save OLD norm
    psi = psi / psi_norm_OLD  # Normalize
    
    print(f"  psi_norm (before evolution): {psi_norm_OLD:.6f}")
    print(f"  ||psi|| (after normalize): {np.linalg.norm(psi):.6f}")
    
    psi_evolved = U @ psi
    print(f"  ||psi_evolved||: {np.linalg.norm(psi_evolved):.6f}")
    
    state_vec_evolved = psi_evolved[:phys_dim]
    u_weighted_evolved = np.real(state_vec_evolved[:nx]) * psi_norm_OLD  # Scale by OLD norm!
    u_decoded_buggy = u_weighted_evolved / (sqrt_rho + 1e-30)
    
    print(f"  u_decoded (buggy): {u_decoded_buggy}")
    print(f"  Change from u0: {np.linalg.norm(u_decoded_buggy - u0):.6e}")
    print()
    
    # ── Method 2: Fixed - don't renormalize ──
    print("Method 2: Fixed - keep unnormalized (physical norm)")
    psi_unnorm = np.zeros(dim, dtype=complex)
    psi_unnorm[:phys_dim] = state_vec
    psi_unnorm_norm = np.linalg.norm(psi_unnorm)
    
    print(f"  ||psi_unnorm||: {psi_unnorm_norm:.6f} (physical magnitude)")
    
    psi_evolved_unnorm = U @ psi_unnorm
    print(f"  ||psi_evolved_unnorm||: {np.linalg.norm(psi_evolved_unnorm):.6f}")
    
    state_vec_evolved_unnorm = psi_evolved_unnorm[:phys_dim]
    u_weighted_evolved_unnorm = np.real(state_vec_evolved_unnorm[:nx])
    u_decoded_fixed = u_weighted_evolved_unnorm / (sqrt_rho + 1e-30)
    
    print(f"  u_decoded (fixed): {u_decoded_fixed}")
    print(f"  Change from u0: {np.linalg.norm(u_decoded_fixed - u0):.6e}")
    print()
    
    # Compare
    diff_methods = np.linalg.norm(u_decoded_buggy - u_decoded_fixed)
    print(f"Difference between methods: {diff_methods:.6e}")
    
    if diff_methods < 1e-10:
        print("  -> Methods give same result (normalization doesn't matter)")
    else:
        print("  -> Methods differ! Normalization is affecting the result")
    
    # ── Method 3: Alternative fix - normalize but use NEW norm ──
    print("\nMethod 3: Alternative - normalize and track NEW norm")
    psi_norm3 = np.zeros(dim, dtype=complex)
    psi_norm3[:phys_dim] = state_vec
    norm_before = np.linalg.norm(psi_norm3)
    psi_norm3 = psi_norm3 / norm_before
    
    psi_evolved3 = U @ psi_norm3
    norm_after = np.linalg.norm(psi_evolved3)  # Should be ~1.0 (unitary)
    
    print(f"  norm before: {norm_before:.6f}")
    print(f"  norm after: {norm_after:.6f}")
    
    # Scale back using BOTH norms
    state_vec_evolved3 = psi_evolved3[:phys_dim]
    u_weighted_evolved3 = np.real(state_vec_evolved3[:nx]) * norm_before  # Scale by input norm
    u_decoded3 = u_weighted_evolved3 / (sqrt_rho + 1e-30)
    
    print(f"  u_decoded (method 3): {u_decoded3}")
    print(f"  Change from u0: {np.linalg.norm(u_decoded3 - u0):.6e}")
    
    return u_decoded_buggy, u_decoded_fixed, u_decoded3


if __name__ == '__main__':
    test_normalization_bug()
