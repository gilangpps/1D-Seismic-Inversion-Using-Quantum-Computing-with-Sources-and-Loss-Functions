"""
Test: Apakah mass-weighting membatalkan dependensi pada mu?
"""

import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian


def test_mass_weighting_cancellation():
    """
    Hypothesis: Mass-weighting [sqrt(rho)*u, (1/sqrt(rho))*v] mungkin
    membatalkan dependensi evolution pada mu.
    
    Mari kita cek:
    1. Apakah H(mu1) dan H(mu2) benar-benar berbeda?
    2. Apakah exp(-iH1*dt) dan exp(-iH2*dt) berbeda?
    3. Apakah setelah mass-weighting, efeknya hilang?
    """
    print("Test: Does mass-weighting cancel mu-dependence?")
    print("="*80)
    
    nx = 3
    dx = 63.0
    dt = 0.005
    
    # Two different mu
    mu1 = np.ones(nx + 2) * 1e10
    mu2 = np.ones(nx + 2) * 4e10
    
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    # Build Hamiltonians
    print("Building Hamiltonians...")
    H1, _, dim, phys_dim, _ = build_hamiltonian(mu1, rho_bc, dx, nx)
    H2, _, dim, phys_dim, _ = build_hamiltonian(mu2, rho_bc, dx, nx)
    
    H_diff = np.linalg.norm(H1 - H2, 'fro')
    print(f"  ||H1 - H2||_F: {H_diff:.6e}")
    print(f"  -> Hamiltonians ARE different")
    print()
    
    # Evolution operators
    U1 = expm(-1j * H1 * dt)
    U2 = expm(-1j * H2 * dt)
    
    U_diff = np.linalg.norm(U1 - U2, 'fro')
    print(f"Evolution operators:")
    print(f"  ||U1 - U2||_F: {U_diff:.6e}")
    print(f"  -> Evolution operators ARE different")
    print()
    
    # Initial state
    u0 = np.array([0.5, 1.0, 0.5])
    v0 = np.zeros(nx)
    
    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    
    # Mass-weighted state
    u_weighted = sqrt_rho * u0
    v_weighted = inv_sqrt_rho * v0
    state_vec = np.concatenate([u_weighted, v_weighted])
    
    psi = np.zeros(dim, dtype=complex)
    psi[:phys_dim] = state_vec
    psi_norm = np.linalg.norm(psi)
    psi = psi / psi_norm
    
    print(f"Initial state:")
    print(f"  u0: {u0}")
    print(f"  mass-weighted psi[:phys_dim]: {psi[:phys_dim]}")
    print()
    
    # Evolve with U1
    psi1 = U1 @ psi
    state_vec1 = psi1[:phys_dim]
    u_weighted1 = np.real(state_vec1[:nx]) * psi_norm
    u1 = u_weighted1 / (sqrt_rho + 1e-30)
    
    # Evolve with U2
    psi2 = U2 @ psi
    state_vec2 = psi2[:phys_dim]
    u_weighted2 = np.real(state_vec2[:nx]) * psi_norm
    u2 = u_weighted2 / (sqrt_rho + 1e-30)
    
    print(f"After evolution:")
    print(f"  u1 (mu=1e10): {u1}")
    print(f"  u2 (mu=4e10): {u2}")
    print(f"  ||u1 - u2||: {np.linalg.norm(u1 - u2):.6e}")
    print()
    
    # Check intermediate steps
    print(f"Intermediate (before un-weighting):")
    print(f"  psi1[:phys_dim]: {psi1[:phys_dim]}")
    print(f"  psi2[:phys_dim]: {psi2[:phys_dim]}")
    print(f"  ||psi1 - psi2|| (in Hilbert space): {np.linalg.norm(psi1 - psi2):.6e}")
    print()
    
    if np.linalg.norm(psi1 - psi2) > 1e-10:
        print("[OK] States differ in Hilbert space after evolution")
    else:
        print("[BUG] States identical in Hilbert space!")
        return
    
    if np.linalg.norm(u1 - u2) < 1e-10:
        print("[BUG] But after decoding, u1 == u2!")
        print("  -> Mass un-weighting or something else cancels the difference")
        
        # Debug: check if the difference is only in velocity component
        v1 = np.real(state_vec1[nx:phys_dim]) * psi_norm * (sqrt_rho + 1e-30)
        v2 = np.real(state_vec2[nx:phys_dim]) * psi_norm * (sqrt_rho + 1e-30)
        print(f"\n  v1: {v1}")
        print(f"  v2: {v2}")
        print(f"  ||v1 - v2||: {np.linalg.norm(v1 - v2):.6e}")
    else:
        print("[OK] u1 and u2 differ after decoding")


if __name__ == '__main__':
    test_mass_weighting_cancellation()
