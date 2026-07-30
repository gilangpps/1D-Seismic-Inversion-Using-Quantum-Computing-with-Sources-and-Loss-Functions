"""
Test: Apakah initial state adalah eigenstate dari H?
Jika ya, maka exp(-iHt)|psi> = exp(-iEt)|psi> tidak bergantung pada struktur H.
"""

import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian


def test_eigenstate_hypothesis():
    """Test: Apakah IC adalah eigenstate dari semua Hamiltonian?"""
    print("Test: Is initial state an eigenstate?")
    print("="*80)
    
    nx = 3
    dx = 63.0
    
    mu1 = np.ones(nx + 2) * 1e10
    mu2 = np.ones(nx + 2) * 4e10
    
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    H1, _, dim, phys_dim, _ = build_hamiltonian(mu1, rho_bc, dx, nx)
    H2, _, dim, phys_dim, _ = build_hamiltonian(mu2, rho_bc, dx, nx)
    
    # Initial state: v=0 (pure displacement, no velocity)
    u0 = np.array([0.5, 1.0, 0.5])
    v0 = np.zeros(nx)
    
    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    
    u_weighted = sqrt_rho * u0
    v_weighted = inv_sqrt_rho * v0  # This is zero!
    state_vec = np.concatenate([u_weighted, v_weighted])
    
    psi = np.zeros(dim, dtype=complex)
    psi[:phys_dim] = state_vec
    psi_norm = np.linalg.norm(psi)
    psi = psi / psi_norm
    
    print(f"Initial state structure:")
    print(f"  u0: {u0}")
    print(f"  v0: {v0}")
    print(f"  psi[:phys_dim]: {psi[:phys_dim]}")
    print(f"  -> Note: v0 = 0, so second half is all zeros")
    print()
    
    # Check if H|psi> = E|psi>
    H1_psi = H1 @ psi
    H2_psi = H2 @ psi
    
    print(f"H|psi> products:")
    print(f"  H1|psi>[:phys_dim]: {H1_psi[:phys_dim]}")
    print(f"  H2|psi>[:phys_dim]: {H2_psi[:phys_dim]}")
    print()
    
    # Check if H|psi> is parallel to |psi> (eigenstate condition)
    # If H|psi> = lambda |psi>, then H|psi> and |psi> are parallel
    def check_parallel(v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1e-15 or v2_norm < 1e-15:
            return False, 0.0
        dot = np.vdot(v1, v2)
        cos_angle = abs(dot) / (v1_norm * v2_norm)
        return cos_angle > 0.999, float(cos_angle)
    
    is_parallel1, cos1 = check_parallel(H1_psi, psi)
    is_parallel2, cos2 = check_parallel(H2_psi, psi)
    
    print(f"Is eigenstate?")
    print(f"  H1|psi> parallel to |psi>? {is_parallel1} (cos={cos1:.6f})")
    print(f"  H2|psi> parallel to |psi>? {is_parallel2} (cos={cos2:.6f})")
    print()
    
    if is_parallel1 or is_parallel2:
        print("[BUG FOUND] Initial state is (approximately) an eigenstate!")
        print("  -> This explains why evolution doesn't change it")
        print("  -> Need to use IC with BOTH u0 and v0 non-zero")
    
    # ── Try with non-zero velocity ──
    print("\n" + "="*80)
    print("Retry with non-zero velocity:")
    print("="*80)
    
    u0_v2 = np.array([0.5, 1.0, 0.5])
    v0_v2 = np.array([0.1, 0.2, 0.1])  # Non-zero velocity
    
    u_weighted_v2 = sqrt_rho * u0_v2
    v_weighted_v2 = inv_sqrt_rho * v0_v2
    state_vec_v2 = np.concatenate([u_weighted_v2, v_weighted_v2])
    
    psi_v2 = np.zeros(dim, dtype=complex)
    psi_v2[:phys_dim] = state_vec_v2
    psi_v2_norm = np.linalg.norm(psi_v2)
    psi_v2 = psi_v2 / psi_v2_norm
    
    print(f"New initial state:")
    print(f"  u0: {u0_v2}")
    print(f"  v0: {v0_v2}")
    print(f"  psi[:phys_dim]: {psi_v2[:phys_dim]}")
    print()
    
    # Evolve with both Hamiltonians
    dt = 0.005
    U1_v2 = expm(-1j * H1 * dt)
    U2_v2 = expm(-1j * H2 * dt)
    
    psi1_v2 = U1_v2 @ psi_v2
    psi2_v2 = U2_v2 @ psi_v2
    
    state_vec1_v2 = psi1_v2[:phys_dim]
    state_vec2_v2 = psi2_v2[:phys_dim]
    
    u1_v2 = np.real(state_vec1_v2[:nx]) * psi_v2_norm / (sqrt_rho + 1e-30)
    u2_v2 = np.real(state_vec2_v2[:nx]) * psi_v2_norm / (sqrt_rho + 1e-30)
    
    print(f"After evolution with non-zero velocity:")
    print(f"  u1: {u1_v2}")
    print(f"  u2: {u2_v2}")
    print(f"  ||u1 - u2||: {np.linalg.norm(u1_v2 - u2_v2):.6e}")
    
    if np.linalg.norm(u1_v2 - u2_v2) > 1e-10:
        print("\n[OK] With non-zero velocity, trajectories DO differ!")
        print("  -> Bug was: v0=0 makes state close to eigenstate")
    else:
        print("\n[STILL BUG] Even with non-zero velocity, trajectories identical")


if __name__ == '__main__':
    test_eigenstate_hypothesis()
