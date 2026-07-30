"""
Critical test: Why does U1@psi == U2@psi despite U1 != U2?

Hypothesis: (H1 - H2) @ psi ≈ 0 (psi in null space of difference)
"""
import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian

nx = 3
dx = 63.0
dt = 0.005

mu1 = np.ones(nx + 2) * 1e10
mu2 = np.ones(nx + 2) * 4e10
rho = np.ones(nx) * 2e3
rho_bc = np.zeros(nx + 2)
rho_bc[1:-1] = rho
rho_bc[0] = rho[0]
rho_bc[-1] = rho[-1]

print("Building Hamiltonians...")
H1, _, dim, phys_dim, _ = build_hamiltonian(mu1, rho_bc, dx, nx)
H2, _, dim, phys_dim, _ = build_hamiltonian(mu2, rho_bc, dx, nx)

print(f"Dimensions: phys_dim={phys_dim}, dim={dim}")
print()

# Initial state (NO mass-weighting in fixed version)
u0 = np.array([0.5, 1.0, 0.5])
v0 = np.zeros(nx)
state_vec = np.concatenate([u0, v0])

psi = np.zeros(dim, dtype=complex)
psi[:phys_dim] = state_vec
psi_norm = np.linalg.norm(psi)
psi = psi / psi_norm

print(f"Initial state:")
print(f"  psi[:phys_dim]: {psi[:phys_dim]}")
print(f"  ||psi||: {np.linalg.norm(psi):.6f}")
print()

# Test (H1 - H2) @ psi
H_diff = H1 - H2
H_diff_psi = H_diff @ psi

print(f"(H1 - H2) analysis:")
print(f"  ||H1 - H2||_F: {np.linalg.norm(H_diff, 'fro'):.3e}")
print(f"  ||(H1-H2)@psi||: {np.linalg.norm(H_diff_psi):.3e}")
print(f"  (H1-H2)@psi [:phys_dim]: {H_diff_psi[:phys_dim]}")
print()

if np.linalg.norm(H_diff_psi) < 1e-10:
    print("[ROOT CAUSE FOUND] (H1-H2)@psi ≈ 0!")
    print("  -> psi is in the null space of (H1-H2)")
    print("  -> This means H1 and H2 act identically on this specific state")
    print()
    
    # Analyze WHY
    print("Analyzing why...")
    # Since psi = [u, v, 0, 0, ...] with v=0
    # And H = [[0, A], [A†, 0]]
    # H @ psi = [[0, A], [A†, 0]] @ [u, 0, ...]
    #         = [A @ [0, ...], A† @ [u, ...]]
    
    print("  State structure: psi = [u, v=0, auxiliary=0]")
    print("  H @ psi = [[0, A], [A†, 0]] @ [u, 0, ...] = [0, A†@u]")
    print()
    print("  A = [[0, I], [K, 0]], so A† = [[0, K†], [I, 0]]")
    print("  A† @ [u, 0] = [[0, K†], [I, 0]] @ [u, 0] = [K†@u, I@u] = [K†@u, u]")
    print()
    print("  So (H1 - H2) @ psi = [0, (K1† - K2†)@u]")
    print("  If this is zero, then (K1 - K2) @ u = 0")
    print()
    
    # Extract K matrices from H
    # K is in A[nx:, :nx], and A is in H[:phys_dim, phys_dim:]
    A1 = H1[:phys_dim, phys_dim:2*phys_dim]
    A2 = H2[:phys_dim, phys_dim:2*phys_dim]
    K1 = A1[nx:phys_dim, :nx]
    K2 = A2[nx:phys_dim, :nx]
    
    print(f"K1 matrix:")
    print(K1)
    print()
    print(f"K2 matrix:")
    print(K2)
    print()
    
    K_diff = K1 - K2
    print(f"(K1 - K2):")
    print(K_diff)
    print()
    
    K_diff_u = K_diff @ u0
    print(f"(K1 - K2) @ u0: {K_diff_u}")
    print(f"||(K1-K2) @ u0||: {np.linalg.norm(K_diff_u):.3e}")
    
    if np.linalg.norm(K_diff_u) < 1e-10:
        print()
        print("[DEEPER ROOT CAUSE] u0 is an eigenvector of BOTH K1 and K2!")
        print("  OR: (K1 - K2) @ u0 = 0 for this specific u0")
        print()
        print("  This is the problem: smooth initial condition u0=[0.5, 1.0, 0.5]")
        print("  is a special eigenmode that doesn't see the difference in K!")
else:
    print("[OK] (H1-H2)@psi != 0, should produce different evolution")
