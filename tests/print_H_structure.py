"""
Print Hamiltonian matrix untuk inspect struktur
"""
import numpy as np
from src.hamiltonian import build_hamiltonian

nx = 3
dx = 63.0
mu = np.ones(nx + 2) * 2e10
rho = np.ones(nx) * 2e3
rho_bc = np.zeros(nx + 2)
rho_bc[1:-1] = rho
rho_bc[0] = rho[0]
rho_bc[-1] = rho[-1]

H, n_qubits, dim, phys_dim, _ = build_hamiltonian(mu, rho_bc, dx, nx)

print(f"nx = {nx}")
print(f"phys_dim = {phys_dim} (should be 2*nx = {2*nx})")
print(f"dilated_dim = {2*phys_dim} (should be 4*nx = {4*nx})")
print(f"padded_dim = {dim} (2^{n_qubits})")
print()

print("Hamiltonian structure (first 12x12 for visibility):")
print("Rows 0-11, Cols 0-11:")
print(np.real(H[:12, :12]))
print()

# Check specific blocks
print("Block analysis:")
print(f"  Block [0:6, 0:6] (upper-left, should be ~0):")
print(f"    ||·||_F = {np.linalg.norm(H[:6, :6], 'fro'):.3e}")
print(f"  Block [0:6, 6:12] (upper-right, should be A):")
print(f"    ||·||_F = {np.linalg.norm(H[:6, 6:12], 'fro'):.3e}")
print(f"  Block [6:12, 0:6] (lower-left, should be A^T):")
print(f"    ||·||_F = {np.linalg.norm(H[6:12, :6], 'fro'):.3e}")
print(f"  Block [6:12, 6:12] (lower-right, should be ~0):")
print(f"    ||·||_F = {np.linalg.norm(H[6:12, 6:12], 'fro'):.3e}")
