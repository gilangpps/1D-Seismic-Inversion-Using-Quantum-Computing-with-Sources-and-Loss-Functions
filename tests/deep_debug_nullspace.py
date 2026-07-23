import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian

nx = 3
dx = 63.0
dt = 0.005

# Multi-mode IC
x_norm = np.arange(nx) / (nx-1)
u0 = 0.5*np.sin(2*np.pi*x_norm) + 0.3*np.sin(4*np.pi*x_norm)
u0 = u0 / np.max(np.abs(u0))
v0 = 0.1 * 100 * np.cos(2*np.pi*x_norm)  # c~100 m/s

mu1 = np.ones(nx+2) * 1e10
mu2 = np.ones(nx+2) * 4e10
rho = np.ones(nx) * 2e3
rho_bc = np.zeros(nx+2); rho_bc[1:-1]=rho; rho_bc[0]=rho[0]; rho_bc[-1]=rho[-1]

print("IC:")
print(f"  u0: {u0}")
print(f"  v0: {v0}")
print()

H1, _, dim, phys_dim, _ = build_hamiltonian(mu1, rho_bc, dx, nx)
H2, _, dim, phys_dim, _ = build_hamiltonian(mu2, rho_bc, dx, nx)

print(f"Hamiltonians: dim={dim}, phys_dim={phys_dim}")
print(f"||H1-H2||_F = {np.linalg.norm(H1-H2, 'fro'):.3e}")

# Encode state
sqrt_rho = np.sqrt(rho)
inv_sqrt_rho = 1.0/sqrt_rho
u_w = sqrt_rho * u0
v_w = inv_sqrt_rho * v0
state_vec = np.concatenate([u_w, v_w])

psi = np.zeros(dim, dtype=complex)
psi[:phys_dim] = state_vec
psi_norm = np.linalg.norm(psi)
psi = psi / psi_norm

print(f"\nState: ||psi|| = {np.linalg.norm(psi):.6f}")
print(f"psi[:phys_dim] = {psi[:phys_dim]}")

# Check (H1-H2)@psi
H_diff = H1 - H2
H_diff_psi = H_diff @ psi

print(f"\n(H1-H2) @ psi:")
print(f"  ||(H1-H2)@psi|| = {np.linalg.norm(H_diff_psi):.6e}")
print(f"  (H1-H2)@psi[:phys_dim] = {H_diff_psi[:phys_dim]}")

if np.linalg.norm(H_diff_psi) < 1e-10:
    print("\n[BUG STILL EXISTS] (H1-H2)@psi = 0 even with new IC!")
    print("Checking components...")
    
    # Check if issue is in u or v component
    u_only = np.zeros(dim, dtype=complex)
    u_only[:nx] = u_w
    v_only = np.zeros(dim, dtype=complex)
    v_only[nx:phys_dim] = v_w
    
    H_diff_u = H_diff @ (u_only / np.linalg.norm(u_only))
    H_diff_v = H_diff @ (v_only / np.linalg.norm(v_only))
    
    print(f"  ||(H1-H2)@psi_u_only|| = {np.linalg.norm(H_diff_u):.6e}")
    print(f"  ||(H1-H2)@psi_v_only|| = {np.linalg.norm(H_diff_v):.6e}")
else:
    print("\n[GOOD] (H1-H2)@psi != 0, should work now")
