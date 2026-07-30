import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian

nx = 3
mu = np.ones(nx+2) * 2e10
rho = np.ones(nx) * 2e3
rho_bc = np.zeros(nx+2); rho_bc[1:-1]=rho; rho_bc[0]=rho[0]; rho_bc[-1]=rho[-1]
dx = 63.0
dt = 0.005

H, _, dim, phys_dim, _ = build_hamiltonian(mu, rho_bc, dx, nx)

print(f"Dimensions: nx={nx}, phys_dim={phys_dim} (2*nx={2*nx}), dim={dim}")
print(f"Dilated_dim before padding: {2*phys_dim} (4*nx={4*nx})")
print()

# Check block structure
dilated_dim = 2 * phys_dim  # 12 for nx=3

if dilated_dim <= dim:
    H_UL = H[:phys_dim, :phys_dim]
    H_UR = H[:phys_dim, phys_dim:dilated_dim]
    H_LL = H[phys_dim:dilated_dim, :phys_dim]
    H_LR = H[phys_dim:dilated_dim, phys_dim:dilated_dim]
    
    print("Block norms:")
    print(f"  UL (should be ~0): {np.linalg.norm(H_UL, 'fro'):.3e}")
    print(f"  UR (should be A):  {np.linalg.norm(H_UR, 'fro'):.3e}")
    print(f"  LL (should be A†): {np.linalg.norm(H_LL, 'fro'):.3e}")
    print(f"  LR (should be ~0): {np.linalg.norm(H_LR, 'fro'):.3e}")
    print()

# Test evolution on pure physical state
u0 = np.array([0, -0.05, -1.0])
v0 = np.array([10, -10, 10])
sqrt_rho = np.sqrt(rho)
inv_sqrt_rho = 1.0/sqrt_rho

u_w = sqrt_rho * u0
v_w = inv_sqrt_rho * v0
state_vec = np.concatenate([u_w, v_w])

# Embed: [physical, auxiliary]
psi = np.zeros(dim, dtype=complex)
psi[:phys_dim] = state_vec
psi_norm = np.linalg.norm(psi)
psi = psi / psi_norm

print(f"Initial state:")
print(f"  psi[:phys_dim]: {psi[:phys_dim]}")
print(f"  psi[phys_dim:dilated_dim]: {psi[phys_dim:dilated_dim]}")
print(f"  ||psi_physical||: {np.linalg.norm(psi[:phys_dim]):.6f}")
print(f"  ||psi_auxiliary||: {np.linalg.norm(psi[phys_dim:dilated_dim]):.6f}")
print()

# Evolve
U = expm(-1j * H * dt)
psi_evolved = U @ psi

print(f"After evolution:")
print(f"  psi[:phys_dim]: {psi_evolved[:phys_dim]}")
print(f"  psi[phys_dim:dilated_dim]: {psi_evolved[phys_dim:dilated_dim]}")
print(f"  ||psi_physical||: {np.linalg.norm(psi_evolved[:phys_dim]):.6f}")
print(f"  ||psi_auxiliary||: {np.linalg.norm(psi_evolved[phys_dim:dilated_dim]):.6f}")
print()

# Check if leakage to auxiliary
if np.linalg.norm(psi_evolved[phys_dim:dilated_dim]) > 0.1:
    print("[WARNING] Significant leakage to auxiliary space!")
    print("  This causes loss of physical information.")
