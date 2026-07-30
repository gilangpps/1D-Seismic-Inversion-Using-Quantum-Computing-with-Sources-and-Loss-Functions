import numpy as np
from scipy.linalg import expm

nx = 7
dx = 63.0
dt = 0.004

# Build K
mu = np.ones(nx+2) * 2e10
rho = np.ones(nx) * 2e3
rho_bc = np.zeros(nx+2)
rho_bc[1:-1] = rho
rho_bc[0] = rho[0]
rho_bc[-1] = rho[-1]

K = np.zeros((nx, nx))
for i in range(nx):
    mu_r = 0.5 * (mu[min(i+1, len(mu)-1)] + mu[min(i+2, len(mu)-1)])
    mu_l = 0.5 * (mu[max(i, 0)] + mu[min(i+1, len(mu)-1)])
    rho_i = rho_bc[i+1]
    K[i,i] = -(mu_r + mu_l) / (rho_i * dx * dx)
    if i+1 < nx:
        K[i,i+1] = mu_r / (rho_i * dx * dx)
    if i > 0:
        K[i,i-1] = mu_l / (rho_i * dx * dx)

# A matrix
A = np.zeros((2*nx, 2*nx))
A[:nx, nx:] = np.eye(nx)
A[nx:, :nx] = K

# Check eigenvalues
eigvals = np.linalg.eigvals(A)
print(f"A eigenvalues:")
print(eigvals)
print(f"\nMax real part: {np.max(np.real(eigvals)):.3e}")
print(f"Min real part: {np.min(np.real(eigvals)):.3e}")

if np.max(np.real(eigvals)) > 0:
    print("\n[PROBLEM] A has positive real eigenvalues!")
    print("exp(A*t) will GROW exponentially, not oscillate.")
    print("This explains norm growth: 1.46 → 13.12")

# Test evolution
U = expm(A * dt)
print(f"\nU = exp(A*dt) spectral radius: {np.max(np.abs(np.linalg.eigvals(U))):.6f}")
print("Should be ~1.0 for stable evolution")
