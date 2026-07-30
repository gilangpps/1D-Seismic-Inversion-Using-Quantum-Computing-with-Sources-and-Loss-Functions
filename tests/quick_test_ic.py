import numpy as np
from src.optimization.objective import SeismicObjective

nx = 7
dx = 63.0
dt = 0.005
steps = 5

# Generate NEW IC (same as main.py)
x_grid = np.arange(nx) * dx
x_norm = (x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0] + 1e-30)
u0 = 0.5*np.sin(2*np.pi*x_norm) + 0.3*np.sin(4*np.pi*x_norm) + 0.2*np.sin(6*np.pi*x_norm)
u0 = u0 / (np.max(np.abs(u0)) + 1e-30)

mu_true = np.ones(nx+1) * 2e10
rho = np.ones(nx) * 2e3
c = np.sqrt(np.mean(mu_true) / np.mean(rho))
v0 = 0.1 * c * np.cos(2*np.pi*x_norm)

print(f"NEW IC: u0={u0}")
print(f"Has negative values: {np.any(u0 < 0)}")
print()

# Test quantum forward
mu1 = np.ones(nx+1) * 1e10
mu2 = np.ones(nx+1) * 4e10

obj = SeismicObjective(nx=nx, dx=dx, dt=dt, steps=steps, engine='quantum')

print("Running with mu1...")
fields1 = obj.quantum_forward_simulate(mu1, rho, u0, v0)

print("Running with mu2...")
fields2 = obj.quantum_forward_simulate(mu2, rho, u0, v0)

diff = np.linalg.norm(fields1[-1][1:-1] - fields2[-1][1:-1])
print(f"\n||fields1[-1] - fields2[-1]|| = {diff:.6e}")
print("Expected: >1e-6 (different)")
print("Result:", "PASS" if diff > 1e-6 else "FAIL")
