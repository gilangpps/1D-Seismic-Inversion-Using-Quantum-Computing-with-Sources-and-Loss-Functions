"""
Test dengan import fresh - pastikan menggunakan code terbaru
"""
import sys
import importlib

# Force reload modules
if 'src.optimization.objective' in sys.modules:
    importlib.reload(sys.modules['src.optimization.objective'])
if 'src.hamiltonian' in sys.modules:
    importlib.reload(sys.modules['src.hamiltonian'])

import numpy as np
from src.optimization.objective import SeismicObjective

print("Testing FIXED quantum forward (no mass-weighting)...")
print("="*80)

nx = 3
dx = 63.0
dt = 0.005
steps = 5

u0 = np.array([0.5, 1.0, 0.5])
v0 = np.zeros(nx)

mu_soft = np.ones(nx + 1) * 1e10
mu_stiff = np.ones(nx + 1) * 4e10
rho = np.ones(nx) * 2e3

print(f"Models:")
print(f"  mu_soft:  {mu_soft[0]:.2e} Pa")
print(f"  mu_stiff: {mu_stiff[0]:.2e} Pa")
print()

obj = SeismicObjective(
    nx=nx, dx=dx, dt=dt, steps=steps,
    measure_every=4, shots=1000, bc='dirichlet',
    seed=42, source_func=None, engine='quantum'
)

print("Forward with mu_soft...")
fields_soft = obj.quantum_forward_simulate(mu_soft, rho, u0, v0)
u_soft_final = fields_soft[-1][1:-1]

print("Forward with mu_stiff...")
fields_stiff = obj.quantum_forward_simulate(mu_stiff, rho, u0, v0)
u_stiff_final = fields_stiff[-1][1:-1]

print()
print("Results:")
print(f"  u_soft:  {u_soft_final}")
print(f"  u_stiff: {u_stiff_final}")
print(f"  ||u_soft - u_stiff||: {np.linalg.norm(u_soft_final - u_stiff_final):.6e}")

if np.linalg.norm(u_soft_final - u_stiff_final) < 1e-10:
    print("\n[STILL BUGGY] Trajectories identical!")
else:
    print("\n[FIXED] Trajectories differ!")
