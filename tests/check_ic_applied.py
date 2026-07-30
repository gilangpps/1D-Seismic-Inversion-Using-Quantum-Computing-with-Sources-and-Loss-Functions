"""
Check: Apakah IC baru benar-benar di-generate?
"""
import numpy as np

nx = 7
dx = 63.0

# Simulate IC generation from main.py
x_grid_interior = np.arange(nx) * dx
x_norm = (x_grid_interior - x_grid_interior[0]) / (x_grid_interior[-1] - x_grid_interior[0] + 1e-30)

# Multi-mode superposition
u0 = (0.5 * np.sin(2 * np.pi * x_norm) + 
      0.3 * np.sin(4 * np.pi * x_norm) + 
      0.2 * np.sin(6 * np.pi * x_norm))

u0 = u0 / (np.max(np.abs(u0)) + 1e-30)

# Check values
print("Expected IC (multi-mode):")
print(f"  u0: {u0}")
print(f"  ||u0||: {np.linalg.norm(u0):.6f}")
print(f"  max(u0): {np.max(np.abs(u0)):.6f}")
print()

# What the logs show
print("From logs - config:")
print("  'u0': [0.32465..., 0.60653..., 0.88249..., 1.0, 0.88249..., 0.60653..., 0.32465...]")
print()
print("This looks like GAUSSIAN, not multi-mode!")
print()

# Check if Gaussian
x_center = x_grid_interior[nx // 2]
sigma_x = 2.0 * dx
u0_gaussian = np.exp(-0.5 * ((x_grid_interior - x_center) / sigma_x) ** 2)

print("Gaussian IC for comparison:")
print(f"  u0_gauss: {u0_gaussian}")
print()

if np.allclose(u0_gaussian, [0.32465246735834974, 0.6065306597126334, 0.8824969025845955, 1.0, 0.8824969025845955, 0.6065306597126334, 0.32465246735834974], atol=1e-6):
    print("[BUG CONFIRMED] main.py is still using OLD Gaussian IC!")
    print("The fix was not applied or code was not reloaded.")
