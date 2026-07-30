"""Quick validation test for Hermitian dilation fix."""
from src.experiment.validate_hamiltonian import run_hamiltonian_validation
from src.distributions import raised_cosine
import numpy as np

# Setup (same as main.py)
nx = 7
mu = raised_cosine(1e10, nx+1, 4, 2, 0.8e10)
rho = raised_cosine(2e3, nx, nx-1, 6, 2e3)

# Initial condition: Gaussian pulse
x = np.arange(nx) * 63
u0 = np.exp(-0.5 * ((x - x[nx//2]) / 126)**2)
v0 = np.zeros(nx)

# Run validation
res = run_hamiltonian_validation(
    mu, rho, u0, v0,
    dx=63.0, dt=0.005, steps=40, nx=nx
)

print(f"\n{'='*60}")
print("VALIDATION RESULT")
print(f"{'='*60}")
print(f"Mean overlap:  {res['mean_overlap']:.6f}")
print(f"Mean L2 error: {res['mean_l2_error']:.6f}")
print(f"{'='*60}\n")

# Check thresholds
if res['mean_overlap'] > 0.95:
    print("✓ VALIDATION PASSED (overlap > 0.95)")
elif res['mean_overlap'] > 0.85:
    print("⚠ VALIDATION PARTIAL (overlap > 0.85)")
else:
    print("✗ VALIDATION FAILED (overlap < 0.85)")
