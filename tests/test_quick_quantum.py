"""
Simplified diagnostic: kenapa loss = 0 dari awal?
"""

import numpy as np
from src.optimization.objective import SeismicObjective
from src.distributions import raised_cosine, homogeneous


def quick_test():
    """Test sederhana: apakah forward simulation berbeda untuk mu berbeda?"""
    print("Quick test: Does quantum forward produce different results for different mu?")
    print("="*80)
    
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 5  # Very short
    
    # Simple initial conditions
    u0 = np.array([0.0, 0.3, 0.6, 1.0, 0.6, 0.3, 0.0])
    v0 = np.zeros(nx)
    
    # Two different models
    mu_true = np.array([1e10, 1.5e10, 2e10, 3e10, 2e10, 1.5e10, 1e10, 1e10])  # Heterogeneous
    mu_init = np.ones(nx + 1) * 1.5e10  # Homogeneous
    rho = np.ones(nx) * 2e3
    
    print(f"mu_true: {mu_true}")
    print(f"mu_init: {mu_init}")
    print()
    
    # Setup objective with quantum engine
    objective = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    
    # Compute reference fields dengan mu_true
    print("Computing reference fields with mu_true...")
    ref_fields = objective.compute_reference_fields(mu_true, rho, u0, v0)
    print(f"Reference: {len(ref_fields)} timesteps")
    print(f"ref_fields[0][1:-1] (IC): {ref_fields[0][1:-1]}")
    print(f"ref_fields[-1][1:-1] (final): {ref_fields[-1][1:-1]}")
    print()
    
    # Forward dengan mu_true (seharusnya match reference)
    print("Forward simulation with mu_true...")
    fields_true = objective.forward_simulate(mu_true, rho, u0, v0)
    print(f"fields_true[-1][1:-1]: {fields_true[-1][1:-1]}")
    print()
    
    # Forward dengan mu_init (seharusnya different dari reference)
    print("Forward simulation with mu_init...")
    fields_init = objective.forward_simulate(mu_init, rho, u0, v0)
    print(f"fields_init[-1][1:-1]: {fields_init[-1][1:-1]}")
    print()
    
    # Compute losses
    loss_true = objective.compute(mu_true, rho, u0, v0)
    loss_init = objective.compute(mu_init, rho, u0, v0)
    
    print("RESULTS:")
    print(f"  loss_true (should be ~0): {loss_true:.6e}")
    print(f"  loss_init (should be >0): {loss_init:.6e}")
    print()
    
    # Check if fields differ
    diff_ref_true = np.linalg.norm(ref_fields[-1][1:-1] - fields_true[-1][1:-1])
    diff_ref_init = np.linalg.norm(ref_fields[-1][1:-1] - fields_init[-1][1:-1])
    diff_true_init = np.linalg.norm(fields_true[-1][1:-1] - fields_init[-1][1:-1])
    
    print(f"  ||ref - fields_true||: {diff_ref_true:.6e}")
    print(f"  ||ref - fields_init||: {diff_ref_init:.6e}")
    print(f"  ||fields_true - fields_init||: {diff_true_init:.6e}")
    print()
    
    # Diagnosis
    if loss_true > 1e-6:
        print("[BUG] Reference and fields_true don't match!")
        print("  -> Quantum forward is non-deterministic or there's a bug in forward_simulate")
    elif loss_init < 1e-6:
        print("[BUG] Loss for different mu is still zero!")
        print("  -> Quantum forward produces same output regardless of mu")
    elif diff_true_init < 1e-10:
        print("[BUG] fields_true and fields_init are identical!")
        print("  -> Hamiltonian doesn't depend on mu OR evolution is broken")
    else:
        print("[OK] Forward simulation works correctly")


if __name__ == '__main__':
    quick_test()
