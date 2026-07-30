# PATCH: Fix Quantum Inversion Zero-Gradient Bug

**Date**: 2026-07-16  
**Issue**: Quantum forward simulation produces identical outputs for different mu values  
**Root Cause**: Initial conditions form eigenmode in null space of (H₁ - H₂)  
**Severity**: CRITICAL (P0) - Blocks all quantum inversion functionality

---

## Root Cause Summary

The Gaussian initial condition `u₀ = exp(-r²)` with `v₀ = 0` is approximately an **eigenmode** of the elastic stiffness operator K. When K scales with μ (as `K(μ) = μ·K(1)`), eigenmodes evolve with phase that depends on μ, but when combined with:
1. Hermitian dilation structure H = [[0,A],[A†,0]]
2. Zero initial velocity (v₀=0)
3. Normalization in quantum state

The result is that `(H₁ - H₂) @ ψ ≈ 0` for the embedded state, making evolution **identical** regardless of μ.

**Empirical evidence**:
```
||H₁ - H₂||_F = 2.138e+04  ← Hamiltonians differ
||(H₁-H₂)@ψ|| = 0.000e+00  ← Difference vanishes on state!
```

---

## Fix Strategy

### PRIMARY FIX: Modify Initial Conditions

Replace smooth Gaussian IC with **non-eigenmode** initial conditions that have strong spatial variation.

#### Changes to `main.py`

**Current (BUGGY)**:
```python
# Smooth Gaussian pulse - creates eigenmode!
u0 = np.exp(-0.5 * ((x_grid_interior - x_center) / sigma_x) ** 2)
v0 = np.zeros(nx)
```

**Fixed (Option A - Localized spike with gradient)**:
```python
# Localized disturbance with spatial gradient (NOT an eigenmode)
u0 = np.zeros(nx)
center_idx = nx // 2
u0[center_idx] = 1.0
if center_idx + 1 < nx:
    u0[center_idx + 1] = 0.3  # Asymmetric!
if center_idx - 1 >= 0:
    u0[center_idx - 1] = 0.5  # Different on each side

# Non-zero velocity to break eigenmode symmetry
v0 = np.zeros(nx)
v0[center_idx] = 0.1 * np.sqrt(np.mean(mu_true) / np.mean(rho_arr))  # Scale with wave speed
```

**Fixed (Option B - Multiple modes superposition)**:
```python
# Superposition of multiple spatial modes - hard to be eigenmode of all K simultaneously
x_norm = (x_grid_interior - x_grid_interior[0]) / (x_grid_interior[-1] - x_grid_interior[0])
u0 = 0.5 * np.sin(2 * np.pi * x_norm) + 0.3 * np.sin(4 * np.pi * x_norm) + 0.2 * np.sin(6 * np.pi * x_norm)

# Non-zero initial velocity
v0 = 0.1 * np.cos(2 * np.pi * x_norm)
```

**Fixed (Option C - Heterogeneous step function)**:
```python
# Step function - definitely not smooth eigenmode
u0 = np.zeros(nx)
u0[:nx//2] = 1.0  # Left half excited
u0[nx//2:] = 0.2  # Right half different

v0 = np.zeros(nx)
# Add small velocity gradient
v0[:nx//2] = 0.05
```

#### Recommendation

Use **Option B (multiple modes)** for publication because:
- Well-defined mathematically
- Smooth (no Gibbs artifacts)
- Guaranteed to not be eigenmode of any single K
- Creates rich spatial gradients that depend on μ

---

### SECONDARY FIX: Revert Mass-Weighting Change

The mass-weighting removal in earlier commit was based on incorrect hypothesis. **REVERT** to original:

```python
# In quantum_forward_simulate():
sqrt_rho = np.sqrt(rho_bc[1:-1])
inv_sqrt_rho = 1.0 / np.maximum(sqrt_rho, 1e-30)

u_weighted = sqrt_rho * u_interior
v_weighted = inv_sqrt_rho * v_interior
state_vec = np.concatenate([u_weighted, v_weighted])

# ... evolution ...

# Decode:
u_weighted_evolved = np.real(state_vec_evolved[:nx]) * psi_norm
u_decoded = u_weighted_evolved / (sqrt_rho + 1e-30)

v_weighted_evolved = np.real(state_vec_evolved[nx:phys_dim]) * psi_norm
v_decoded = v_weighted_evolved * (sqrt_rho + 1e-30)
```

**Rationale**: Mass-weighting is correct for the elastic wave formulation. The bug was in IC, not in encoding.

---

### TERTIARY FIX: Strengthen Source Term

Make source spatially heterogeneous to create more mu-dependent dynamics:

```python
# In wave/__init__.py, modify source functions:
def gaussian_source(center, width, amplitude, n_points, t0=0.05, sigma_t=0.017):
    def src_func(i, t):
        x_contrib = amplitude * np.exp(-0.5 * ((i - center) / width) ** 2)
        t_contrib = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
        
        # Add spatial gradient to make it non-uniform
        gradient_factor = 1.0 + 0.3 * np.sin(np.pi * i / n_points)
        
        return x_contrib * t_contrib * gradient_factor
    return src_func
```

This makes source amplitude spatially varying, creating waves that interact differently with heterogeneous μ.

---

## Implementation Plan

### Step 1: Test Fix with Minimal Change

Create test to verify Option B IC produces mu-dependent evolution:

```python
# test_ic_fix.py
import numpy as np
from src.optimization.objective import SeismicObjective

nx = 7
u0_old = np.exp(-0.5 * (np.arange(nx) - nx//2)**2 / 2**2)  # Gaussian (buggy)

x_norm = np.arange(nx) / (nx - 1)
u0_new = (0.5 * np.sin(2*np.pi*x_norm) + 
          0.3 * np.sin(4*np.pi*x_norm) + 
          0.2 * np.sin(6*np.pi*x_norm))  # Multi-mode (fixed)

v0 = 0.1 * np.cos(2*np.pi*x_norm)

# Test both
for label, u0_test in [("OLD_BUGGY", u0_old), ("NEW_FIXED", u0_new)]:
    obj = SeismicObjective(nx=nx, dx=63, dt=0.005, steps=10, engine='quantum')
    fields1 = obj.quantum_forward_simulate(mu1, rho, u0_test, v0)
    fields2 = obj.quantum_forward_simulate(mu2, rho, u0_test, v0)
    diff = np.linalg.norm(fields1[-1] - fields2[-1])
    print(f"{label}: ||field1 - field2|| = {diff:.6e}")
```

Expected output:
```
OLD_BUGGY: ||field1 - field2|| = 1.234e-14  (no difference)
NEW_FIXED: ||field1 - field2|| = 5.678e-03  (clear difference!)
```

### Step 2: Apply to main.py

Modify sweep configuration initial condition generation.

### Step 3: Run Full Pipeline

```bash
python main.py
```

Verify:
- ✅ Loss > 0 initially
- ✅ Gradient > 1e-10
- ✅ mu changes across iterations
- ✅ Loss decreases
- ✅ Final mu closer to mu_true than mu_initial

### Step 4: Update Documentation

- README.md: Add section on IC requirements
- BUG_REPORT: Mark as RESOLVED
- AUDIT_REPORT: Add Bug #16 entry

---

## Acceptance Criteria

1. **Unit test passes**: Different μ → different trajectories (Δ > 1e-6)
2. **Gradient non-zero**: ||∇J|| > 1e-10 in first 10 iterations
3. **Optimization progresses**: Loss decreases monotonically for first 20 iterations
4. **Physical correctness**: Recovered μ has lower misfit than initial μ
5. **No regression**: Hamiltonian validation still passes (overlap > 0.80)

---

## Risks and Mitigations

**Risk 1**: New IC may be harder to encode in quantum amplitude encoding
- **Mitigation**: Multi-mode superposition is smooth, no encoding issues expected

**Risk 2**: Non-zero v₀ may violate physical initialization assumptions
- **Mitigation**: v₀ = 0.1 * c is small (10% of wave speed), physically reasonable

**Risk 3**: Change breaks classical reference trajectory
- **Mitigation**: Classical solver handles arbitrary IC, no issue

---

## Files to Modify

1. `main.py` - Change IC generation (lines ~175-180)
2. `src/optimization/objective.py` - REVERT mass-weighting removal
3. `BUG_REPORT_QUANTUM_INVERSION.md` - Update status to RESOLVED
4. `README.md` - Add IC requirements section
5. `test_ic_fix.py` - NEW test file

---

## Rollback Plan

If fix causes issues:
```bash
git revert HEAD  # Revert IC change
# Restore Gaussian IC
# Document as "KNOWN LIMITATION: Smooth Gaussian IC not supported"
```

---

## Timeline

- **Implementation**: 30 min
- **Testing**: 1 hour
- **Documentation**: 30 min
- **Total**: 2 hours

---

## Sign-off

- [ ] Code changes implemented
- [ ] Unit tests pass
- [ ] End-to-end test passes  
- [ ] Documentation updated
- [ ] Peer review (if applicable)
- [ ] Ready to merge

---

**Status**: READY FOR IMPLEMENTATION
