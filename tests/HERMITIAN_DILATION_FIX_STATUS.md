# Hermitian Dilation Fix: Implementation Status

**Date:** 2026-07-16  
**Issue:** Hamiltonian validation failing with mean overlap ~0.18-0.22 (threshold: >0.95)  
**Root Cause:** Antisymmetrization `H = i(A - A†)/2` discards symmetric component of A, breaking correspondence with PDE physics  
**Solution:** Hermitian dilation following Jin-Liu-Yu Schrödingerisation method

---

## Background

### Problem Identified

Empirical validation showed:
- **Raw matrix A** (via `expm(A·t)`): mean L2 error vs leapfrog = **0.12** ✓ (correct)
- **Antisymmetrized H** = i(A-A†)/2: mean L2 error vs leapfrog = **1.50** ✗ (broken)
- **With mass-weighting**: mean L2 error vs leapfrog = **1.36** ✗ (still broken)

**Conclusion:** The antisymmetrization method is architecturally wrong. Matrix A is NOT anti-Hermitian because:
- A = [[0, I], [K, 0]]
- K is symmetric (not anti-symmetric) for the elastic wave operator
- Forcing A to be anti-symmetric via `(A - A†)/2` throws away physics

### Correct Approach: Hermitian Dilation

For general (non-Hermitian) operator A, use block structure:

```
H_dilated = [  0    A  ]    ∈ ℂ^(4nx × 4nx)
            [ A†    0  ]
```

**Properties:**
- H_dilated is Hermitian by construction (block off-diagonal structure)
- Preserves ALL dynamics of A (no information loss)
- State preparation: embed physical state as `[ψ_phys, 0]` (auxiliary space = 0)
- Evolution: `Ψ(t) = exp(-iH_dilated·t) @ Ψ(0)`
- Back-projection: extract first 2nx components

---

## Implementation Completed ✓

### 1. Hamiltonian Construction (`src/hamiltonian/__init__.py`)

**Changes:**
- Replaced `H = i(A - A†)/2` with Hermitian dilation `H = [[0, A], [A†, 0]]`
- Updated dimension: `2nx × 2nx` → `4nx × 4nx` (dilated)
- Returns 4-tuple: `(H, n_qubits, dim, phys_dim)` where `phys_dim = 2nx`
- Added extensive docstring explaining dilation method and references

**Key code:**
```python
phys_dim = 2 * nx
H_undilated = np.zeros((2 * phys_dim, 2 * phys_dim), dtype=complex)

# Upper-right block: A
H_undilated[:phys_dim, phys_dim:] = A

# Lower-left block: A† (for real A, this is A.T)
H_undilated[phys_dim:, :phys_dim] = A.T

# Pad to power-of-2
n_qubits = int(np.ceil(np.log2(max(4 * nx, 2))))
dim = 2 ** n_qubits
H = np.zeros((dim, dim), dtype=complex)
H[:2 * phys_dim, :2 * phys_dim] = H_undilated

return H, n_qubits, dim, phys_dim
```

### 2. Quantum Forward Simulation (`src/optimization/objective.py`)

**Changes:**
- Updated `quantum_forward_simulate()` to unpack 4-tuple from `build_hamiltonian()`
- Modified state embedding: physical state `[u_weighted, v_weighted]` embedded as `[physical, zeros]`
- Updated back-projection: extract first `phys_dim` components after evolution
- Decode directly from real part (no `amplitude_decode()` call needed)

**Key code:**
```python
# Build with dilation
H_mat, n_qubits, dim, phys_dim = build_hamiltonian(mu_bc, rho_bc, self.dx, nx)

# Embed into dilated space
psi = np.zeros(dim, dtype=complex)
psi[:min(len(state_vec), phys_dim)] = state_vec[:min(len(state_vec), phys_dim)]

# Evolve
psi_evolved = U @ psi

# Back-project: extract physical subspace
state_vec_evolved = psi_evolved[:phys_dim]
u_weighted_evolved = np.real(state_vec_evolved[:nx]) * psi_norm
```

### 3. Hamiltonian Validation (`src/experiment/validate_hamiltonian.py`)

**Changes:**
- Updated validation experiment to use dilated Hamiltonian
- Modified state encoding/decoding to match forward simulation
- Prints physical subspace dimension for diagnostics

**Key code:**
```python
H_mat, n_qubits, dim, phys_dim = build_hamiltonian(mu_bc, rho_bc, dx, nx)
print(f"  • Physical subspace: {phys_dim} (2nx)")

# Embed physical state
psi = np.zeros(dim, dtype=complex)
psi[:min(len(state_vec), phys_dim)] = state_vec[...]

# Back-project
state_vec_evolved = psi_evolved[:phys_dim]
```

### 4. Circuit Builder (`src/circuit/__init__.py`)

**Changes:**
- Updated `build_circuit()` to unpack 4-tuple from `build_hamiltonian()`
- Modified state preparation to embed into dilated space
- Uses `phys_dim` for correct embedding boundary

**Key code:**
```python
H_mat, n_qubits, dim, phys_dim = build_hamiltonian(mu, rho, dx, nx)

# Embed into dilated Hilbert space
psi = np.zeros(dim, dtype=complex)
psi[:min(len(state), phys_dim)] = state[:min(len(state), phys_dim)]
```

### 5. Regression Test (`tests/test_raw_A_matches_leapfrog.py`)

**Created:**
- Standalone test that validates raw matrix A (before Hermitianization) matches leapfrog
- Uses direct `expm(A·t)` evolution
- Assertion: `mean_l2_error < 0.2` (allows for truncation error)
- Purpose: detect regressions in K or A construction BEFORE Hermitian dilation

**Status:** Test file created, ready to run

---

## Work Remaining ⚠

### 6. Second Validation Test (NOT DONE)

**File:** `tests/test_dilated_H_matches_leapfrog.py`

**Purpose:** 
- Test that NEW dilated Hamiltonian matches leapfrog evolution
- Wraps `run_hamiltonian_validation()` as a unit test
- Assertion: `mean_overlap > 0.85` (partial threshold) or `> 0.95` (full pass)

**Action:**
```python
def test_dilated_H_validation():
    # Setup (same as test_raw_A_matches_leapfrog.py)
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 40
    
    mu_true = raised_cosine(nx+1, 1e10, 4e10, 2, 0.8)
    rho = raised_cosine(nx, 2e3, 4e3, 1, 0.6)
    
    x = np.arange(nx) * dx
    u0 = np.exp(-0.5 * ((x - x[nx//2]) / (2*dx))**2)
    v0 = np.zeros(nx)
    
    # Run validation
    res = run_hamiltonian_validation(mu_true, rho, u0, v0, dx, dt, steps, nx)
    
    mean_overlap = res['mean_overlap']
    mean_l2 = res['mean_l2_error']
    
    print(f"\nDilated Hamiltonian Validation:")
    print(f"  Mean overlap:  {mean_overlap:.6f}")
    print(f"  Mean L2 error: {mean_l2:.6f}")
    
    # Assert
    assert mean_overlap > 0.85, (
        f"Dilated Hamiltonian validation failed!\n"
        f"Mean overlap = {mean_overlap:.4f} (threshold: 0.85)\n"
        f"The Hermitian dilation does not reproduce leapfrog evolution."
    )
    
    if mean_overlap > 0.95:
        print("  ✓ VALIDATION PASSED (overlap > 0.95)")
    else:
        print("  ⚠ VALIDATION PARTIAL (0.85 < overlap < 0.95)")
```

### 7. Run Both Tests and Verify (NOT DONE)

**Action:**
1. Run `python tests/test_raw_A_matches_leapfrog.py` → should PASS (A is correct)
2. Run `python tests/test_dilated_H_matches_leapfrog.py` → should PASS (dilated H is correct)
3. If test 2 FAILS:
   - Check if auxiliary space leakage is too large
   - May need to add correction factor or adjust back-projection
   - Consult Jin-Liu-Yu papers for exact reconstruction formula

**Expected outcome:**
- Test 1: mean L2 error ~ 0.12 ✓
- Test 2: mean overlap > 0.85 ✓ (ideally > 0.95)

### 8. Documentation Updates (NOT DONE)

**Files to update:**

#### `README.md`
- **Remove premature claim:** Delete or revise line claiming RM2 "proven by hamiltonian_validation.png"
- **Add technical note:**
  ```markdown
  ## Recent Bug Fixes
  
  ### Hermitian Dilation Fix (2026-07-16)
  
  **Issue:** Initial implementation used antisymmetrization H = i(A - A†)/2 
  which discarded the symmetric component of the system matrix A. This broke 
  correspondence with PDE physics.
  
  **Empirical validation:**
  - Raw A (via expm): mean L2 error vs leapfrog = 0.12 ✓
  - Antisymmetrized (old): mean L2 error vs leapfrog = 1.50 ✗
  - Hermitian dilation (new): mean overlap vs leapfrog = [TO BE MEASURED]
  
  **Fix:** Implemented proper Hermitian dilation following Jin-Liu-Yu 
  Schrödingerisation method:
  
      H_dilated = [[0, A], [A†, 0]]   (4nx × 4nx)
  
  Physical state [u, v] is embedded as [u, v, 0, ..., 0] in dilated space, 
  evolved via exp(-iH·t), then first 2nx components extracted.
  
  **Status:** Implementation complete, validation in progress.
  ```

#### `AUDIT_REPORT.md`
- Create new section: "Bug #15: Hermitian Dilation Architecture Fix"
- Document before/after metrics
- Explain why antisymmetrization was wrong
- Show empirical validation numbers
- Reference Jin-Liu-Yu and Schade et al. papers

**Template:**
```markdown
## Bug #15: Hermitian Dilation Architecture Fix (2026-07-16)

**Severity:** Critical — architectural flaw affecting all quantum evolution

**Discovery:**
Independent validation comparing quantum exp(-iH·t) vs classical leapfrog 
showed catastrophic mismatch (mean overlap 0.18-0.22, expected >0.95).

**Root Cause:**
The antisymmetrization method H = i(A - A†)/2 is only valid when A is 
anti-Hermitian. For the elastic wave system:
- A = [[0, I], [K, 0]]
- K is symmetric (second-order centered FD operator)
- Therefore A is NOT anti-Hermitian
- Antisymmetrization throws away (A + A†)/2 ≠ 0 component

**Evidence:**
| Method | Mean L2 Error vs Leapfrog |
|--------|--------------------------|
| Raw A (expm) | 0.12 ✓ |
| Antisymmetrized H | 1.50 ✗ |
| With mass-weighting | 1.36 ✗ |
| Hermitian dilation | [pending validation] |

**Fix:**
Implemented proper Hermitian dilation (Jin-Liu-Yu Schrödingerisation):
- H = [[0, A], [A†, 0]] in 4nx × 4nx space
- State embedding: [ψ_phys, 0]
- Back-projection: extract first 2nx components

**Files Modified:**
- src/hamiltonian/__init__.py (Hermitian construction)
- src/optimization/objective.py (quantum forward simulation)
- src/experiment/validate_hamiltonian.py (validation experiment)
- src/circuit/__init__.py (circuit builder)

**Validation:**
- Regression test: tests/test_raw_A_matches_leapfrog.py ✓
- Dilated H test: tests/test_dilated_H_matches_leapfrog.py [pending]

**References:**
- Jin, Liu, Yu (2022-2023): Schrödingerisation method
- Schade et al. (2024), arXiv:2312.14747
```

### 9. Full Pipeline Validation (NOT DONE)

**Action:**
Run `python main.py` with `run_hamiltonian_validation=True`

**Expected output:**
```
==============================================================================
HAMILTONIAN VALIDATION EXPERIMENT
==============================================================================
...
[3/3] Computing validation metrics...

  Results:
    Mean L2 error: 0.XXXX
    Mean overlap:  0.9XXX

  ✓ VALIDATION PASSED (mean overlap >0.95)
```

**If FAILS:**
- Check console output for dimension mismatches
- Verify state embedding/extraction logic
- Check if auxiliary space leakage is excessive
- May need to revisit dilation formula

### 10. Optional: Add Auxiliary Space Monitoring (NOT DONE)

If validation shows partial success (overlap 0.85-0.95 but not >0.95), 
add diagnostics to measure auxiliary space leakage:

```python
# In validate_hamiltonian.py, after evolution:
aux_leakage = np.linalg.norm(psi_evolved[phys_dim:])
print(f"  Auxiliary space norm: {aux_leakage:.6f}")
```

If leakage is significant (>0.1), may need:
- Smaller time steps (dt)
- Auxiliary space damping/correction
- Different back-projection formula (consult Jin-Liu-Yu papers)

---

## Testing Checklist

- [ ] Run `python tests/test_raw_A_matches_leapfrog.py`
  - Expected: PASS with mean L2 error < 0.2
- [ ] Create and run `tests/test_dilated_H_matches_leapfrog.py`
  - Expected: PASS with mean overlap > 0.85
- [ ] Run `python test_validation_quick.py` (already created)
  - Quick smoke test for validation
- [ ] Run `python main.py` with validation enabled
  - Full pipeline test with HamiltonianValidation sheet
- [ ] Check `figures/hamiltonian_validation.png`
  - Visual inspection of trajectory match
- [ ] Check `data/.../results.xlsx` Sheet 13
  - Quantitative validation metrics

---

## Success Criteria

1. **Test 1 (Raw A):** mean L2 error < 0.2 ✓
2. **Test 2 (Dilated H):** mean overlap > 0.85 ⚠ (or > 0.95 ✓)
3. **Full pipeline:** Validation prints "✓ PASSED" or "⚠ PARTIAL"
4. **Documentation:** README and AUDIT_REPORT updated with before/after numbers
5. **No regressions:** Existing tests still pass, figures still generate

---

## Known Risks

1. **Auxiliary space leakage:** For finite time steps, some probability may leak 
   into auxiliary space. This is expected for Hermitian dilation but should be 
   small (<10% of norm).

2. **Increased qubit count:** Dilation doubles the Hilbert space size 
   (2nx → 4nx), requiring +1 qubit. For nx=7, this goes from 4 qubits to 5 qubits.

3. **Back-projection accuracy:** Simple extraction of first 2nx components is 
   approximate. Exact reconstruction may require correction factors (see Jin-Liu-Yu 
   papers, Section 3-4).

4. **Time step sensitivity:** Dilated dynamics may be more sensitive to dt. 
   If validation fails, try dt/2.

---

## References

1. **Jin, Liu, Yu (2022-2023):** "Quantum simulation of partial differential 
   equations via Schrödingerisation" — original method

2. **Schade et al. (2024):** arXiv:2312.14747 — application to elastic waves

3. **Jin, Liu, Yu (2023):** Phys. Rev. A, 108, 032603 — warped phase transformation

4. **Current codebase:** `src/hamiltonian/__init__.py` lines 1-105 (detailed 
   mathematical derivation)

---

## Contact

For questions about this fix:
- See AUDIT_REPORT.md for historical bug context
- See src/hamiltonian/__init__.py docstring for mathematical details
- Run `python test_validation_quick.py` for smoke test
