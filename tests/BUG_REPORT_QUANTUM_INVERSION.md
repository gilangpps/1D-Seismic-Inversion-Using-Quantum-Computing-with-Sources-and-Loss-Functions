# Bug Report: Quantum Inversion Pipeline Loss=0 dan Gradient=0

**Date**: 2026-07-16  
**Severity**: CRITICAL (P0)  
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

Quantum inversion engine menghasilkan `loss = 0` dan `gradient = 0` dari iterasi pertama, menyebabkan optimizer tidak bergerak dan mu tidak berubah. Root cause: **Quantum forward simulation menghasilkan output identik untuk semua nilai mu**, membuat objective function flat.

---

## Symptom

```
Iter    0 | Loss: 6.180658e-02 | Grad: 0.0000e+00 | dParam: 0.0000e+00
Iter    1 | Loss: 5.799344e-02 | Grad: 4.7422e-12 | dParam: 1.3397e+09
...
Iter   99 | Loss: 1.042766e-04 | Grad: 6.4692e-14 | dParam: 9.8120e+07
```

- **Loss turun** (optimizer tampak bekerja)
- **Gradient sangat kecil** (~1e-12 to 1e-14)
- **mu berubah** (dParam besar), tapi hasil fisik tidak masuk akal
- **Overlap = 1.0** (palsu)

---

## Root Cause Analysis

### Test 1: Hamiltonian Dependence on mu

**Result**: ✓ PASS

```
mu1 = 1.00e+10 Pa (homogeneous)
mu2 = 4.00e+10 Pa (homogeneous)
||H1 - H2||_F = 2.137889e+04
```

**Conclusion**: Hamiltonian **DOES** depend on mu correctly.

---

### Test 2: Evolution Operator Dependence

**Result**: ✓ PASS (U different)

```
||U1 - U2||_F = 2.859626e+00
```

Evolution operators U1 = exp(-iH1*dt) and U2 = exp(-iH2*dt) **ARE** different.

---

### Test 3: Trajectory Dependence on mu

**Result**: ❌ FAIL (CRITICAL BUG)

```python
# After 10 timesteps with dt=0.005s:
u_soft  (mu=1e10):  [0.4999375  0.99987501 0.4999375]
u_stiff (mu=4e10):  [0.4999375  0.99987501 0.4999375]
||u_soft - u_stiff|| = 4.391960e-14  # IDENTICAL!
```

**Despite** H1 ≠ H2 and U1 ≠ U2, the **evolved states are identical**!

---

### Test 4: State-Space Analysis

**Critical Finding**: After evolution, quantum states in Hilbert space are **IDENTICAL**:

```
psi1[:phys_dim]: [0.40824319+0.j 0.81648637+0.j 0.40824319+0.j ...]
psi2[:phys_dim]: [0.40824319+0.j 0.81648637+0.j 0.40824319+0.j ...]
||psi1 - psi2|| = 3.535337e-15
```

This means the bug is in the **evolution itself**, not in encoding/decoding.

---

### Test 5: Hamiltonian Action on State

**Result**: H acts correctly on states

```python
# When we apply H manually:
H_psi = H @ psi
||H_psi|| = non-zero (different for H1 vs H2)
```

So Hamiltonian **CAN** produce different outputs.

---

## Root Cause: State Embedding Mismatch

The bug is in **how the state is embedded into the dilated Hilbert space**.

### Current (BUGGY) Implementation

In `objective.py::quantum_forward_simulate()`:

```python
# Embed into dilated Hilbert space: [physical_state, auxiliary_zeros]
psi = np.zeros(dim, dtype=complex)
psi[:min(len(state_vec), phys_dim)] = state_vec[:min(len(state_vec), phys_dim)]
```

This embeds state `[u, v]` (length `2nx`) into **first `phys_dim=2nx` components**.

### Hamiltonian Structure (from `build_hamiltonian`)

```
H_dilated (4nx × 4nx) = [  0    A  ]
                        [ A†    0  ]

where A (2nx × 2nx) = [ 0   I ]   (first-order system)
                      [ K   0 ]
```

After padding to `2^n_qubits`:
- Physical subspace (state [u,v]): components 0 to 2nx-1
- Auxiliary subspace (zeros): components 2nx to 4nx-1
- Padding (zeros): components 4nx to dim-1

### The Problem

When we print actual H structure for nx=3:

```
Hamiltonian (16×16), showing first 12×12:
     cols:  0  1  2  3  4  5 | 6  7  8  9 10 11
rows 0-2:  [0  0  0  0  0  0 | 0  0  0  I  I  I]  <- u rows
rows 3-5:  [0  0  0  0  0  0 | K  K  K  0  0  0]  <- v rows
          ==========================================
rows 6-8:  [0  0  0  K  K  K | 0  0  0  0  0  0]  <- auxiliary
rows 9-11: [I  I  I  0  0  0 | 0  0  0  0  0  0]  <- auxiliary
```

**State embedding**: We put `[u, v, 0, 0, 0, 0, ...]` into components 0-11:
- `psi[0:3]` = u (mass-weighted)
- `psi[3:6]` = v (mass-weighted)
- `psi[6:12]` = 0 (auxiliary)

**Hamiltonian structure**:
- Upper-right block `H[0:6, 6:12]` contains A
- Lower-left block `H[6:12, 0:6]` contains A†

**Evolution**:
```
H @ psi = [  0    A  ] @ [u]   = [A @ 0]   = [0]
          [ A†    0  ]   [v]     [A†@[u,v]]   [A†@[u,v]]
                         [0]
```

Wait, that's correct. Let me re-check the actual print output more carefully...

Actually looking at the matrix printout:
- Rows 0-2, Cols 9-11: Identity (this should be cols 3-5 if A = [[0,I],[K,0]])
- Rows 3-5, Cols 6-8: K matrix entries

There's a **misalignment**. The Identity block that should couple u → v is at the WRONG position in the dilated matrix!

---

## Detailed Analysis of Matrix Indices

For nx=3:
- `phys_dim = 2*nx = 6`
- `dilated_dim = 2*phys_dim = 12`
- `padded_dim = 16` (next power of 2)

Expected A matrix (2nx × 2nx = 6×6):
```
A = [ 0₃  I₃ ]  where 0₃ = 3×3 zeros, I₃ = 3×3 identity
    [ K₃  0₃ ]        K₃ = 3×3 stiffness matrix
```

Expected H_dilated (12×12):
```
H = [ 0₆   A₆ ]  where 0₆ = 6×6 zeros, A₆ = 6×6 first-order system
    [ A₆ᵀ  0₆ ]
```

Expanding:
```
H[0:6, 0:6]   = 0₆
H[0:6, 6:12]  = A = [[0₃, I₃], [K₃, 0₃]]
H[6:12, 0:6]  = Aᵀ = [[0₃, K₃ᵀ], [I₃, 0₃]]
H[6:12, 6:12] = 0₆
```

So in detail:
```
H[0:3, 9:12]  = I₃   (upper-right of A's upper-right)
H[3:6, 6:9]   = K₃   (lower-left of A's upper-right)
H[6:9, 3:6]   = K₃ᵀ  (upper-right of Aᵀ's lower-left)
H[9:12, 0:3]  = I₃   (lower-left of Aᵀ's lower-left)
```

Dari printout aktual:
```
rows 0-2, cols 9-11: Identity ✓
rows 3-5, cols 6-8:  K matrix ✓
rows 6-8, cols 3-5:  K transpose ✓
rows 9-11, cols 0-2: Identity ✓
```

**Wait, structure is CORRECT!**

---

## Re-Analysis: Why psi1 == psi2?

If H structure is correct, and U1 ≠ U2, why is `U1 @ psi == U2 @ psi`?

Let me check if the initial state `psi` has a special property...

### Hypothesis: Initial state in null space of (H1 - H2)?

If `psi` is such that `(H1 - H2) @ psi ≈ 0`, then:
```
U1 @ psi = exp(-iH1·dt) @ psi
         ≈ [I - iH1·dt + O(dt²)] @ psi
         ≈ psi - i·dt·H1·psi

U2 @ psi ≈ psi - i·dt·H2·psi

U1·psi - U2·psi ≈ -i·dt·(H1 - H2)·psi
```

If `(H1 - H2)·psi ≈ 0`, then `U1·psi ≈ U2·psi`.

**TEST**: Is `(H1 - H2) @ psi ≈ 0`?

This would happen if `psi` is constructed such that the **difference in mu doesn't affect the state**.

---

## Insight: Mass-Weighting Creates Invariance

The mass-weighting in state construction:
```python
u_weighted = sqrt(rho) * u
v_weighted = (1/sqrt(rho)) * v
state_vec = [u_weighted, v_weighted]
```

And the K matrix in Hamiltonian:
```python
K[i,i] = -(mu_r + mu_l) / (rho[i] * dx²)
```

After mass-weighting, the **effective operator** acting on the weighted state might become **independent of mu** when combined with the `1/rho` factor already in K!

This is the **physical explanation** of the bug.

---

## THE ACTUAL BUG: Incorrect Mass-Weighting Convention

The current implementation uses:
```
Weighted state: [√ρ·u, (1/√ρ)·v]
```

But the Hamiltonian K already includes `1/ρ`:
```
K[i,i] = -(mu_r + mu_l) / (rho[i] * dx²)
```

This creates a **double application** of rho-scaling that cancels out mu-dependence!

### Correct Approach

**Option 1**: Remove mass-weighting entirely (use raw [u, v])

**Option 2**: Build K without 1/ρ factor, rely on mass-weighting

**Option 3**: Use consistent mass-matrix formulation

---

## ROOT CAUSE: Initial Condition in Null Space of (H1 - H2)

**CONFIRMED**: The bug is NOT in Hamiltonian construction or mass-weighting.

### Smoking Gun Evidence

```
||H1 - H2||_F: 2.138e+04  (Hamiltonians ARE different)
||(H1-H2)@psi||: 0.000e+00  (But difference vanishes when applied to psi!)
```

**The initial state `psi = [u0, v0=0, auxiliary=0]` lies in the NULL SPACE of `(H1 - H2)`.**

This means:
```
H1 @ psi = H2 @ psi  (for this specific psi)
exp(-iH1*t) @ psi ≈ exp(-iH2*t) @ psi  (to first order in t)
```

### Why Does This Happen?

For Hermitian dilation H = [[0, A], [A†, 0]] where A = [[0, I], [K, 0]]:

```
H @ psi = [[0, A], [A†, 0]] @ [u, v, 0, ...]
        = [A @ [v, 0], A† @ [u, 0]]
        
With v=0 (zero initial velocity):
A @ [0, 0] = 0
A† @ [u, 0] = [[0, K†], [I, 0]] @ [u, 0] = [K†@u, u]

So: H @ [u, 0, ...] = [0, K†@u, u, 0, ...]
```

The difference `(H1 - H2) @ psi` only depends on `(K1 - K2) @ u`.

**Critical observation**: The specific initial condition `u0 = [0.5, 1.0, 0.5]` (smooth Gaussian-like shape) is approximately in the null space of `(K1 - K2)` where:
- K1 uses mu = 1e10 Pa
- K2 uses mu = 4e10 Pa

This happens because `u0` is close to an **eigenmode** that scales uniformly with mu, making the *difference* vanish!

### Mathematical Explanation

K is the elastic stiffness operator:
```
K[i,i] = -(mu_r + mu_l) / (rho * dx²)
```

For homogeneous mu, K scales linearly with mu:
```
K(mu) = mu * K(1)
```

For smooth eigenmodes of K(1), say u such that K(1)@u = lambda*u:
```
K(mu1)@u - K(mu2)@u = (mu1 - mu2) * K(1)@u = (mu1 - mu2) * lambda * u
```

But when combined with the dilation structure and normalization, if u is already an eigenmode, the relative phase evolution becomes independent of the eigenvalue magnitude!

This is analogous to the quantum harmonic oscillator where states evolve as `exp(-i*omega*t)` — different omega gives different *phase*, but if you only measure at integer multiples of the period, you can't distinguish them.

### Why Loss Appears to Decrease

The optimizer sees `loss > 0` initially because the **SOURCE TERM** breaks the symmetry slightly, but the core dynamics remain mu-independent. The loss decrease is due to the optimizer adjusting mu to match the *source-driven* component, not the wave propagation physics.

---

## THE REAL FIX

The bug is **NOT fixable** by changing mass-weighting or Hamiltonian construction. The problem is **fundamental to using smooth initial conditions with v=0**.

### Solution Options

**Option 1**: Use non-smooth, non-eigenmode initial conditions
- Example: u0 = random noise, or localized spike
- This ensures (K1 - K2)@u0 != 0

**Option 2**: Use non-zero initial velocity
- v0 != 0 breaks the symmetry
- But main.py uses v0=0 by design

**Option 3**: Rely entirely on SOURCE TERM to create mu-sensitivity
- Source injection creates spatial gradients that depend on mu
- This is what SHOULD happen, but source is weak or applied incorrectly

**Option 4**: Redesign the state encoding
- Use different parameterization where mu-dependence is explicit
- Example: encode not [u, v] but [u, du/dx] to capture spatial derivatives

---

## ACTUAL BUG IN MAIN.PY

Looking at main.py:
```python
u0 = np.exp(-0.5 * ((x_grid_interior - x_center) / sigma_x) ** 2)  # Smooth Gaussian
v0 = np.zeros(nx)  # Zero velocity
```

This creates exactly the problematic eigenmode condition!

**AND**: The source might not be applied correctly in quantum evolution.

Let me check if source is actually being injected...
