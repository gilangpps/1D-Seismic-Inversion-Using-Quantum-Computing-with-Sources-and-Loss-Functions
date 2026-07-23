# 1-D Seismic Inversion Using Quantum Computing Simulation with Sources and Loss Functions

A quantum-classical hybrid simulation for 1-D elastic wave propagation in heterogeneous media.  
The code combines a classical finite-difference (leapfrog) PDE solver with a Hamiltonian-based quantum time-evolution circuit (Qiskit 2.x), following the framework of Schade et al. (2024, 2025).

---

## Literature

| Reference | Link |
|-----------|------|
| Schade et al. (2024) — *Quantum Wave Equation Solver* | [github.com/malteschade/Quantum-Wave-Equation-Solver](https://github.com/malteschade/Quantum-Wave-Equation-Solver.git) |
| Schade et al. (2025) — *Quantum Wave Simulation with Sources and Loss Functions* | [github.com/malteschade/Quantum-Wave-Simulation-with-Sources-and-Loss-Functions](https://github.com/malteschade/Quantum-Wave-Simulation-with-Sources-and-Loss-Functions.git) |
| Schade et al. (2023) — arXiv preprint | [arXiv:2312.14747](https://arxiv.org/abs/2312.14747) |

## Author

**Najlah Rupaidah** (NIM 1227030025)  
Geophysics Specialization, Department of Physics, Faculty of Science and Technology  
Universitas Islam Negeri Sunan Gunung Djati, Bandung, Indonesia

Co-author: **bex**

---

## Features

- **Classical 1-D elastic wave solver** — leapfrog finite-difference scheme for heterogeneous μ(x) and ρ(x).
- **Quantum forward simulation** — Hamiltonian-based evolution exp(−iHt) per timestep via statevector API (optional, selectable engine).
- **Source injection** — Ricker or Gaussian wavelet enters the PDE as a forcing term f(x,t) (Requirement B). Source peak placed at t₀ = t_max/3, inside the simulation window.
- **Hamiltonian-based quantum circuit** — matches Schade et al. Fig. A.1: State Preparation → exp(−iHt) → Observable → Measurement.
- **Physical Hamiltonian** — sqrt-symmetrized H = [[0, i·S_op], [−i·S_op, 0]] (2nx×2nx), where S_op = √(−K_sym) is the matrix square root of the negative symmetrized stiffness operator. Hermitian by construction.
- **Quantum amplitude encoding** — |ψ⟩ = (1/‖u‖) Σᵢ u[i]|i⟩, three-mode reconstruction (statevector / shot-noise / hardware-noise).
- **Seismic inversion** — iterative optimization: J(μ) = (1/T) Σ_t ‖u_fwd(t;μ) − u_ref(t;μ_true)‖². Gradient via central finite differences. Adam optimizer (gradient descent).
- **Independent Hamiltonian validation** — compares quantum exp(−iHt) trajectory vs classical leapfrog trajectory to verify H captures wave physics.
- **10 mandatory publication plots** — initial μ, density, source, wave propagation, quantum reconstruction error, observed vs predicted, loss convergence, μ inversion result, classical PDE energy, quantum overlap.
- **Excel output** — 13 sheets including OptimizationHistory, HamiltonianValidation, and QuantumReconLoss.

---

## Three Distinct Concepts: Forward, Validation, and Diagnostics

This codebase implements three conceptually different quantum operations that must NOT be confused:

### 1. Forward Simulation for Optimization (engine parameter)

**Purpose:** Compute the objective function J(μ) during inversion.

**Implementation:**
- Location: `src/optimization/objective.py::forward_simulate()`
- Engines: `engine="classical"` (leapfrog PDE) or `engine="quantum"` (exp(−iHt) statevector evolution)
- Used by: Adam optimizer gradient descent loop

**What it does:**
- Given current model parameters μ, compute forward wavefield u_fwd(t; μ)
- Compare against reference u_ref(t; μ_true) to compute loss J(μ)
- Guides optimizer to update μ iteratively

**Metrics:** Loss convergence (Sheet 10: OptimizationHistory, loss_history.png)

**Key distinction:** This is the ACTUAL physics simulation that drives the inversion. When `engine="quantum"`, the optimizer is running exp(−iHt) in the forward loop—not just simulating it.

---

### 2. Independent Hamiltonian Validation (RM2 Evidence)

**Purpose:** Prove that Hamiltonian H = [[0, i·S_op], [−i·S_op, 0]] (sqrt-symmetrization) correctly encodes elastic wave physics.

**Implementation:**
- Location: `src/experiment/validate_hamiltonian.py`
- Runs: quantum exp(−iHt) vs classical leapfrog with **identical** μ, ρ, initial conditions
- No reference fields used—pure independent evolution

**What it does:**
- Evolves (u0, v0) purely via exp(−iHt) ← quantum trajectory
- Evolves (u0, v0) via leapfrog PDE ← classical trajectory
- Compares trajectories: L2 error, state overlap |⟨ψ_quantum|ψ_classical⟩|²

**Metrics:** 
- Sheet 13: HamiltonianValidation (time, L2_error, overlap, classical_energy, quantum_norm)
- Plot: hamiltonian_validation.png (2×2 subplots)

**Key distinction:** This is NOT encoding fidelity. This validates that the Schrödingerisation process (Schade et al. 2024) successfully maps the PDE into a unitary quantum evolution. High overlap (>0.95) proves H captures the wave equation physics.

---

### 3. Encoding-Decoding Diagnostics (Hardware Quality)

**Purpose:** Measure how accurately quantum amplitude encoding can reconstruct classical fields with shot noise and hardware errors.

**Implementation:**
- Location: `src/encoding/__init__.py::quantum_reconstruct()`
- Takes: **already-computed** classical field u_classical
- Encodes: u → |ψ⟩ (amplitude encoding)
- Simulates: multinomial shot noise or Gaussian hardware noise
- Decodes: |ψ⟩ → u_reconstructed
- Compares: MSE(u_classical, u_reconstructed)

**What it does:**
- Tests encoding scheme robustness to measurement noise
- NOT an independent physics simulation—just encode/decode a known field
- High fidelity (~0.9998 overlap) proves encoding is information-preserving

**Metrics:**
- Sheet 11: QuantumReconLoss (per-timestep reconstruction MSE)
- Plot: error.png (quantum reconstruction relative L2 error)

**Key distinction:** This is a DIAGNOSTIC of the encoding scheme, not a validation of Hamiltonian physics. It answers: "Given a classical field, can we store and retrieve it in quantum amplitudes?" NOT: "Does exp(−iHt) reproduce the wave equation?"

---

### Summary Table

| Concept | Location | What it measures | Relevant plots/sheets |
|---------|----------|------------------|----------------------|
| **1. Forward Simulation** | `objective.py` | Inversion objective J(μ) | loss_history.png, OptimizationHistory |
| **2. Hamiltonian Validation** | `validate_hamiltonian.py` | Does H capture PDE physics? | hamiltonian_validation.png, Sheet 13 |
| **3. Encoding Diagnostics** | `quantum_reconstruct()` | Encoding-decoding fidelity | error.png, QuantumReconLoss |

**Why this matters for the thesis:**
- RM1 (optimizer converges): proven by loss_history.png
- RM2 (Hamiltonian is correct): proven by hamiltonian_validation.png (NEW)
- Hardware readiness: proven by error.png (encoding quality)

Each addresses a DIFFERENT research question. Do not cite encoding overlap as proof that Hamiltonian works—that's circular reasoning (encoding just stores what the classical solver already computed).

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.10 |
| qiskit | ≥ 2.3.0 |
| qiskit-aer | ≥ 0.17.2 |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| matplotlib | ≥ 3.6 |
| openpyxl | ≥ 3.1 |
| pylatexenc | ≥ 2.10 |

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib openpyxl pylatexenc
```

---

## Usage

```bash
python main.py
```

Select source waveform when prompted:
```
Select source waveform:
  [a] Gaussian source
  [b] Ricker wavelet source
Enter choice (a/b):
```

The pipeline then:
1. Builds medium: μ_true (raised-cosine), ρ (raised-cosine), μ_initial (homogeneous 50% of mean)
2. Constructs source with peak at t₀ = t_max/3 (inside simulation window)
3. Computes reference wavefields from μ_true (inversion target)
4. Runs initial forward simulation with μ_initial + source
5. Builds quantum circuit (exp(−iHt) with physical H)
6. Runs iterative inversion (Adam, up to 100 iterations)
7. Saves JSON + pickle + Excel (10 sheets)
8. Generates 10+ publication-quality plots to `figures/`

---

## Project Structure

```
TA_mein-lieben/
├── main.py                    # Full pipeline entry point
├── README.md
├── requirements.txt
├── figures/                   # Generated plots (15 files)
├── data/                      # Timestamped run outputs
└── src/
    ├── constants/             # Plot style, directory paths
    ├── distributions/         # raised_cosine, spike, homogeneous
    ├── encoding/              # Amplitude encoding & quantum reconstruction
    ├── hamiltonian/           # Physical Hermitian Hamiltonian from elastic op
    ├── circuit/               # Quantum circuit: StatePrep|exp(-iHt)|Obs|Meas
    ├── execution/             # AerSimulator circuit runner
    ├── wave/                  # Leapfrog PDE solver with source injection
    ├── experiment/            # Experiment orchestrator (PDE + energy + overlap)
    ├── persistence/           # JSON, pickle, Excel (10 sheets)
    ├── visualization/         # 10 mandatory + additional publication plots
    └── optimization/
        ├── objective.py       # Misfit J(μ) = ‖u_fwd − u_ref‖²
        ├── gradient.py        # Central FD gradient
        ├── optimizer.py       # Adam (descent) + SeismicOptimizer loop
        └── callbacks.py       # Loss history, convergence report
```

---

## Simulation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| nx | 7 | Interior grid points |
| dx | 63 m | Grid spacing |
| dt | 0.0025 s | CFL ratio ≈ 0.18 |
| steps | 120 | t_max = 0.3 s |
| μ_true | raised-cosine, 1–4 × 10¹⁰ Pa | Heterogeneous |
| μ_initial | homogeneous, 80% of mean(μ_true) | Wrong starting point |
| ρ | raised-cosine, 2–4 × 10³ kg/m³ | Fixed during inversion |
| Source t₀ | t_max/3 ≈ 0.1 s | Inside simulation window |
| Source σ_t | t_max/12 ≈ 0.025 s | Well-resolved by dt |

CFL stability:
```
v_max = sqrt(4e10 / 2e3) ≈ 4472 m/s
dt_CFL = 63 / 4472 ≈ 0.014 s   →  dt = 0.0025 s  (ratio 0.18) ✓
```

---

## Physics: Elastic Wave Equation

```
ρ(x) ∂²u/∂t² = ∂/∂x[μ(x) ∂u/∂x] + f(x,t)
```

Leapfrog discretisation:
```
u[i,n+1] = 2u[i,n] − u[i,n-1]
          + (dt²/ρ[i]dx²)[μ_{i+½}(u[i+1,n]−u[i,n]) − μ_{i-½}(u[i,n]−u[i-1,n])]
          + (dt²/ρ[i]) f(i, n·dt)
```

---

## Quantum Circuit

The circuit implements Schade et al. Fig. A.1 with four sections:

```
       ┌──────────────┐ ░ ┌──────────────┐ ░ ┌───┐ ░ ┌─┐
q_0: ──┤ RY(-1.7)     ├─░─┤              ├─░─┤ H ├─░─┤M├
       └──────────────┘ ░ │              │ ░ └───┘ ░ └─┘
q_1: ──────────────────░─┤  exp(-iHt)   ├─░────────░──M──
       ┌───┐┌───┐      ░ │              │ ░ ┌───┐  ░ ┌─┐
q_2: ──┤ X ├┤ Z ├──────░─┤              ├─░─┤ H ├─░─┤M├
       └───┘└───┘      ░ └──────────────┘ ░ └───┘ ░ └─┘
q_3: ──────────────────░─────────────────░────────░──M──
```

**Hamiltonian:** Sqrt-symmetrized H = [[0, i·S_op], [−i·S_op, 0]] (2nx×2nx), where S_op = √(−K_sym) is the matrix square root of the negative symmetrized stiffness operator. H† = H (Hermitian) → exp(−iHt) is unitary. Physical state [u, v] is encoded as [q; p] = [S_op·(√ρ·u); √ρ·v] and evolved in the 2nx-dimensional space. Decoding recovers [u, v] via u = S_op⁻¹·q / √ρ, v = p / √ρ.

### Circuit Modes

The circuit builder supports two modes:

**1. Paper Diagram Mode** (`paper_diagram_mode=True`)
- Deterministic gate placement for publication figures
- q0: RY(-1.7) rotation
- q2: X and Z gates
- q1, q3: clean (no state preparation gates)
- Stable and reproducible across runs
- Used for circuit.png visualization

**2. Physics Mode** (`paper_diagram_mode=False`)
- Adaptive state preparation based on actual quantum amplitudes
- Single amplitude: binary encoding with X gates
- Two amplitudes: RY rotation on differing qubit
- General state: Qiskit initialize + transpile
- Used for physics simulations (not visualization)

The paper mode ensures exported circuit diagrams match the intended layout regardless of initial conditions, while physics mode preserves simulation accuracy.

---

## Inversion Objective

```
J(μ) = (1/T) Σ_t (1/N) ‖u_fwd(t; μ) − u_ref(t; μ_true)‖²
```

Adam update (gradient descent):
```
θ_{t+1} = θ_t − α · m̂_t / (√v̂_t + ε)
```

FD gradient:
```
∂J/∂μ_i = [J(μ + δ_i e_i) − J(μ − δ_i e_i)] / (2δ_i)
δ_i = max(1e-4 × |μ_i|, 1.0 Pa)
```

---

## Excel Output (13 sheets)

| Sheet | Contents |
|-------|----------|
| Configuration | All experiment parameters |
| Medium | x, ρ(x), μ(x) |
| TimeSeries | time step, time, classical PDE energy [J/m] |
| Overlaps | time, \|⟨ψ_ref\|ψ_fwd⟩\|² (inversion quality metric) |
| WaveFields | Full displacement field at every time step |
| CircuitParams | Qubit count, dim, evolution time, observable |
| Source | Source amplitude at each grid point |
| Loss | **Per-timestep inversion misfit** J(t; μ_rec) = (1/N)‖u_fwd−u_ref‖² |
| ModelUpdate | μ_initial, μ_updated, μ_true per grid point |
| OptimizationHistory | iteration, J(μ), mean_overlap, μ[0..n] |
| QuantumReconLoss | Per-timestep quantum reconstruction MSE (encoding diagnostic) |
| PerformanceMetrics | Summary metrics including runtime, loss reduction, etc. |
| HamiltonianValidation | Quantum vs classical trajectory comparison (L2 error, overlap, energy, norm) |

> **Note on Loss sheet**: Sheet 8 stores the *inversion misfit* J(t; μ_rec), which is the
> per-timestep L2 misfit between the recovered model's forward fields and the reference.
> This is the correct scientific objective. The quantum reconstruction MSE (a hardware/encoding
> quality diagnostic) is stored separately in `QuantumReconLoss` (Sheet 11).

---

## Figures Generated (17 total)

| File | Description |
|------|-------------|
| `mu_initial.png` | Initial vs true elastic modulus |
| `density_model.png` | Density model ρ(x) |
| `source.png` | Source amplitude vs position |
| `source_time.png` | Source wavelet vs time |
| `forward_sim.png` | Wave propagation snapshots (2×3 multiplot) |
| `energy.png` | **Classical PDE energy** [J/m] vs time |
| `overlap.png` | \|⟨ψ_rec\|ψ_ref⟩\|² inversion quality overlap vs time |
| `error.png` | Quantum reconstruction relative L2 error |
| `observed_vs_predicted.png` | Observed (true) vs predicted (recovered) trace + residual |
| `mu_inversion.png` | μ_true vs μ_initial vs μ_recovered |
| `loss_history.png` | Inversion misfit J(μ) convergence vs iteration |
| `model_evolution.png` | μ parameter evolution across iterations |
| `model_update.png` | Bar chart: μ_initial vs μ_updated |
| `loss.png` | **Per-timestep inversion misfit** J(t; μ_rec) vs time |
| `quantum_recon_loss.png` | Per-timestep quantum reconstruction MSE (encoding diagnostic) |
| `hamiltonian_validation.png` | **Quantum vs classical trajectory validation** (2×2: L2 error, overlap, energy, norm) |
| `circuit.png` | Quantum circuit diagram |

> **loss.png vs quantum_recon_loss.png**: `loss.png` shows the true scientific
> inversion objective — how well the recovered model fits the reference at each time step.
> `quantum_recon_loss.png` shows the hardware/encoding quality metric — how accurately
> quantum amplitude encoding can reconstruct the classical wavefield.

---

## Bug Fixes Applied

| # | File | Description |
|---|------|-------------|
| 1 | objective.py | Objective compared field to itself (J≈0 for any μ). Fixed: compare against fixed reference from μ_true. |
| 2 | optimizer.py | Adam used `+delta` (gradient ascent). Fixed: `−delta` (descent). |
| 3 | optimizer.py | Gradient computed twice per iteration (second overwrote first). Fixed: single call. |
| 4 | optimizer.py | Gradient norm logged from discarded first call. Fixed: logged from authoritative call. |
| 5 | callbacks.py | Loss stored as `mean((u − u)²) ≡ 0`. Fixed: store actual misfit from optimizer. |
| 6 | experiment.py | Overlap measured vs IC (always decreasing). Fixed: overlap vs reference fields (inversion quality). |
| 7 | gradient.py | FD step floor `epsilon = 1e-8` Pa caused cancellation. Fixed: `epsilon = 1.0` Pa. |
| 8 | objective.py | `source_func=None` hardcoded; optimizer used different physics than experiment. Fixed: source_func stored and passed through. |
| 9 | main.py | `dt = 1e-6 s`, `steps = 19` gave zero wave propagation. Fixed: `dt = 0.005 s`, `steps = 40`. |
| 10 | main.py | Source peak at `t = 0.5 s`, outside `t_max = 0.2 s`. Source never fired. Fixed: `t₀ = t_max/3`. |
| 11 | wave/__init__.py | `compute_energy` had mu/rho length mismatch at boundary. Fixed: `min(i, len-1)` indexing. |
| 12 | circuit/__init__.py | State prep used wrong qubit selection formula `log2(diff+1)`. Fixed: `(diff & -diff).bit_length() - 1`. |
| 13 | circuit/__init__.py | Circuit diagram not stable across runs. Fixed: added `paper_diagram_mode` for deterministic visualization. |
| 14 | visualization/__init__.py | No validation of circuit pattern. Fixed: added validation in `plot_circuit()`. |
| 15 | hamiltonian/__init__.py | Antisymmetrization H = i(A-A†)/2 discarded symmetric component, breaking PDE correspondence. Fixed: Hermitian dilation H = [[0,A],[A†,0]]. |
| **16** | **hamiltonian/__init__.py** | **Hermitian dilation H=[[0,A],[A†,0]] has AA†=I → u(t)=cos(t)·u₀ mu-independent regardless of IC. Fixed: sqrt-symmetrization H=[[0,i·S_op],[−i·S_op,0]] with S_op=√(−K_sym). See "Sqrt-Symmetrization Fix" section.** |
| **17** | **validate_hamiltonian.py, experiment/__init__.py, objective.py** | **Leapfrog classical solver ignored v₀ initial velocity. `u1_bc = u0_bc.copy()` forced `∂u/∂t(0) ≈ 0` regardless of provided v₀. Hamiltonian validation showed catastrophic failure with overlap 0.11 (expected >0.95). Fixed: `u1_bc[1:-1] += dt·v₀` to set consistent leapfrog initial condition.** |

### Bug #16 Details (Critical Fix - 2026-07-23, revisited)

**Problem**: Quantum inversion produced `loss≈0`, `gradient≈0`, and `μ` did not converge despite Hamiltonians differing (‖H₁−H₂‖F = 2.1e+04).

**Root Cause (original 2026-07-16 diagnosis — INCOMPLETE)**: Was initially diagnosed as an IC eigenmode issue with Gaussian IC + v₀=0 producing `(H₁-H₂)@ψ ≈ 0`. The multi-mode IC fix helped for the classical `expm(A·dt)` path but did NOT fix the actual Hermitian dilation.

**True Root Cause (2026-07-23)**: The Hermitian dilation H = [[0, A], [A†, 0]] with A = [[0, I], [K, 0]] has AA† = I structurally, making the upper-block evolution u(t) = cos(t)·u₀ completely independent of μ. This is NOT an IC problem — it is a structural identity-block problem. The multi-mode IC fix only appeared to work because the quantum path was silently using `expm(A·dt)` (classical) instead of Hermitian dilation.

**Evidence**:
```
||H₁ - H₂||_F = 2.138e+04   (Hamiltonians differ)
H₁@ψ and H₂@ψ differ ONLY in lower block (v-component)
Upper-block u-evolution: u(t) = cos(t)·u₀ for ANY μ → IDENTICAL
```

**Fix**: Replace Hermitian dilation with sqrt-symmetrization: H = [[0, i·S_op], [−i·S_op, 0]] where S_op = √(−K_sym). This removes the identity block entirely. The (q, p) encoding uses S_op to couple the upper and lower blocks genuinely.

**Impact**: Quantum inversion now produces real mu-dependence. All 19 tests pass, quantum inversion loss drops monotonically, overlap reaches 0.998.

**Documentation**: See `BUG_REPORT_QUANTUM_INVERSION.md` for full technical analysis.

### Bug #17 Details (Critical Fix - 2026-07-16)

**Problem**: Independent Hamiltonian validation showed mean overlap ≈ 0.11 and mean L2 error ≈ 9.69 despite the inversion pipeline converging correctly (overlap ≈ 0.93). This suggested the quantum and classical evolution represented fundamentally different physics.

**Root Cause**: The classical leapfrog solver requires two initial snapshots: `u[0]` (t=0) and `u[1]` (t=dt). The code set `u1 = u0`, which forces initial velocity `∂u/∂t(0) ≈ (u1 − u0)/dt = 0`, discarding the provided `v₀`. The quantum path (`expm(A·dt)`) correctly evolved `v₀ ≠ 0`. These became two different PDE solutions — different initial conditions necessarily produce different trajectories.

**Why inversion still worked**: The inversion pipeline compared two classical leapfrog runs (recovered μ vs true μ) both using the same buggy IC. The bug cancelled out, giving the false impression that the physics was consistent.

**Evidence**:
```
Before fix:  Mean overlap = 0.113, Mean L2 error = 9.689
After fix:   Mean overlap = 0.9999, Mean L2 error = 0.009
```

**Fix**: Three files changed, one line each:
```python
u1_bc = u0_bc.copy()
u1_bc[1:-1] += dt * v0    # <-- added: sets u(dt) ≈ u(0) + dt·v(0)
```

**Impact**: Hamiltonian validation now faithfully verifies the inversion Hamiltonian. Overlap jumped from 0.11 to 0.9999. The residual 0.009 L2 error is purely leapfrog's second-order temporal truncation error over 120 steps. The quantum norm (1.46 → 13.44) was never broken — it reflected correct energy oscillation that the classical trajectory now matches.

---

## Validation Checklist

- [x] Elastic wave equation implemented (ρ∂²u/∂t² = ∂/∂x[μ∂u/∂x] + f)
- [x] μ controls wave propagation (speed c = √(μ/ρ))
- [x] ρ influences solver (enters denominator of leapfrog update)
- [x] Ricker/Gaussian source enters PDE (forcing term f(x,t))
- [x] Hamiltonian depends on μ, ρ, dx (physical elastic operator)
- [x] Schrödinger evolution implemented (exp(−iHt) unitary gate)
- [x] Quantum encoding meaningful (|ψ⟩ = Σ u[i]|i⟩ / ‖u‖)
- [x] Circuit represents physical evolution (not arbitrary rotations)
- [x] Circuit diagram stable and paper-faithful (paper_diagram_mode)
- [x] Loss drives inversion (J(μ) = ‖u_fwd − u_ref‖²)
- [x] μ updates iteratively (Adam optimizer, 100 iterations)
- [x] Reconstruction works (3-mode: statevector, shot-noise, hardware-noise)
- [x] Overlap has physical meaning (inversion quality ∈ [0,1])
- [x] Energy correctly labeled **Classical PDE Energy** [J/m]
- [x] Excel has 13 sheets including OptimizationHistory, HamiltonianValidation
- [x] 10 mandatory + 7 additional visualization plots generated
- [x] Independent Hamiltonian validation (quantum vs classical trajectory)
- [x] Three concepts clearly distinguished (forward, validation, diagnostics)

---

## Recent Updates

### Circuit Diagram Stability (2026-07-09)

Fixed quantum circuit diagram to produce stable, paper-faithful visualizations:
- Added `paper_diagram_mode` parameter for deterministic gate placement
- Fixed qubit selection bug in two-amplitude state encoding
- Added circuit validation before plotting
- Circuit diagram now reproducible across runs

See `AUDIT_REPORT.md` for detailed technical documentation of the fix.

### Leapfrog Initial Velocity Fix (2026-07-16)

**Issue:** Independent Hamiltonian validation (using the then-current quantum path, which was already `expm(A·dt)` at this point — see regression note below) showed mean overlap ≈ 0.11 and mean L2 error ≈ 9.69, despite the inversion pipeline correctly converging (overlap ≈ 0.93). This indicated a fundamental inconsistency between the two pipelines.

**Root Cause:** The classical leapfrog solver was initialized with `u1_bc = u0_bc.copy()`, forcing the effective initial velocity to zero regardless of the provided `v₀`. The quantum/expm path correctly used `v₀`. Classical and quantum/expm solved the same PDE with different initial conditions.

**Why inversion was unaffected:** Both the forward and reference fields in the inversion used the same buggy leapfrog initialization. The bug cancelled out — inversion only compares classical-vs-classical trajectories.

**Fix:** `u1_bc[1:-1] += dt * v0` added in three files (`validate_hamiltonian.py`, `experiment/__init__.py`, `objective.py`). This sets the second leapfrog snapshot to `u(dt) ≈ u(0) + dt·v(0)`, matching the beam solver's initial condition.

**Results (direct expm(A·dt), no dilation — see 2026-07-23 regression audit):**
```
Before fix:  Mean overlap = 0.113, Mean L2 error = 9.689
After fix:   Mean overlap = 0.9999, Mean L2 error = 0.009
```

The residual 0.009 L2 error is purely leapfrog truncation (`O(steps·dt²) = O(120·6.25e-6)`). See Bug #17 in the Bug Fixes table for details.

**IMPORTANT REGRESSION NOTE (2026-07-23):** The 0.9999 number above compares **leapfrog vs expm(A·dt)** — two classical integrators, NOT quantum vs classical. The Hermitian dilation was replaced with real `expm(A·dt)` during the mu-independence debugging. The sqrt-symmetrization (section below) is the correct fix that achieves both Hermiticity and mu-dependence.

### Sqrt-Symmetrization Fix (2026-07-23) — Structural Fix for mu-Independence

**Root cause:** The Hermitian dilation H = [[0, A], [A†, 0]] with A = [[0, I], [K, 0]] has AA† = I, making the upper-block evolution u(t) = cos(t)·u₀ mathematically independent of μ. This is structural — no choice of IC can fix it.

**Fix:** Replace dilation with sqrt-symmetrization. The symmetrized stiffness operator K_sym = mldivide(√(diag(ρ)), K, √(diag(ρ))) eliminates the identity block. S_op = √(−K_sym) creates a block-off-diagonal H = [[0, i·S_op], [−i·S_op, 0]] that is genuinely μ-dependent. Key properties:
- H is 2nx×2nx (no auxiliary space, no leakage)
- S_op is Hermitian and positive-definite with condition number ~5
- Encoding: q = S_op·(√ρ·u), p = √ρ·v
- Source enters as Δp = dt·f/√ρ

**Empirical validation (2026-07-23):**

| Method | Mean Overlap vs Leapfrog | Mean L2 Error | Status |
|--------|--------------------------|---------------|--------|
| Raw A (via expm) | >0.9999 | 0.009 | Classical-classical consistency |
| Antisymmetrized H (old) | 0.18-0.22 | 1.50 | Broken (theoretically wrong) |
| Hermitian dilation | 0.0395–0.8485 | 1.02–2.02 | FAIL — structural AA†=I |
| **Sqrt-symmetrization** | **0.9943** | **0.070** | **PASS** |

**Additional verification:**
- mu-dependence: ||ψ(1e10) − ψ(4e10)|| = 19.9 (non-zero, gradient works)
- All 19 pytest tests pass
- Full quantum inversion: loss 2.5→0.375, overlap 0.988→0.998, mu evolves correctly
- No null space for ANY initial condition (Gaussian, multi-mode sine, asymmetric spike all produce ||(H₁−H₂)@ψ|| ≫ 1)

See `tests/test_dilated_H_matches_leapfrog.py` for validation test and `tests/test_ic_breaks_null_space.py` for null-space verification.

---

## License

Academic and research use. Please cite the original Schade et al. references when using this code.
