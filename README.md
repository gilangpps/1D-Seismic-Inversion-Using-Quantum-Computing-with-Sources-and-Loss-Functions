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
- **Source injection** — Ricker or Gaussian wavelet enters the PDE as a forcing term f(x,t) (Requirement B). Source peak placed at t₀ = t_max/3, inside the simulation window.
- **Hamiltonian-based quantum circuit** — matches Schade et al. Fig. A.1: State Preparation → exp(−iHt) → Observable → Measurement.
- **Physical Hamiltonian** — H = i·(A − Aᵀ)/2, explicitly dependent on μ, ρ, dx. Hermiticity guaranteed by construction.
- **Quantum amplitude encoding** — |ψ⟩ = (1/‖u‖) Σᵢ u[i]|i⟩, three-mode reconstruction (statevector / shot-noise / hardware-noise).
- **Seismic inversion** — iterative optimization: J(μ) = (1/T) Σ_t ‖u_fwd(t;μ) − u_ref(t;μ_true)‖². Gradient via central finite differences. Adam optimizer (gradient descent).
- **10 mandatory publication plots** — initial μ, density, source, wave propagation, quantum reconstruction error, observed vs predicted, loss convergence, μ inversion result, classical PDE energy, quantum overlap.
- **Excel output** — 10 sheets including OptimizationHistory (loss, overlap, μ per iteration).

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
| dt | 0.005 s | CFL ratio ≈ 0.35 |
| steps | 40 | t_max = 0.2 s |
| μ_true | raised-cosine, 1–4 × 10¹⁰ Pa | Heterogeneous |
| μ_initial | homogeneous, 50% of mean(μ_true) | Wrong starting point |
| ρ | raised-cosine, 2–4 × 10³ kg/m³ | Fixed during inversion |
| Source t₀ | t_max/3 ≈ 0.067 s | Inside simulation window |
| Source σ_t | t_max/12 ≈ 0.017 s | Well-resolved by dt |

CFL stability:
```
v_max = sqrt(4e10 / 2e3) ≈ 4472 m/s
dt_CFL = 63 / 4472 ≈ 0.014 s   →  dt = 0.005 s  (ratio 0.35) ✓
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

```
       ┌──────────────┐ ░ ┌──────────────┐ ░ ┌───┐ ░ ┌─┐
q_0: ──┤ State Prep   ├─░─┤              ├─░─┤ H ├─░─┤M├
       └──────────────┘ ░ │  exp(-iHt)   │ ░ └───┘ ░ └─┘
q_1: ──────────────────░─┤              ├─░────────░──M──
       ┌───┐┌───┐      ░ │              │ ░ ┌───┐  ░ ┌─┐
q_2: ──┤ X ├┤ Z ├──────░─┤              ├─░─┤ H ├─░─┤M├
       └───┘└───┘      ░ └──────────────┘ ░ └───┘ ░ └─┘
```

H = i·(A − Aᵀ)/2, where A is the first-order elastic system matrix.  
H† = H (Hermitian) → exp(−iHt) is unitary.

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

## Excel Output (10 sheets)

| Sheet | Contents |
|-------|----------|
| Configuration | All experiment parameters |
| Medium | x, ρ(x), μ(x) |
| TimeSeries | time step, time, classical PDE energy |
| Overlaps | time, \|⟨ψ_ref\|ψ_fwd⟩\|² |
| WaveFields | Full displacement field at every time step |
| CircuitParams | Qubit count, dim, evolution time, observable |
| Source | Source amplitude at each grid point |
| Loss | Per-timestep quantum reconstruction MSE |
| ModelUpdate | μ_initial, μ_updated per grid point |
| OptimizationHistory | iteration, loss, mean_overlap, μ[0..n] |

---

## Figures Generated (15 total)

| File | Description |
|------|-------------|
| `mu_initial.png` | Initial vs true elastic modulus |
| `density_model.png` | Density model ρ(x) |
| `source.png` | Source amplitude vs position |
| `source_time.png` | Source wavelet vs time |
| `forward_sim.png` | Wave propagation snapshots (2×3 multiplot) |
| `energy.png` | **Classical PDE energy** vs time |
| `overlap.png` | Quantum state overlap vs time |
| `error.png` | Quantum reconstruction relative L2 error |
| `observed_vs_predicted.png` | Observed vs predicted seismic trace + residual |
| `mu_inversion.png` | μ_true vs μ_initial vs μ_recovered |
| `loss_history.png` | Misfit J(μ) convergence vs iteration |
| `model_evolution.png` | μ parameter evolution across iterations |
| `model_update.png` | Bar chart: μ_initial vs μ_updated |
| `loss.png` | Per-timestep reconstruction loss |
| `circuit.png` | Quantum circuit diagram |

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
- [x] Loss drives inversion (J(μ) = ‖u_fwd − u_ref‖²)
- [x] μ updates iteratively (Adam optimizer, 100 iterations)
- [x] Reconstruction works (3-mode: statevector, shot-noise, hardware-noise)
- [x] Overlap has physical meaning (inversion quality ∈ [0,1])
- [x] Energy correctly labeled **Classical PDE Energy** [J/m]
- [x] Excel has 10 sheets including OptimizationHistory
- [x] 10 mandatory visualization plots generated

---

## License

Academic and research use. Please cite the original Schade et al. references when using this code.
