# 1-D Seismic Inversion Using Quantum Computing Simulation with Sources and Loss Functions

A quantum-classical hybrid simulation for 1-D elastic wave propagation in heterogeneous media. The code combines a classical finite-difference (leapfrog) PDE solver with a Hamiltonian-based quantum time-evolution circuit built using Qiskit 2.x, following the framework of Schade et al.

---

## Acknowledgements & Literature

This work is built upon and references the following repositories and publications:

| Reference | Link |
|-----------|------|
| Schade, M. et al. (2024) — *Quantum Wave Equation Solver* | [github.com/malteschade/Quantum-Wave-Equation-Solver](https://github.com/malteschade/Quantum-Wave-Equation-Solver.git) |
| Schade, M. et al. (2024) — *Quantum Wave Simulation with Sources and Loss Functions* | [github.com/malteschade/Quantum-Wave-Simulation-with-Sources-and-Loss-Functions](https://github.com/malteschade/Quantum-Wave-Simulation-with-Sources-and-Loss-Functions.git) |
| Schade, M. et al. (2023) — arXiv preprint | [arXiv:2312.14747](https://arxiv.org/abs/2312.14747) |

## Author

**Najlah Rupaidah** (NIM 1227030025)
Geophysics Specialization, Department of Physics, Faculty of Science and Technology
Universitas Islam Negeri Sunan Gunung Djati, Bandung, Indonesia

Co-author: **bex**

---

## Features

- **Classical 1-D wave solver** — finite-difference leapfrog scheme supporting heterogeneous density and elastic modulus (Dirichlet / Neumann BCs).
- **Hamiltonian-based quantum circuit** — matches the structure of Schade et al. Fig. A1:
  - State Preparation (R_Y, X, Z gates)
  - Time Evolution (exp(-iHt) unitary block)
  - Observable (Pauli basis rotation via H gates)
  - Measurement (all qubits)
- **Quantum reconstruction** — simulates exact (statevector), shot-noise, and hardware-noise quantum reconstructions.
- **Source visualization** — Gaussian or Ricker wavelet source amplitude plotted against spatial position and time evolution, selectable at runtime. Note: the external source term peaks at t = 0.5 s, which falls outside the simulation window (t_max ≈ 0.2 s); the source plots are generated for visualization purposes while the wave solver is driven by the initial-condition spike.
- **Seismic inversion objective** — misfit MSE between current-model wavefields and fixed reference wavefields from the true model, averaged over all time steps.
- **Iterative optimization framework** — gradient-based elastic modulus recovery using Adam optimizer:
  - Reference fields computed once from the true model before the loop begins
  - Forward simulation → misfit evaluation → gradient descent → model update
  - Deterministic (noiseless) reconstruction for clean gradient signal
  - Best-model checkpoint saved throughout optimization
  - Moving-average early stopping robust to stochastic fluctuations
  - Gradient averaging over multiple evaluations (configurable)
  - Divergence detection with automatic revert to best model
  - Per-iteration diagnostics: loss, Δloss, gradient statistics, overlap, μ statistics
- **Publication-quality plots** — forward simulation multiplot, energy, overlap, error, loss, model evolution, source, and circuit diagram styled to match the reference.
- **Data export** — JSON config, pickle results, and Excel workbook with 9+ sheets.
- **Modular architecture** — functions are separated into domain-specific modules under `src/`.

## Requirements

| Package | Version |
|---------|---------|
| Python | >= 3.10 |
| Qiskit | >= 2.0 |
| qiskit-aer | >= 0.15 |
| numpy | >= 1.23 |
| scipy | >= 1.9 |
| matplotlib | >= 3.6 |
| openpyxl | >= 3.1 |
| pylatexenc | >= 2.10 |

Install all dependencies:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib openpyxl pylatexenc
```

## Project Structure

```
TA_mein-lieben/
├── main.py                       # Main simulation entry point
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── figures/                      # Generated plots
├── data/                         # Experiment output (JSON + pickle + Excel)
├── venv-quantum/                 # Python virtual environment
├── docs/                         # Documentation
└── src/                          # Modular source code
    ├── constants/                # Plot settings, ENUMS, directory paths
    │   └── __init__.py
    ├── distributions/            # Raised cosine, spike, homogeneous functions
    │   └── __init__.py
    ├── encoding/                 # Amplitude encoding & quantum reconstruction
    │   └── __init__.py
    ├── hamiltonian/              # Hermitian Hamiltonian construction for wave equation
    │   └── __init__.py
    ├── circuit/                  # Quantum circuit builder (Schade et al. style)
    │   └── __init__.py
    ├── execution/                # Circuit execution on AerSimulator
    │   └── __init__.py
    ├── wave/                     # Classical 1-D elastic wave solver (leapfrog)
    │   └── __init__.py
    ├── experiment/               # Experiment runner (orchestrates simulation)
    │   └── __init__.py
    ├── persistence/              # Data saving (JSON, pickle, Excel)
    │   └── __init__.py
    ├── visualization/            # Plotting utilities (forward sim, energy, overlap, error,
    │   └── __init__.py           #   circuit, source, source_time, loss, model update,
    │                             #   loss history, model evolution)
    └── optimization/             # Iterative optimization framework
        ├── __init__.py           # Module exports
        ├── objective.py          # SeismicObjective: forward sim + misfit loss vs reference
        ├── gradient.py           # FiniteDifferenceGradient: central FD gradient
        ├── optimizer.py          # AdamOptimizer + SeismicOptimizer (descent, diagnostics)
        └── callbacks.py          # Logging, loss history, convergence reports
```

## Module Overview

| Module | Responsibility |
|--------|---------------|
| `src.constants` | Global constants, plot style configuration |
| `src.distributions` | Medium property generators (raised cosine, spike, homogeneous) |
| `src.encoding` | Amplitude encoding/decoding, quantum state reconstruction |
| `src.hamiltonian` | Build Hermitian H from elastic wave equation for unitary time evolution |
| `src.circuit` | Build quantum circuit with state prep, exp(-iHt), observable, measurement |
| `src.execution` | Run transpiled circuit on AerSimulator |
| `src.wave` | Classical 1-D leapfrog finite-difference solver, Gaussian and Ricker wavelet source functions, energy computation |
| `src.experiment` | Orchestrate full experiment: set up medium, run PDE, compute metrics. Overlap computed against reference (true-model) fields when available |
| `src.persistence` | Save/export results to JSON, pickle, and Excel |
| `src.visualization` | Generate all publication-quality figures (forward sim, energy, overlap, error, circuit, source, loss, model update, loss history, model evolution) |
| `src.optimization` | Iterative inversion framework: misfit objective, FD gradient, Adam optimizer with descent sign, best-model checkpoint, moving-average early stopping |

## Usage

Run the full simulation pipeline:

```bash
python main.py
```

At startup, you will be prompted to select the source waveform:

```
Select source waveform:
  [a] Gaussian source
  [b] Ricker wavelet source
Enter choice (a/b):
```

Press **a** for a Gaussian source or **b** for a Ricker wavelet source. The selected waveform is used for visualization (source amplitude vs position and vs time plots). The wave solver is driven by an initial-condition spike independent of this choice.

The script then executes the following steps:

1. Initialize optimization framework (objective, gradient, optimizer)
2. Compute reference wavefields from the true elastic model (held fixed for all iterations)
3. Run initial forward simulation with the starting model
4. Build quantum circuit (Group 0, Index 10)
5. Run iterative inversion (forward sim → misfit → gradient → Adam update → best-model checkpoint)
6. Save experiment data (JSON + pickle)
7. Save Excel workbook (9+ sheets)
8. Generate source plots (selected source amplitude vs position and vs time)
9. Generate optimization plots (loss history, model evolution)
10. Generate forward simulation and analysis plots (forward sim, energy, overlap, error, model update)

## Simulation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid points (nx) | 7 | Interior points; total grid size nx+2 with ghost nodes |
| Grid spacing (dx) | 63 m | |
| Time step (dt) | 0.005 s | CFL ratio ≈ 0.35; wave travels ~22 m/step |
| Steps | 40 | Wave traverses full grid and reflects |
| Boundary conditions | Dirichlet | Zero displacement at both ends |
| True model (mu_true) | raised-cosine, 1–4 × 10¹⁰ Pa | Heterogeneous elastic modulus |
| Initial model (mu_initial) | homogeneous, 50% of mean(mu_true) | Deliberately wrong starting point |
| Density (rho) | raised-cosine, 2–4 × 10³ kg/m³ | Held fixed during inversion |

### CFL Stability Condition

```
dt ≤ dx / v_max
v_max = sqrt(mu_max / rho_min) = sqrt(4e10 / 2e3) ≈ 4472 m/s
dt_CFL = 63 / 4472 ≈ 0.014 s
dt used = 0.005 s  (CFL ratio = 0.35, well within stability)
```

## Quantum Circuit Structure

The circuit follows the Hamiltonian simulation approach from Schade et al.:

```
         ┌──────────┐ ░ ┌──────────────┐ ░ ┌───┐ ░ ┌─┐
  q_0: ──┤ R_Y(-θ)  ├─░─┤              ├─░─┤ H ├─░─┤M├───
         └──────────┘ ░ │              │ ░ └───┘ ░ └╥┘┌─┐
  q_1: ───────────────░─┤  exp(-iHt)   ├─░───────░──╫─┤M├─
         ┌───┐┌───┐  ░ │              │ ░ ┌───┐ ░  ║ └╥┘┌─┐
  q_2: ──┤ X ├┤ Z ├──░─┤              ├─░─┤ H ├─░──╫──╫─┤M├
         └───┘└───┘  ░ │              │ ░ └───┘ ░  ║  ║ └╥┘┌─┐
  q_3: ───────────────░─┤              ├─░───────░──╫──╫──╫─┤M├
                      ░ └──────────────┘ ░       ░  ║  ║  ║ └╥┘
  c: 4/═════════════════════════════════════════════╩══╩══╩══╩═
```

| Section | Purpose |
|---------|---------|
| **State Preparation** | Encode initial conditions (displacement + velocity) into qubit amplitudes via R_Y, X, Z gates |
| **Time Evolution** | Apply Hamiltonian exp(-iHt) as a unitary block, where H is derived from the 1-D elastic wave equation |
| **Observable** | Rotate into Pauli measurement basis (XZXZ) using Hadamard gates |
| **Measurement** | Measure all qubits in computational basis |

## Excel Output

The `results.xlsx` workbook contains 9+ sheets:

| Sheet | Contents |
|-------|----------|
| Configuration | All experiment parameters |
| Medium | Grid position, density (rho), elastic modulus (mu) |
| TimeSeries | Time step, time, energy |
| Overlaps | Time, squared quantum overlap against reference fields |
| WaveFields | Full displacement field at every time step |
| CircuitParams | Qubit count, Hilbert dimension, evolution time, observable |
| Source | Source amplitude at each grid position (Gaussian or Ricker wavelet) |
| Loss | Reconstruction MSE loss per time step |
| ModelUpdate | Initial and updated elastic modulus (mu) |
| OptimizationHistory | Loss history and model evolution per iteration |

## Optimization Framework

The seismic inversion recovers the elastic modulus (mu) by minimizing the misfit between forward-simulated wavefields and fixed reference wavefields from the true model.

### Algorithm

```
# Step 0: compute reference fields ONCE from the true model
ref_fields = forward_simulate(mu_true, rho, u0, v0)

for iteration in range(max_iterations):
    # Forward simulation with current model
    fields = forward_simulate(mu_current, rho, u0, v0)

    # Misfit: compare current fields against fixed reference
    loss = (1/T) * sum_t mean_x( (u_fwd(t; mu) - u_ref(t))^2 )

    # Gradient via central finite differences (single call)
    gradient = [J(mu + delta*e_i) - J(mu - delta*e_i)] / (2*delta)

    # Adam update: SUBTRACT delta (gradient descent)
    mu_current = adam_step(mu_current, gradient)   # theta -= alpha * m_hat / sqrt(v_hat)

    # Best-model checkpoint
    if loss < best_loss:
        mu_best = mu_current.copy()

    # Moving-average early stopping
    if ma_improvement < tolerance for patience iterations:
        break
```

### Mathematical Formulation

Objective function (misfit between current model and true model):

```
J(m) = (1/T) * sum_t (1/N) * ||u_fwd(t; m) - u_ref(t; m_true)||^2
```

where `u_ref` is computed once from the true model `m_true` and held fixed.

Adam parameter update (gradient **descent**):

```
m_t  = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t  = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m̂_t = m_t  / (1 - beta1^t)
v̂_t = v_t  / (1 - beta2^t)
theta_{t+1} = theta_t - alpha * m̂_t / (sqrt(v̂_t) + epsilon)
```

Finite-difference gradient with adaptive step size:

```
dJ/dm_i = [J(m + delta_i * e_i) - J(m - delta_i * e_i)] / (2 * delta_i)
delta_i  = max(delta_scale * |m_i|,  epsilon_abs)
           with delta_scale = 1e-4,  epsilon_abs = 1.0 Pa
```

Overlap metric (inversion quality, approaches 1 as mu → mu_true):

```
overlap(t) = |<psi_fwd(t; mu_current) | psi_ref(t; mu_true)>|^2
```

### Why Finite-Difference Gradient?

1. **Analytical gradients** would require the adjoint-state method combined with the chain rule through quantum amplitude encoding — a research-level derivation not yet implemented.
2. **Adam** handles the small, consistent gradients produced by this problem (gradients ~ 1e-12 J/Pa for mu ~ 1e10 Pa) through its per-parameter adaptive scaling.
3. **Deterministic reconstruction** (`use_deterministic=True`) eliminates shot noise from gradient evaluations, producing clean monotonic loss decrease.

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 100 | Maximum optimization iterations |
| `learning_rate` | 5e9 Pa | Adam learning rate (scaled to mu ~ 1e10 Pa) |
| `convergence_tolerance` | 1e-12 | MA-loss improvement threshold for early stop |
| `early_stopping_patience` | 25 | Consecutive non-improving MA windows before stop |
| `delta_scale` | 1e-4 | Relative FD step size: delta_i = delta_scale × \|mu_i\| |
| `epsilon_abs` | 1.0 Pa | Absolute minimum FD step (protects near-zero mu) |
| `use_deterministic` | True | Use noiseless quantum reconstruction for gradients |
| `n_grad_avg` | 1 | Number of gradient evaluations to average (reduces shot noise) |
| `ma_window` | 5 | Window length for moving-average early stopping |

---

## Bug Fixes and Scientific Corrections (V&V Audit)

The following bugs were identified and corrected in the optimization pipeline. All fixes are documented in source code comments.

### Bug 1 — Trivial objective (critical)

**File:** `src/optimization/objective.py`

**Before:** The loss compared each field against its own quantum reconstruction:

```python
J = mean_t ||u_classical(t; mu) - quantum_decode(quantum_encode(u_classical(t; mu)))||^2
```

Because amplitude encoding is lossless in the deterministic case, `u_quantum ≈ u_classical` for any value of mu. The loss was identically near zero and its gradient with respect to mu was numerically zero. The optimizer had no signal to follow.

**After:** The loss compares the current-model wavefield against fixed reference fields from the true model:

```python
J(mu) = mean_t ||u_fwd(t; mu) - u_ref(t; mu_true)||^2
```

### Bug 2 — Adam gradient ascent (critical)

**File:** `src/optimization/optimizer.py`, `AdamOptimizer.step()`

**Before:** `new_theta = theta + delta` — moves parameters in the direction of *increasing* loss (maximization).

**After:** `new_theta = theta - delta` — correct gradient descent.

### Bug 3 — Double gradient computation per iteration

**File:** `src/optimization/optimizer.py`, `run_optimization()`

**Before:** `compute()` was called first, then immediately overwritten by `compute_with_regularization()`. The first call's result was discarded, wasting 2× the forward-simulation budget per iteration.

**After:** A single call to `compute_with_regularization()` inside `_compute_gradient()`.

### Bug 4 — Stale gradient norm in logs

**File:** `src/optimization/optimizer.py`

**Before:** The logged gradient norm was taken from the discarded first call (Bug 3). The log showed an identical norm for all 50 iterations.

**After:** Gradient norm is computed from the single authoritative gradient used for the Adam step.

### Bug 5 — Trivial zero loss in callback

**File:** `src/optimization/callbacks.py`, `LossHistoryCallback`

**Before:**

```python
loss_ts[i] = np.mean((u_classical - u_classical) ** 2)  # always 0
```

**After:** The callback stores the misfit loss value passed directly from the optimizer. The per-timestep breakdown uses `(u_classical - u_quantum)^2` as a diagnostic.

### Bug 6 — Overlap measures IC decay, not inversion progress

**File:** `src/experiment/__init__.py`

**Before:** `ref_sv = amplitude_encode(fields[0])` — overlap was always `|⟨ψ(0)|ψ(t)⟩|²`, which measures how much the wavefield has dispersed from the initial spike. This monotonically decreases as the wave propagates and cannot rise during inversion.

**After:** When `reference_fields` are provided, overlap is computed as `|⟨ψ_fwd(t;mu)|ψ_ref(t;mu_true)⟩|²`. This starts below 1 for a wrong initial model and increases toward 1 as the optimizer recovers the true elastic modulus.

### Bug 7 — Finite-difference step size floor too small

**File:** `src/optimization/gradient.py`

**Before:** `epsilon = 1e-8` Pa as the absolute minimum step. For mu ~ 1e10 Pa, a step of 1e-8 Pa causes catastrophic cancellation in double-precision arithmetic.

**After:** `epsilon = 1.0` Pa. The adaptive term `delta_scale × |mu_i| ≈ 1e6 Pa` dominates for realistic mu values; the floor only activates near zero.

### Bug 8 — Inconsistent physics between optimizer and experiment

**File:** `src/optimization/objective.py`, `forward_simulate()`

**Before:** `source_func=None` was hardcoded, while `run_experiment_1d()` used a Gaussian/Ricker source. The optimizer minimized misfit between source-free waves and sourced reference fields — physically inconsistent PDEs.

**After:** `SeismicObjective` stores `source_func` at construction and passes it to every `evolve_1d_wave` call, ensuring the optimizer and experiment use identical physics.

### Bug 9 — Zero wave propagation (dt too small)

**File:** `main.py`

**Before:** `dt = 1e-6 s`, `steps = 19`. Total simulation time = 19 μs. At v_max ≈ 4472 m/s and dx = 63 m, the wave moved only 0.085 m in 19 steps — essentially zero propagation. All fields were identical regardless of mu, giving a loss of ~ 3.6 × 10⁻¹⁵ (machine precision).

**After:** `dt = 0.005 s`, `steps = 40`. CFL ratio = 0.35 (stable). The wave traverses the full 567 m grid and reflects, giving fields that differ measurably between models.

### Bug 10 — Initial model identical to true model in effect

**File:** `main.py`

**Before:** `mu_initial = homogeneous(mean(mu_true))`. With near-zero propagation (Bug 9), fields from the mean model were indistinguishable from the true model fields.

**After:** `mu_initial = homogeneous(0.5 × mean(mu_true))`. The starting wave speed is 71% of the true mean wave speed, producing an initial misfit of ~0.062 and real gradient signal throughout the grid.

---

## Remaining Limitations

| Limitation | Description |
|------------|-------------|
| **Finite-difference cost** | Gradient requires 2 × N_params forward simulations per iteration. For nx = 7 this is 16 sims/iter; for nx = 63 it becomes 128 sims/iter. The adjoint-state method would reduce this to 2 sims/iter regardless of grid size (`compute_gradient_adjoint` is a placeholder). |
| **Shot noise on gradients** | With stochastic quantum reconstruction (shots < ∞), each FD evaluation adds measurement noise. Setting `use_deterministic=True` removes this; alternatively increase `n_grad_avg` for averaging. |
| **Underdetermined inversion** | nx = 7 with 40 time steps is severely underdetermined. Adding Tikhonov regularization (`reg_weight > 0`) with a smooth prior improves stability for larger grids. |
| **Slow convergence for large mu gap** | Adam moves ~lr Pa/step. Moving from 11.6 GPa to 23.1 GPa (11.5 GPa gap) at lr = 5 × 10⁹ Pa/step requires ~20+ iterations per parameter. Increasing lr or running more iterations recovers more of the true model. |
| **Source term inactive during simulation** | The Gaussian/Ricker source peaks at t = 0.5 s. At t_max = 0.2 s the source amplitude is < 10⁻¹⁷. Source plots are generated for visualization but the source does not drive the wave physics. A future version should shift the source peak inside the simulation window. |

## License

This project is for academic and research purposes. Please cite the original references when using this code.
