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
- **Source visualization** — Gaussian or Ricker wavelet source amplitude plotted against spatial position and time evolution, selectable at runtime.
- **Loss function analysis** — inversion-style MSE loss between classical and quantum fields at every timestep.
- **Model update (proof of concept)** — iterative elastic modulus update driven by mean reconstruction loss.
- **Publication-quality plots** — forward simulation multiplot, energy, overlap, error, loss, source (position & time), model update, and circuit diagram styled to match the reference.
- **Data export** — JSON config, pickle results, and Excel workbook with 9 sheets.
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
    └── visualization/            # Plotting utilities (forward sim, energy, overlap, error, circuit, source, source_time, loss, model update)
        └── __init__.py
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
| `src.experiment` | Orchestrate full experiment: set up medium, run PDE, compute metrics |
| `src.persistence` | Save/export results to JSON, pickle, and Excel |
| `src.visualization` | Generate all publication-quality figures (forward sim, energy, overlap, error, circuit, source, loss, model update) |

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

Press **a** for a Gaussian source or **b** for a Ricker wavelet source. The selected waveform is then used throughout the simulation and plot generation.

The script then executes 10 steps:

1. Classical wave simulation (leapfrog finite-difference)
2. Build quantum circuit (Group 0, Index 10)
3. Save experiment data (JSON + pickle)
4. Save Excel workbook (9 sheets)
5. Generate source plots (selected source amplitude vs position and vs time)
6. Generate loss analysis (MSE between classical and quantum fields)
7. Run model update (elastic modulus correction via mean loss)
8. Generate forward simulation multiplot
9. Generate analysis plots (energy, overlap, error)
10. Generate circuit diagram

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

The `results.xlsx` workbook contains 9 sheets:

| Sheet | Contents |
|-------|----------|
| Configuration | All experiment parameters |
| Medium | Grid position, density (rho), elastic modulus (mu) |
| TimeSeries | Time step, time, energy |
| Overlaps | Time, squared quantum overlap |
| WaveFields | Full displacement field at every time step |
| CircuitParams | Qubit count, Hilbert dimension, evolution time, observable |
| Source | Source amplitude at each grid position (Gaussian or Ricker wavelet) |
| Loss | Reconstruction MSE loss per time step |
| ModelUpdate | Initial and updated elastic modulus (mu) |

## License

This project is for academic and research purposes. Please cite the original references when using this code.
