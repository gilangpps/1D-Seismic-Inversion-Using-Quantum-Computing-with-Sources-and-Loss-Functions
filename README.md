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
- **Publication-quality plots** — forward simulation multiplot, energy, overlap, error, and circuit diagram styled to match the reference.
- **Data export** — JSON config, pickle results, and Excel workbook with multiple sheets.

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
101225_malte_schade_integration/
├── main.py                  # Main simulation script
├── README.md                # This file
├── penjelasan_kode.txt      # Simple code explanation (Bahasa Indonesia)
├── figures/
│   ├── forward_sim.png      # 2x3 multiplot: medium properties + wave snapshots
│   ├── energy.png           # Total energy vs time
│   ├── overlap.png          # Quantum state overlap with initial condition
│   ├── error.png            # Relative L2 reconstruction error
│   └── circuit.png          # Quantum circuit diagram (Fig. A1 style)
└── data/
    └── <timestamp>/
        ├── configs.json     # Experiment parameters
        ├── data.pkl         # Simulation results (pickle)
        └── results.xlsx     # Simulation results (Excel, 6 sheets)
```

## Usage

Run the full simulation pipeline:

```bash
python main.py
```

This executes 7 steps:

1. Classical wave simulation (leapfrog finite-difference)
2. Build quantum circuit (Group 0, Index 10)
3. Save experiment data (JSON + pickle)
4. Save Excel workbook (6 sheets)
5. Generate forward simulation multiplot
6. Generate analysis plots (energy, overlap, error)
7. Generate circuit diagram

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

The `results.xlsx` workbook contains 6 sheets:

| Sheet | Contents |
|-------|----------|
| Configuration | All experiment parameters |
| Medium | Grid position, density (rho), elastic modulus (mu) |
| TimeSeries | Time step, time, energy |
| Overlaps | Time, squared quantum overlap |
| WaveFields | Full displacement field at every time step |
| CircuitParams | Qubit count, Hilbert dimension, evolution time, observable |

## License

This project is for academic and research purposes. Please cite the original references when using this code.
