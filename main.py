"""
1-D Seismic Inversion Using Quantum Computing Simulation with Sources and Loss Functions
Reference to paper: Schade et al., arXiv:2312.14747 (2023).

Main repository literature can be found at:
Schade, et al. (2024); https://github.com/malteschade/Quantum-Wave-Equation-Solver.git
Schade, et al. (2024); https://github.com/malteschade/Quantum-Wave-Simulation-with-Sources-and-Loss-Functions.git

Compiled by: Najlah Rupaidah (NIM 1227030025)
Geophysics Specialization, Department of Physics, Faculty of Science and Technology
Universitas Islam Negeri Sunan Gunung Djati, Bandung, Indonesia

co-author: bex

Output:
   data/<timestamp>/configs.json    — experiment parameters
   data/<timestamp>/data.pkl        — simulation results (pickle)
   data/<timestamp>/results.xlsx    — simulation results (Excel)
   figures/forward_sim.png          — medium + wave snapshots multiplot
   figures/energy.png               — total energy vs time
   figures/overlap.png              — quantum state fidelity vs time
   figures/error.png                — relative L2 reconstruction error
   figures/circuit.png              — quantum circuit diagram
   figures/source.png               — source spatial distribution
   figures/source_time.png          — source temporal profile
   figures/loss.png                 — reconstruction loss (time) + inversion loss (iterations)
   figures/model_update.png         — model comparison true/initial/updated + difference
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from src.constants import configure_plot_style
from src.distributions import raised_cosine, homogeneous
from src.experiment import run_experiment_1d
from src.circuit import build_paper_circuit
from src.wave import gaussian_source, ricker_wavelet_source
from src.encoding import quantum_reconstruct
from src.persistence import save_experiment, save_to_excel
from src.visualization import (
    plot_forward_sim,
    plot_energy,
    plot_overlap,
    plot_error,
    plot_circuit,
    plot_source,
    plot_source_time,
    plot_loss,
    plot_model_update,
)

configure_plot_style()

if __name__ == "__main__":
    print("Select source waveform:")
    print("  [a] Gaussian source")
    print("  [b] Ricker wavelet source")
    choice = input("Enter choice (a/b): ").strip().lower()
    while choice not in ('a', 'b'):
        choice = input("Invalid input. Enter 'a' for Gaussian or 'b' for Ricker wavelet: ").strip().lower()
    source_name = 'Gaussian' if choice == 'a' else 'Ricker wavelet'
    source_func = gaussian_source if choice == 'a' else ricker_wavelet_source
    print(f"\nUsing {source_name} source.\n")

    n_iter_input = input("Enter number of inversion iterations (default 5): ").strip()
    n_iter = int(n_iter_input) if n_iter_input.isdigit() and int(n_iter_input) > 0 else 5

    nx = 7
    config = {
        'nx': nx,
        'dx': 1,
        'dt': 0.0001,
        'steps': 19,
        'bc': 'dirichlet',
        'bcs': {'left': 'DBC', 'right': 'DBC'},
        'shots': 1000,
        'measure_every': 4,
    }

    mu_true = raised_cosine(3e10, nx + 1, nx, 6, 1e10)
    rho_arr = raised_cosine(2e3, nx, nx - 1, 6, 2e3)

    u0 = np.zeros(nx)
    u0[nx // 2 + 1] = 1.0
    u0[nx // 2 + 2] = -1.1
    v0 = homogeneous(0, nx)

    config_save = {
        'nx': nx, 'dx': config['dx'], 'dt': config['dt'],
        'steps': config['steps'], 'bc': config['bc'],
        'shots': config['shots'], 'measure_every': config['measure_every'],
        'mu': mu_true.tolist(), 'rho': rho_arr.tolist(), 'u0': u0.tolist(),
    }

    print("=" * 65)
    print("  Quantum Wave Simulation - 1D Forward Experiment")
    print("  Ref: Schade et al., arXiv:2312.14747 (2023)")
    print("=" * 65)
    print(f"  Grid points (nx)     : {nx}")
    print(f"  Grid spacing (dx)    : {config['dx']}")
    print(f"  Time step (dt)       : {config['dt']}")
    print(f"  Evolution steps (nt) : {config['steps']}")
    print(f"  Boundary conditions  : {config['bcs']}")
    print(f"  Shots per circuit    : {config['shots']}")
    print("=" * 65)

    print("\n[1/10] Running classical wave simulation...")
    fields, results, x = run_experiment_1d(
        nx=nx, dx=config['dx'], dt=config['dt'], steps=config['steps'],
        mu_arr=mu_true, rho_arr=rho_arr, u0=u0, v0=v0,
        measure_every=config['measure_every'], shots=config['shots'],
        bc=config['bc'],
    )
    print(f"       {len(fields)} time steps computed.")

    circuit_time_idx = 10
    print(f"\n[2/10] Building quantum circuit (Group 0, Index {circuit_time_idx})...")
    qc_paper, circuit_meta = build_paper_circuit(
        u0, v0, mu_true, rho_arr,
        dx=config['dx'], dt=config['dt'],
        time_step_idx=circuit_time_idx, nx=nx,
        observable=['X', 'Z', 'X', 'Z'],
        group_idx=0,
    )
    print(f"       Qubits          : {circuit_meta['n_qubits']}")
    print(f"       Hilbert dim     : {circuit_meta['dim']}")
    print(f"       Evolution time  : {circuit_meta['time']:.6f} s")
    print(f"       Observable      : {circuit_meta['observable']}")

    np.random.seed(42)
    center_idx = nx // 2 + 1
    width_idx = 2
    amplitude = 1.0
    waveform = source_func(center_idx, width_idx, amplitude, nx + 2)
    source_amp = np.array([waveform(i, 0.5) for i in range(nx + 2)])
    results['source_amplitude'] = source_amp
    results['source_name'] = source_name

    t_vals = np.linspace(0, 1.0, 500)
    source_time = np.array([waveform(center_idx, t) for t in t_vals])

    np.random.seed(123)
    mu_initial = mu_true * (1 + 0.02 * (np.random.rand(len(mu_true)) - 0.5))
    mu_current = mu_initial.copy()
    loss_iterations = []
    
    print(f"\n[3/10] Running {n_iter} inversion iterations...")
    for iteration in range(n_iter):
        fields_iter, results_iter, x_iter = run_experiment_1d(
            nx=nx, dx=config['dx'], dt=config['dt'], steps=config['steps'],
            mu_arr=mu_current, rho_arr=rho_arr, u0=u0, v0=v0,
            measure_every=config['measure_every'], shots=config['shots'],
            bc=config['bc'],
        )
        loss_vals = []
        for i in range(1, len(fields_iter)):
            qr = quantum_reconstruct(fields_iter[i], shots=config['shots'])
            u_classical = fields_iter[i][1:-1]
            u_quantum = qr[1:-1]
            loss_vals.append(np.mean((u_classical - u_quantum) ** 2))
        mean_loss = np.mean(loss_vals)
        loss_iterations.append(mean_loss)
        mu_current = mu_current * (1 - 0.001 * mean_loss)
    
    results['mu_true'] = mu_true
    results['mu_initial'] = mu_initial
    results['mu_updated'] = mu_current
    results['loss'] = loss_vals
    results['loss_iterations'] = loss_iterations

    print("\n[4/10] Saving experiment data...")
    exp_dir = save_experiment(fields, results, x, config_save)

    print("\n[5/10] Saving Excel workbook...")
    save_to_excel(fields, results, x, config_save, circuit_meta, exp_dir)

    print("\n[6/10] Generating source plots...")
    plot_source(source_amp, x, source_name=source_name)
    plot_source_time(t_vals, source_time, source_name=source_name)

    print("\n[7/10] Generating loss analysis...")
    plot_loss(results)

    print("\n[8/10] Running model update...")
    plot_model_update(results)

    print("\n[9/10] Generating forward simulation plots...")
    plot_forward_sim(fields, results, x, config)

    print("\n[10/10] Generating analysis plots...")
    plot_energy(results)
    plot_overlap(results)
    plot_error(fields, results, config)

    print("\n[11/10] Generating circuit figure...")
    plot_circuit(qc_paper, circuit_meta)

    print("\nAll figures saved to figures/")
    print("All data saved to data/")
    print("Done.")
