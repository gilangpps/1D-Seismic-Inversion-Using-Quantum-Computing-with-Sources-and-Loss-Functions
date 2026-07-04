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
  figures/forward_sim.png          — multiplot: medium + wave snapshots (2x3)
  figures/energy.png               — energy vs time
  figures/overlap.png              — quantum overlap vs time
  figures/error.png                — relative L2 reconstruction error
  figures/circuit.png              — quantum circuit diagram
  figures/loss_history.png          — loss vs iteration (optimization)
  figures/model_evolution.png       — mu parameters evolution
  figures/convergence_report.png   — convergence statistics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.constants import configure_plot_style
from src.distributions import raised_cosine, homogeneous
from src.wave import gaussian_source, ricker_wavelet_source
from src.encoding import quantum_reconstruct
from src.experiment import run_experiment_1d
from src.circuit import build_paper_circuit
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
from src.optimization import (
    SeismicObjective,
    FiniteDifferenceGradient,
    SeismicOptimizer,
    OptimizationLogger,
    LossHistoryCallback,
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

    nx = 7
# Suggested:
#
#   nx = [
#       7,
#       15,
#       31,
#       63
#   ]

    config = {
        'nx': nx,
        'dx': 63,
        'dt': 0.000001,
        'steps': 19,
        'bc': 'dirichlet',
        'bcs': {'left': 'DBC', 'right': 'DBC'},
        'shots': 1000,
        'measure_every': 4,
    }

    mu_arr = raised_cosine(3e10, nx + 1, nx, 6, 1e10)
    rho_arr = raised_cosine(2e3, nx, nx - 1, 6, 2e3)

    u0 = np.zeros(nx)
    u0[nx // 2 + 1] = 1.0
    u0[nx // 2 + 2] = -1.1
    v0 = homogeneous(0, nx)

    config_save = {
        'nx': nx, 'dx': config['dx'], 'dt': config['dt'],
        'steps': config['steps'], 'bc': config['bc'],
        'shots': config['shots'], 'measure_every': config['measure_every'],
        'mu': mu_arr.tolist(), 'rho': rho_arr.tolist(), 'u0': u0.tolist(),
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

    print("\n[1/10] Creating optimization framework...")

    # Initialize optimization components
    configs_for_optim = {
        'nx': nx, 'dx': config['dx'], 'dt': config['dt'],
        'steps': config['steps'], 'bc': config['bc'],
        'shots': config['shots'], 'measure_every': config['measure_every'],
        'mu': mu_arr.tolist(), 'rho': rho_arr.tolist(), 'u0': u0.tolist(),
        'v0': v0.tolist(),
    }

    objective = SeismicObjective(
        nx=nx, dx=config['dx'], dt=config['dt'],
        steps=config['steps'], measure_every=config['measure_every'],
        shots=config['shots'], bc=config['bc'], seed=42,
    )

    gradient = FiniteDifferenceGradient(
        objective_fn=None,  # Will be set by optimizer
        delta_scale=1e-4,
        epsilon=1e-8,
    )

    loss_callback = LossHistoryCallback()
    logger = OptimizationLogger()

    print("  ✓ Optimization framework initialized")

    print("\n[2/10] Running initial forward simulation...")
    fields, results, x = run_experiment_1d(
        nx=nx, dx=config['dx'], dt=config['dt'], steps=config['steps'],
        mu_arr=mu_arr, rho_arr=rho_arr, u0=u0, v0=v0,
        measure_every=config['measure_every'], shots=config['shots'],
        bc=config['bc'],
    )
    print(f"       {len(fields)} time steps computed.")

    circuit_time_idx = 10
    print(f"\n[3/10] Building quantum circuit (Group 0, Index {circuit_time_idx})...")
    qc_paper, circuit_meta = build_paper_circuit(
        u0, v0, mu_arr, rho_arr,
        dx=config['dx'], dt=config['dt'],
        time_step_idx=circuit_time_idx, nx=nx,
        observable=['X', 'Z', 'X', 'Z'],
        group_idx=0,
    )

    np.random.seed(42)
    center_idx = nx // 2 + 1
    width_idx = 2
    amplitude = 1.0
    waveform = source_func(center_idx, width_idx, amplitude, nx + 2)
    source_amp = np.array([waveform(i, 0.5) for i in range(nx + 2)])
    results['source_amplitude'] = source_amp
    results['source_name'] = source_name

    t_vals = np.linspace(0, 1.0, 10000)
    source_time = np.array([waveform(center_idx, t) for t in t_vals])

    loss_arr = []
    for i in range(1, len(fields)):
        qr = quantum_reconstruct(fields[i], shots=config['shots'])
        u_classical = fields[i][1:-1]
        u_quantum = qr[1:-1]
        loss_val = np.mean((u_classical - u_quantum) ** 2)
        loss_arr.append(loss_val)
    results['loss'] = loss_arr

    mean_loss = np.mean(loss_arr)

    print("\n[4/10] Initializing optimization process...")

    optimizer = SeismicOptimizer(
        configs=configs_for_optim,
        objective=objective,
        gradient=gradient,
        loss_history_callback=loss_callback,
        logger=logger,
        max_iterations=50,
        convergence_tolerance=1e-6,
        early_stopping_patience=15,
        learning_rate=0.01,
        use_deterministic=False,
    )

    print("  ✓ Optimizer initialized with iterative optimization framework")

    print("\n[5/10] Running iterative inversion optimization...")
    opt_results = optimizer.run_optimization()
    print(f"       Completed {opt_results['num_iterations']} iterations")
    print(f"       Final loss: {opt_results['final_loss']:.6e}")
    print(f"       Loss reduction: {(1 - opt_results['final_loss']/opt_results['convergence_report']['initial_loss'])*100:.2f}%")

    mu_final = opt_results['mu_final']
    results['mu_final'] = mu_final.tolist()
    results['optimization_history'] = opt_results

    # Add keys for backward-compatible model update plot
    mu_initial = np.array(configs_for_optim['mu'])
    results['mu_initial'] = mu_initial.tolist()
    results['mu_updated'] = mu_final.tolist()

    print("\n[6/10] Saving experiment data...")
    exp_dir = save_experiment(fields, results, x, config_save)

    print("\n[7/10] Saving Excel workbook...")
    save_to_excel(fields, results, x, config_save, circuit_meta, exp_dir)

    print("\n[8/10] Generating source plots...")
    plot_source(source_amp, x, source_name=source_name)
    plot_source_time(t_vals, source_time, source_name=source_name)

    print("\n[9/10] Generating optimization analysis plots...")

    # Loss history plot
    from src.visualization import plot_loss_history, plot_model_evolution
    try:
        plot_loss_history(opt_results)
        plot_model_evolution(opt_results)
    except Exception as e:
        print(f"    Warning: Could not generate optimization plots: {e}")

    print("\n[10/10] Generating forward simulation and analysis plots...")
    plot_forward_sim(fields, results, x, config)
    plot_energy(results)
    plot_overlap(results)
    plot_error(fields, results, config)
    plot_model_update(results)

    print("\nAll figures saved to figures/")
    print("All data saved to data/")
    print("Done.")