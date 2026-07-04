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
    source_func_factory = gaussian_source if choice == 'a' else ricker_wavelet_source
    print(f"\nUsing {source_name} source.\n")

    nx = 7
    # Suggested grid sizes: 7, 15, 31, 63

    # ── Stability and propagation check ──────────────────────────────────
    # Leapfrog CFL condition: dt <= dx / v_max
    # With dx=63m and mu_max~4e10Pa, rho_min~2e3 kg/m³:
    #   v_max = sqrt(4e10/2e3) ≈ 4472 m/s  →  dt_max ≈ 0.014 s
    # We use dt=0.005s so the wave travels ~half the grid per step,
    # giving clearly visible propagation over 'steps' time steps.
    # The original dt=1e-6 s moved the wave only ~0.085 m in 19 steps
    # on a 63 m grid — effectively zero propagation, making all fields
    # identical regardless of mu.
    dx = 63.0
    dt = 0.005          # s  (well within CFL, meaningful propagation)
    steps = 40          # enough for wave to traverse grid and reflect

    config = {
        'nx': nx,
        'dx': dx,
        'dt': dt,
        'steps': steps,
        'bc': 'dirichlet',
        'bcs': {'left': 'DBC', 'right': 'DBC'},
        'shots': 1000,
        'measure_every': 4,
    }

    # ── Medium parameters ─────────────────────────────────────────────────
    # mu_true: heterogeneous raised-cosine profile (what inversion recovers)
    # mu_initial: homogeneous starting model — 50% of true mean, clearly wrong
    mu_true   = raised_cosine(3e10, nx + 1, nx, 6, 1e10)
    rho_arr   = raised_cosine(2e3,  nx,     nx - 1, 6, 2e3)

    # Starting model: 50% of true mean — wrong enough to create real misfit,
    # but physically stable (positive mu, satisfies CFL).
    mu_initial = homogeneous(0.5 * np.mean(mu_true), nx + 1)

    u0 = np.zeros(nx)
    u0[nx // 2] = 1.0       # single-point IC displacement (centre of grid)
    v0 = homogeneous(0, nx)

    # ── Source: NO external source term ──────────────────────────────────
    # The original source (Gaussian/Ricker) peaks at t=0.5s with sigma=0.05s.
    # At dt=0.005s, steps=40 → t_max=0.2s, the source is still inactive
    # (exp(-(0.2-0.5)²/(2×0.05²)) ≈ 5e-18 ≈ 0).
    # The IC spike u0[nx//2]=1 already excites a propagating wave without
    # needing an external source, giving a clean inversion problem.
    # The source function is kept for visualization only (source plots).
    center_idx = nx // 2 + 1
    width_idx  = 2
    amplitude  = 1.0
    waveform   = source_func_factory(center_idx, width_idx, amplitude, nx + 2)
    # source_func for the wave solver: None (IC-only excitation)
    solver_source = None

    config_save = {
        'nx': nx, 'dx': config['dx'], 'dt': config['dt'],
        'steps': config['steps'], 'bc': config['bc'],
        'shots': config['shots'], 'measure_every': config['measure_every'],
        'mu': mu_initial.tolist(), 'rho': rho_arr.tolist(), 'u0': u0.tolist(),
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

    # ── Objective: source_func=None (IC-only, no external source term)
    # Both reference and forward simulations use solver_source=None so
    # physics is perfectly consistent between them.
    objective = SeismicObjective(
        nx=nx, dx=config['dx'], dt=config['dt'],
        steps=config['steps'], measure_every=config['measure_every'],
        shots=config['shots'], bc=config['bc'], seed=42,
        source_func=solver_source,   # None → IC-only excitation
    )

    gradient = FiniteDifferenceGradient(
        objective_fn=None,      # injected by optimizer
        delta_scale=1e-4,
        epsilon=1.0,            # Bug 7 fix: 1 Pa floor (was 1e-8)
    )

    loss_callback = LossHistoryCallback()
    logger = OptimizationLogger()

    print("  ✓ Optimization framework initialized")

    # ── Reference fields (Bug 1 fix) ──────────────────────────────────────
    # Compute reference (true-model) wavefields ONCE before optimisation.
    # The objective will compare current-model fields against these throughout.
    print("\n[1b] Computing reference (true-model) fields for inversion target...")
    ref_fields = objective.compute_reference_fields(
        mu_true, rho_arr, u0, v0
    )
    print(f"     Reference: {len(ref_fields)} time steps computed with TRUE model.")

    print("\n[2/10] Running initial forward simulation (initial model)...")
    # Uses initial (wrong) model — no external source, IC-only excitation
    fields, results, x = run_experiment_1d(
        nx=nx, dx=config['dx'], dt=config['dt'], steps=config['steps'],
        mu_arr=mu_initial, rho_arr=rho_arr, u0=u0, v0=v0,
        source_params=None,            # IC-only
        measure_every=config['measure_every'], shots=config['shots'],
        bc=config['bc'],
        reference_fields=ref_fields,   # compare against true model
    )
    print(f"       {len(fields)} time steps computed.")

    circuit_time_idx = 10
    print(f"\n[3/10] Building quantum circuit (Group 0, Index {circuit_time_idx})...")
    qc_paper, circuit_meta = build_paper_circuit(
        u0, v0, mu_true, rho_arr,
        dx=config['dx'], dt=config['dt'],
        time_step_idx=circuit_time_idx, nx=nx,
        observable=['X', 'Z', 'X', 'Z'],
        group_idx=0,
    )

    np.random.seed(42)
    source_amp = np.array([waveform(i, 0.5) for i in range(nx + 2)])
    results['source_amplitude'] = source_amp
    results['source_name'] = source_name

    t_vals = np.linspace(0, 1.0, 10000)
    source_time = np.array([waveform(center_idx, t) for t in t_vals])

    # Per-timestep quantum reconstruction loss (diagnostic, not inversion objective)
    loss_arr = []
    for i in range(1, len(fields)):
        qr = quantum_reconstruct(fields[i], shots=config['shots'])
        u_classical = fields[i][1:-1]
        u_quantum   = qr[1:-1]
        loss_arr.append(float(np.mean((u_classical - u_quantum) ** 2)))
    results['loss'] = loss_arr
    mean_loss = np.mean(loss_arr)

    print("\n[4/10] Initializing optimization process...")

    # Config dict for optimizer (uses initial model as starting point)
    configs_for_optim = {
        'nx': nx, 'dx': config['dx'], 'dt': config['dt'],
        'steps': config['steps'], 'bc': config['bc'],
        'shots': config['shots'], 'measure_every': config['measure_every'],
        'mu': mu_initial.tolist(),   # start from INITIAL model
        'rho': rho_arr.tolist(),
        'u0': u0.tolist(),
        'v0': v0.tolist(),
    }

    # Learning rate: gradients are ~1e-12 [J/Pa], mu ~ 1e10 Pa.
    # Adam normalizes per parameter: update ≈ lr × sign(gradient).
    # We need ~12 GPa total change over 50 iterations → lr ~ 2.4e8 Pa/iter.
    # Using lr=5e9 gives faster convergence since Adam tracks sign consistently.
    optimizer = SeismicOptimizer(
        configs=configs_for_optim,
        objective=objective,
        gradient=gradient,
        loss_history_callback=loss_callback,
        logger=logger,
        max_iterations=100,
        convergence_tolerance=1e-12,   # tight: stop only on true convergence
        early_stopping_patience=25,    # allow full exploration
        learning_rate=5e9,             # ~50 GPa/10 iters covers the mu gap
        use_deterministic=True,        # no shot noise → clean gradient signal
        reg_weight=0.0,
        n_grad_avg=1,
        ma_window=5,
    )

    print("  ✓ Optimizer initialized with iterative optimization framework")

    print("\n[5/10] Running iterative inversion optimization...")
    opt_results = optimizer.run_optimization()
    print(f"       Completed {opt_results['num_iterations']} iterations")
    print(f"       Final loss:   {opt_results['final_loss']:.6e}")
    print(f"       Best loss:    {opt_results['best_loss']:.6e}")
    initial_loss_opt = opt_results['convergence_report'].get('initial_loss', 1.0)
    if initial_loss_opt > 0:
        reduction_pct = (1.0 - opt_results['best_loss'] / initial_loss_opt) * 100
        print(f"       Loss reduction (best): {reduction_pct:.2f}%")

    mu_final = opt_results['mu_best']    # use BEST checkpoint, not final
    results['mu_final']  = mu_final.tolist()
    results['mu_initial'] = mu_initial.tolist()
    results['mu_updated'] = mu_final.tolist()
    results['mu_true']    = mu_true.tolist()
    results['optimization_history'] = opt_results

    print("\n[6/10] Saving experiment data...")
    exp_dir = save_experiment(fields, results, x, config_save)

    print("\n[7/10] Saving Excel workbook...")
    save_to_excel(fields, results, x, config_save, circuit_meta, exp_dir)

    print("\n[8/10] Generating source plots...")
    plot_source(source_amp, x, source_name=source_name)
    plot_source_time(t_vals, source_time, source_name=source_name)

    print("\n[9/10] Generating optimization analysis plots...")
    from src.visualization import plot_loss_history, plot_model_evolution
    try:
        plot_loss_history(opt_results)
        plot_model_evolution(opt_results)
    except Exception as e:
        print(f"    Warning: Could not generate optimization plots: {e}")

    print("\n[10/10] Generating forward simulation and analysis plots...")
    # Plot using the TRUE model's fields for reference forward-sim display
    fields_true, results_true, _ = run_experiment_1d(
        nx=nx, dx=config['dx'], dt=config['dt'], steps=config['steps'],
        mu_arr=mu_true, rho_arr=rho_arr, u0=u0, v0=v0,
        source_params=None,           # IC-only, consistent with optimizer
        measure_every=config['measure_every'], shots=config['shots'],
        bc=config['bc'],
    )
    results_true['source_amplitude'] = source_amp
    results_true['source_name'] = source_name
    results_true['mu_initial'] = mu_initial.tolist()
    results_true['mu_updated'] = mu_final.tolist()
    results_true['mu_true']    = mu_true.tolist()
    results_true['loss'] = loss_arr

    plot_forward_sim(fields_true, results_true, x, config)
    plot_energy(results_true)
    plot_overlap(results_true)
    plot_error(fields_true, results_true, config)
    plot_model_update(results)   # initial vs optimized mu

    print("\nAll figures saved to figures/")
    print("All data saved to data/")
    print("Done.")
