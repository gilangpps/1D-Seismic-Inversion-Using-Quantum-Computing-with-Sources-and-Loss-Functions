"""
1-D Seismic Inversion Using Quantum Computing Simulation
with Sources and Loss Functions.

This entry point now runs a sequence of sweep configurations one by one.
Each sweep produces its own subfolder for data and figures, using the sweep
name as the folder name.
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.circuit import build_paper_circuit
from src.hamiltonian import check_ic_breaks_degeneracy
from src.constants import configure_plot_style, FIGURES_DIR
from src.distributions import homogeneous, raised_cosine
from src.encoding import quantum_reconstruct
from src.experiment import run_experiment_1d
from src.experiment.validate_hamiltonian import run_hamiltonian_validation, format_validation_for_excel
from src.noise import add_gaussian_noise_to_fields, noise_summary
from src.optimization import (
    FiniteDifferenceGradient,
    LossHistoryCallback,
    OptimizationLogger,
    SeismicObjective,
    SeismicOptimizer,
)
from src.persistence import save_experiment, save_to_excel
from src.visualization import (
    plot_circuit,
    plot_density_model,
    plot_velocity_model,
    plot_velocity_inversion,
    plot_energy,
    plot_error,
    plot_forward_sim,
    plot_hamiltonian_validation,
    plot_initial_mu,
    plot_loss,
    plot_loss_history,
    plot_model_evolution,
    plot_model_update,
    plot_mu_model,
    plot_observed_vs_predicted,
    plot_overlap,
    plot_quantum_recon_loss,
    plot_source,
    plot_source_time,
    plot_source_receiver_geometry,
    plot_noisy_data_comparison,
    plot_mu_inversion,
    plot_inversion_process_grid,
    plot_noise_level_summary,
)
from src.wave import gaussian_source, ricker_wavelet_source, set_source_peak

warnings.filterwarnings("ignore")

configure_plot_style()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SWEEPS = [
    {
    "name": "gaussian_source",
    "n_qubits": 4,
    "n_clbits": 4,
    "nx": 7,
    "dx_m": 63.0,
    "dt_s": 0.0025,
    "steps": 120,
    "source_type": "gaussian",
    "t0_s": 0.045,
    "sigma_t_s": 0.030,
    "source_x": "center",
    "q1_0_gate": "ry",
    "q1_0_theta": -1.7,
    "q1_2_gate_1": "x",
    "q1_2_gate_2": "z",
    "mu_true_range_Pa": [1e10, 4e10],
    "rho_range_kgm3": [2e3, 4e3],
    "mu_initial_factor": 0.80,
    "engine": "quantum",
    "run_hamiltonian_validation": True,
},
    {
    "name": "ricker_wavelet_source",
    "n_qubits": 4,
    "n_clbits": 4,
    "nx": 7,
    "dx_m": 63.0,
    "dt_s": 0.0025,
    "steps": 120,
    "source_type": "ricker",
    "f0_hz": 10.0,
    "t0_s": 0.045,
    "source_x": "center",
    "q1_0_gate": "ry",
    "q1_0_theta": -1.7,
    "q1_2_gate_1": "x",
    "q1_2_gate_2": "z",
    "mu_true_range_Pa": [1e10, 4e10],
    "rho_range_kgm3": [2e3, 4e3],
    "mu_initial_factor": 0.80,
    "engine": "quantum",
    "run_hamiltonian_validation": True,
},
    {
    "name": "ricker_wavelet_source_20hz",
    "n_qubits": 4,
    "n_clbits": 4,
    "nx": 7,
    "dx_m": 63.0,
    "dt_s": 0.0025,
    "steps": 120,
    "source_type": "ricker",
    "f0_hz": 20.0,
    "t0_s": 0.045,
    "source_x": "center",
    "q1_0_gate": "ry",
    "q1_0_theta": -1.7,
    "q1_2_gate_1": "x",
    "q1_2_gate_2": "z",
    "mu_true_range_Pa": [1e10, 4e10],
    "rho_range_kgm3": [2e3, 4e3],
    "mu_initial_factor": 0.80,
    "engine": "quantum",
    "run_hamiltonian_validation": True,
},
    # Noise-robustness sweeps: same setup as ricker_wavelet_source, but the
    # "observed" forward data used as the inversion target has Gaussian
    # noise added at several levels (1%, 2.5%, 5%, 10% of peak amplitude),
    # per the advisor's request. Each level gets its own sweep so it
    # produces its own figures/data folder AND its own per-iteration
    # "Inversion Process" panel figure; main() also builds a combined
    # relative-error-vs-noise-level summary plot across all of them.
] + [
    {
        "name": f"ricker_wavelet_source_noisy_{tag}",
        "n_qubits": 4,
        "n_clbits": 4,
        "nx": 7,
        "dx_m": 63.0,
        "dt_s": 0.0025,
        "steps": 120,
        "source_type": "ricker",
        "f0_hz": 10.0,
        "t0_s": 0.045,
        "source_x": "center",
        "q1_0_gate": "ry",
        "q1_0_theta": -1.7,
        "q1_2_gate_1": "x",
        "q1_2_gate_2": "z",
        "mu_true_range_Pa": [1e10, 4e10],
        "rho_range_kgm3": [2e3, 4e3],
        "mu_initial_factor": 0.80,
        "engine": "quantum",
        "run_hamiltonian_validation": True,
        "add_noise": True,
        "noise_level": level,
        "noise_seed": 123,
    }
    for tag, level in [("1pct", 0.01), ("2p5pct", 0.025), ("5pct", 0.05), ("10pct", 0.10)]
]


def build_medium_from_sweep(sweep):
    nx = int(sweep["nx"])
    mu_min, mu_max = sweep.get("mu_true_range_Pa", [1e10, 4e10])
    rho_min, rho_max = sweep.get("rho_range_kgm3", [2e3, 4e3])
    mu_true = raised_cosine((mu_max - mu_min), nx, nx - 1, 6, mu_min)
    mu_initial_factor = float(sweep.get("mu_initial_factor", 0.50))

    # Density (rho) is treated as KNOWN/fixed background information — it is
    # not inverted for in this project, only mu is. "Known", however, does
    # not mean flat: a real 1-D earth model has density increasing with
    # depth, so rho_true is given the same raised-cosine depth profile as
    # mu_true (rising from rho_min at the shallowest sample to rho_max at
    # the deepest one, e.g. 2000 -> 4000 kg/m^3). This rho_arr is used
    # everywhere density is needed (Hamiltonian, forward simulation,
    # optimizer, all plots).
    rho_arr = raised_cosine((rho_max - rho_min), nx, nx - 1, 6, rho_min)

    # DESIGN DECISION (finalised): mu_initial is the HOMOGENEOUS starting
    # guess, not v_initial. This matches the thesis scope — only mu is
    # actually inverted for (fed into the Hamiltonian/optimizer/loss
    # function), so the standard FWI convention of "no prior information
    # -> flat starting model" applies directly to mu, not to a derived
    # quantity like v_s = sqrt(mu/rho).
    #
    # Consequence (expected, not a bug): since rho(z) is depth-varying but
    # mu_initial is flat, v_initial = sqrt(mu_initial/rho(z)) will NOT be
    # flat — it curves smoothly with depth. This shows up consistently in
    # velocity_inversion.png (v_initial curve) and in the "Initial Model"
    # panel of the inversion-process heatmaps (v_s color gradient there
    # instead of a single uniform color), since both of those plot the
    # derived v_s, not mu directly. mu_inversion.png and mu_initial.png
    # (which plot mu directly) still show mu_initial as a flat line.
    mu_initial = homogeneous(mu_initial_factor * float(np.mean(mu_true)), nx)

    return mu_true, rho_arr, mu_initial


def build_source_from_sweep(sweep, nx, dt, steps):
    source_type = sweep.get("source_type", "gaussian").lower()
    t_max = steps * dt
    t0 = float(sweep.get("t0_s", t_max / 3.0))
    if source_type == "gaussian":
        sigma_t = float(sweep.get("sigma_t_s", max(t_max / 12.0, 2.0 * dt)))
        source_func_factory = gaussian_source
        source_name = "Gaussian"
    else:
        f0_hz = float(sweep.get("f0_hz", 25.0))
        sigma_t = max(2.0 * dt, 1.0 / (np.pi * f0_hz))
        source_func_factory = ricker_wavelet_source
        source_name = "Ricker wavelet"

    center_idx = nx // 2 + 1
    width_idx = 2
    amplitude = 1.0

    solver_source = source_func_factory(
        center_idx, width_idx, amplitude, nx + 2, t0=t0, sigma_t=sigma_t
    )
    set_source_peak(solver_source, t0=t0, sigma_t=sigma_t)

    source_params = {
        "kind": source_type,
        "center": center_idx,
        "width": width_idx,
        "amplitude": amplitude,
    }
    return solver_source, source_params, source_name, t0, sigma_t, center_idx, width_idx, amplitude


def run_single_sweep(sweep, base_output_dir, reg_weight_override=None):
    sweep_name = sweep["name"]
    nx = int(sweep["nx"])
    dx = float(sweep["dx_m"])
    dt = float(sweep["dt_s"])
    steps = int(sweep["steps"])

    # Optional additive noise on the "observed" forward data used as the
    # inversion target (Requirement: robustness test under noisy data).
    add_noise = bool(sweep.get("add_noise", False))
    noise_level = float(sweep.get("noise_level", 0.05))
    noise_seed = int(sweep.get("noise_seed", 123))

    data_dir = base_output_dir / "data" / sweep_name
    figure_dir = base_output_dir / "figures" / sweep_name
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "nx": nx,
        "dx": dx,
        "dt": dt,
        "steps": steps,
        "bc": "dirichlet",
        "bcs": {"left": "DBC", "right": "DBC"},
        "shots": 1200,
        "measure_every": 4,
        "sweep_name": sweep_name,
        "n_qubits": sweep.get("n_qubits", 4),
        "n_clbits": sweep.get("n_clbits", 4),
        "add_noise": add_noise,
        "noise_level": noise_level if add_noise else None,
        "noise_seed": noise_seed if add_noise else None,
    }

    mu_true, rho_arr, mu_initial = build_medium_from_sweep(sweep)
    
    # Initial conditions: Asymmetric spike (proven non-eigenmode of K)
    #
    # BUG FIX (2026-07-23): Previous ICs (Gaussian, multi-mode sinusoids) were
    # eigenmodes or superpositions of eigenmodes of the stiffness operator K(1).
    # For homogeneous mu, K(mu) = mu * K(1), so eigenmodes of K(1) scale
    # uniformly — (H1-H2)@psi = 0 despite H1 != H2. This makes quantum evolution
    # mu-independent and gradient = 0 (see BUG_REPORT_QUANTUM_INVERSION.md §290).
    #
    # Key insight: ANY linear combination of sine functions sin(k*pi*x/L) for
    # integer k is an eigenmode of K(1) on a uniform Dirichlet domain.
    # Superposition of multiple sinusoids ≠ non-eigenmode (was wrong in v1).
    #
    # Fix: Use a spatially localized asymmetric spike. This has broad spectral
    # content and is NOT an eigenmode of K(1), guaranteeing (H1-H2)@psi != 0.
    #
    # Reference: BUG_REPORT_QUANTUM_INVERSION.md Option 1, PROMPT.md Fix A
    rng = np.random.default_rng(seed=42)
    spike_center = int(0.3 * nx)
    u0 = np.zeros(nx)

    if spike_center-2 >= 0:
        u0[spike_center-2] = 0.08

    if spike_center-1 >= 0:
        u0[spike_center-1] = 0.30

    u0[spike_center] = 1.0

    if spike_center+1 < nx:
        u0[spike_center+1] = 0.55

    if spike_center+2 < nx:
        u0[spike_center+2] = 0.22

    if spike_center+3 < nx:
        u0[spike_center+3] = 0.05
    v0 = np.zeros(nx)

    solver_source, source_params, source_name, t0, sigma_t, center_idx, width_idx, amplitude = build_source_from_sweep(sweep, nx, dt, steps)
    t_max = steps * dt
    x_grid = np.arange(nx + 2) * dx
    source_amp = np.array([solver_source(i, t0) for i in range(nx + 2)])
    t_plot = np.linspace(0, t_max * 1.2, 5000)
    source_time = np.array([solver_source(center_idx, t) for t in t_plot])

    print(f"\n{'=' * 72}")
    print(f"Running sweep: {sweep_name}")
    print(f"  nx={nx}, dx={dx}, dt={dt}, steps={steps}")
    print(f"  source={source_name}, t0={t0:.4f}s, sigma_t={sigma_t:.4f}s")
    print(f"  output dir: {figure_dir}")
    print(f"  data dir: {data_dir}")
    print(f"{'=' * 72}")

    program_start = time.perf_counter()

    engine = sweep.get("engine", "classical")  # Forward simulation engine
    print(f"  Forward simulation engine: {engine}")

    objective = SeismicObjective(
        nx=nx,
        dx=dx,
        dt=dt,
        steps=steps,
        measure_every=config["measure_every"],
        shots=config["shots"],
        bc=config["bc"],
        seed=42,
        source_func=solver_source,
        engine=engine,  # Pass engine parameter
    )
    gradient_obj = FiniteDifferenceGradient(objective_fn=None)  # default: delta_scale=1e-3, epsilon=1e6
    loss_callback = LossHistoryCallback()
    logger = OptimizationLogger()

    ref_fields = objective.compute_reference_fields(mu_true, rho_arr, u0, v0)

    # ── Optional noise injection on the observed/target data ──────────────
    # If enabled, the optimizer inverts against a noisy version of the
    # true-model wavefield (mimicking real, imperfect recordings), while
    # `ref_fields` (clean) is kept around for diagnostics/plots that should
    # depict the underlying physics rather than the noisy "measurement".
    noise_stats = None
    if add_noise:
        ref_fields_noisy = add_gaussian_noise_to_fields(
            ref_fields, noise_level=noise_level, seed=noise_seed,
        )
        objective.set_reference_fields(ref_fields_noisy)
        noise_stats = noise_summary(ref_fields, ref_fields_noisy)
        print(
            f"  [NOISE] add_noise=True, level={noise_level*100:.1f}%, "
            f"achieved SNR={noise_stats['snr_db']:.1f} dB "
            f"(rms noise={noise_stats['rms_noise']:.3e})"
        )
        ref_fields_for_overlap = ref_fields_noisy
    else:
        ref_fields_for_overlap = ref_fields

    run_experiment_1d(
        nx=nx,
        dx=dx,
        dt=dt,
        steps=steps,
        mu_arr=mu_initial,
        rho_arr=rho_arr,
        u0=u0,
        v0=v0,
        source_params=source_params,
        measure_every=config["measure_every"],
        shots=config["shots"],
        bc=config["bc"],
        reference_fields=ref_fields_for_overlap,
    )

    circuit_time_idx = 10
    qc_paper, circuit_meta = build_paper_circuit(
        u0,
        v0,
        mu_true,
        rho_arr,
        dx=dx,
        dt=dt,
        time_step_idx=circuit_time_idx,
        nx=nx,
        observable=["X", "Z", "X", "Z"],
        group_idx=0,
        paper_diagram_mode=True,  # Use deterministic paper-style gates for visualization
    )

    np.random.seed(42)
    quantum_recon_loss_arr = []
    for i in range(1, len(ref_fields)):
        qr = quantum_reconstruct(ref_fields[i], shots=config["shots"])
        u_cl = ref_fields[i][1:-1]
        u_qu = qr[1:-1]
        quantum_recon_loss_arr.append(float(np.mean((u_cl - u_qu) ** 2)))

    # Pre-flight null-space check (see BUG_REPORT_QUANTUM_INVERSION.md §290)
    # Verifies that (H(mu_true) - H(mu_initial)) @ psi0 != 0, ensuring
    # optimizer will see mu-dependent dynamics and non-zero gradient.
    if engine == "quantum":
        mu_bc_initial = np.zeros(nx + 2)
        mu_bc_initial[1:-1] = mu_initial
        mu_bc_initial[0] = mu_initial[0]
        mu_bc_initial[-1] = mu_initial[-1]
        mu_bc_true = np.zeros(nx + 2)
        mu_bc_true[1:-1] = mu_true
        mu_bc_true[0] = mu_true[0]
        mu_bc_true[-1] = mu_true[-1]
        rho_bc = np.zeros(nx + 2)
        rho_bc[1:-1] = rho_arr
        rho_bc[0] = rho_arr[0]
        rho_bc[-1] = rho_arr[-1]
        check_ic_breaks_degeneracy(
            mu_bc_initial, mu_bc_true, rho_bc, dx, nx, u0, v0, tol=1e-3
        )

    max_iterations = int(os.getenv("MAX_ITERATIONS", "950"))
    # Tikhonov regularization R(mu) = ||mu - mu_prior||^2 (Thesis Eq. 2.15;
    # matches g(s) = ||s - s0||^2, Nguyen & Tura 2025 Eq. 14-15, per advisor's
    # request). mu_prior defaults to mu_initial inside SeismicOptimizer.
    # reg_weight is a hyperparameter (lambda): 0.0 disables it (prior
    # behaviour, matches what was actually run before). The value below is
    # only a starting point — pick a properly-tuned lambda with your advisor,
    # e.g. via the L-curve method described in the reference paper's Methods
    # section (plot ||mu-mu_prior|| vs. the data-misfit residual on a
    # log-log scale for a few lambda values and pick the corner).
    reg_weight = (
        reg_weight_override
        if reg_weight_override is not None
        else float(os.getenv("REG_WEIGHT", "1e-23"))
    )
    optimizer = SeismicOptimizer(
        configs={
            "nx": nx,
            "dx": dx,
            "dt": dt,
            "steps": steps,
            "bc": config["bc"],
            "shots": config["shots"],
            "measure_every": config["measure_every"],
            "mu": mu_initial.tolist(),
            "rho": rho_arr.tolist(),
            "u0": u0.tolist(),
            "v0": v0.tolist(),
        },
        objective=objective,
        gradient=gradient_obj,
        loss_history_callback=loss_callback,
        logger=logger,
        max_iterations=max_iterations,
        convergence_tolerance=1e-5,
        early_stopping_patience=50,
        learning_rate=3e8,
        use_deterministic=True,
        reg_weight=reg_weight,
        n_grad_avg=1,
        ma_window=5,
    )
    opt_results = optimizer.run_optimization()
    mu_recovered = opt_results["mu_best"]
    mu_prior_arr = optimizer.mu_prior
    regularization_norm = float(np.sum((mu_recovered - mu_prior_arr) ** 2))
    reg_mean_sq = float(np.mean((mu_recovered - mu_prior_arr) ** 2))
    data_misfit = float(opt_results["best_loss"] - reg_weight * reg_mean_sq)

    fields_rec, results_rec, _ = run_experiment_1d(
        nx=nx,
        dx=dx,
        dt=dt,
        steps=steps,
        mu_arr=mu_recovered,
        rho_arr=rho_arr,
        u0=u0,
        v0=v0,
        source_params=source_params,
        measure_every=config["measure_every"],
        shots=config["shots"],
        bc=config["bc"],
        reference_fields=ref_fields_for_overlap,
    )
    fields_true, results_true, _ = run_experiment_1d(
        nx=nx,
        dx=dx,
        dt=dt,
        steps=steps,
        mu_arr=mu_true,
        rho_arr=rho_arr,
        u0=u0,
        v0=v0,
        source_params=source_params,
        measure_every=config["measure_every"],
        shots=config["shots"],
        bc=config["bc"],
    )

    inversion_loss_ts = objective.compute_time_series_loss(fields_rec)

    np.random.seed(42)
    quantum_recon_loss_rec = []
    for i in range(1, len(fields_rec)):
        qr = quantum_reconstruct(fields_rec[i], shots=config["shots"])
        u_cl = fields_rec[i][1:-1]
        u_qu = qr[1:-1]
        quantum_recon_loss_rec.append(float(np.mean((u_cl - u_qu) ** 2)))

    # ── Hamiltonian Validation Experiment ─────────────────────────────────
    # Run independent quantum vs classical comparison to validate that
    # Hamiltonian H correctly captures elastic wave physics
    # NOTE: Validation runs WITHOUT source to test pure Hamiltonian dynamics
    # (Source handling via operator splitting is tested separately in forward simulation)
    hamiltonian_validation = None
    if sweep.get("run_hamiltonian_validation", False):
        print("\n" + "="*72)
        print("Running Hamiltonian validation experiment...")
        print("="*72)
        hamiltonian_validation = run_hamiltonian_validation(
            mu_arr=mu_true,  # Use true model for validation
            rho_arr=rho_arr,
            u0=u0,
            v0=v0,
            dx=dx,
            dt=dt,
            steps=steps,
            nx=nx,
            bc=config["bc"],
            source_func=None,  # No source: validates pure H = [[0,A],[A†,0]] dynamics
        )
        print(f"Hamiltonian validation complete: mean overlap = {hamiltonian_validation['mean_overlap']:.6f}")
        print("="*72 + "\n")

    runtime_seconds = time.perf_counter() - program_start

    results_save = dict(results_rec)
    results_save["source_amplitude"] = source_amp
    results_save["source_name"] = source_name
    results_save["loss"] = inversion_loss_ts.tolist()
    results_save["quantum_recon_loss"] = quantum_recon_loss_arr
    results_save["quantum_recon_loss_rec"] = quantum_recon_loss_rec
    results_save["mu_initial"] = mu_initial.tolist()
    results_save["mu_updated"] = mu_recovered.tolist()
    results_save["mu"] = mu_true.tolist()
    results_save["mu_true"] = mu_true.tolist()
    results_save["optimization_history"] = opt_results
    if hamiltonian_validation is not None:
        results_save["hamiltonian_validation"] = hamiltonian_validation
    results_save["performance_metrics"] = {
        "Number of Qubits": qc_paper.num_qubits,
        "Circuit Depth": qc_paper.depth(),
        "Gate Count": qc_paper.size(),
        "Gate Operations": str(dict(qc_paper.count_ops())),
        "Measurement Shots": config["shots"],
        "Runtime (s)": runtime_seconds,
        "Relative Model Error": float(np.linalg.norm(mu_recovered - mu_true) / np.linalg.norm(mu_true)),
        "Average Quantum Reconstruction Loss": float(np.mean(quantum_recon_loss_rec)),
        "Initial Loss": float(opt_results["convergence_report"].get("initial_loss", 1.0)),
        "Best Loss": float(opt_results["best_loss"]),
        "Final Loss": float(opt_results["final_loss"]),
        "Loss Reduction (%)": float((1 - opt_results["best_loss"] / opt_results["convergence_report"].get("initial_loss", 1.0)) * 100)
        if opt_results["convergence_report"].get("initial_loss", 1.0) > 0 else 0.0,
        "Regularization Weight (lambda)": reg_weight,
        "Regularization Norm ||mu-mu_prior||^2 (sum)": regularization_norm,
        "Regularization Mean Squared": reg_mean_sq,
        "Data Misfit (best_loss - lambda*mean_sq)": data_misfit,
    }
    if hamiltonian_validation is not None:
        results_save["performance_metrics"]["Hamiltonian Validation Mean Overlap"] = float(hamiltonian_validation["mean_overlap"])
        results_save["performance_metrics"]["Hamiltonian Validation Mean L2 Error"] = float(hamiltonian_validation["mean_l2_error"])

    config_save = {
        "nx": nx,
        "dx": dx,
        "dt": dt,
        "steps": steps,
        "bc": config["bc"],
        "shots": config["shots"],
        "measure_every": config["measure_every"],
        "mu": mu_initial.tolist(),
        "rho": rho_arr.tolist(),
        "u0": u0.tolist(),
        "v0": v0.tolist(),
        "source_type": source_name,
        "source_t0": t0,
        "source_sigma_t": sigma_t,
        "sweep_name": sweep_name,
        "mu_true_range_Pa": sweep.get("mu_true_range_Pa", [1e10, 4e10]),
        "rho_range_kgm3": sweep.get("rho_range_kgm3", [2e3, 4e3]),
        "mu_initial_factor": sweep.get("mu_initial_factor", 0.50),
        "engine": engine,  # Forward simulation engine
        "reg_weight": reg_weight,
        "run_hamiltonian_validation": sweep.get("run_hamiltonian_validation", False),
        "add_noise": add_noise,
        "noise_level": noise_level if add_noise else None,
        "noise_seed": noise_seed if add_noise else None,
        "achieved_snr_db": noise_stats["snr_db"] if noise_stats else None,
    }

    exp_dir = save_experiment(fields_rec, results_save, x_grid, config_save, data_dir=data_dir)
    save_to_excel(fields_rec, results_save, x_grid, config_save, circuit_meta, exp_dir)

    plot_initial_mu(mu_initial, mu_true=mu_true, dx=dx, fig_dir=figure_dir)
    plot_density_model(rho_arr, dx=dx, fig_dir=figure_dir)
    plot_mu_model(mu_true, dx=dx, fig_dir=figure_dir)
    plot_velocity_model(mu_true, rho_arr, dx=dx, fig_dir=figure_dir)
    plot_source(source_amp, x_grid, fig_dir=figure_dir, source_name=source_name)
    plot_source_time(t_plot, source_time, fig_dir=figure_dir, source_name=source_name)
    plot_source_receiver_geometry(
        mu_true, rho_arr=rho_arr, dx=dx, source_idx=center_idx - 1,
        receiver_indices=list(range(nx)), fig_dir=figure_dir,
    )

    results_true_plot = dict(results_true)
    results_true_plot["source_amplitude"] = source_amp
    results_true_plot["source_name"] = source_name
    results_true_plot["mu_initial"] = mu_initial.tolist()
    results_true_plot["mu_updated"] = mu_recovered.tolist()
    results_true_plot["mu_true"] = mu_true.tolist()
    results_true_plot["loss"] = inversion_loss_ts.tolist()
    plot_forward_sim(fields_true, results_true_plot, x_grid, config, fig_dir=figure_dir)
    plot_energy(results_true, fig_dir=figure_dir)
    plot_overlap(results_rec, fig_dir=figure_dir)
    plot_error(fields_true, results_true_plot, config, fig_dir=figure_dir)

    # "Observed" data: use the noisy target the optimizer actually inverted
    # against (if noise is enabled), so the plot reflects what was really
    # used as the inversion target rather than the idealized clean field.
    fields_obs_for_plot = ref_fields_noisy if add_noise else fields_true
    plot_observed_vs_predicted(fields_obs_for_plot, fields_rec, results_save, config, fig_dir=figure_dir)

    if add_noise:
        rec_idx = nx // 2
        clean_trace = np.array([f[rec_idx + 1] for f in ref_fields])
        noisy_trace = np.array([f[rec_idx + 1] for f in ref_fields_noisy])
        plot_noisy_data_comparison(
            results_true["times"], clean_trace, noisy_trace,
            fig_dir=figure_dir, receiver_idx=rec_idx,
            noise_level=noise_level, snr_db=noise_stats["snr_db"],
        )

    plot_mu_inversion(mu_true, mu_initial, mu_recovered, dx=dx, fig_dir=figure_dir)
    plot_velocity_inversion(mu_true, mu_initial, mu_recovered, rho_arr, dx=dx, fig_dir=figure_dir)

    # Per-iteration "Initial Model / 1st Iteration / ... " panel figure
    # (mirrors the advisor's reference figure), for noiseless or noisy runs.
    plot_inversion_process_grid(
        opt_results["mu_history"], rho_arr, dx=dx,
        noise_level=noise_level if add_noise else None,
        fig_dir=figure_dir,
    )

    plot_loss_history(opt_results, fig_dir=figure_dir)
    plot_model_evolution(opt_results, dx=dx, fig_dir=figure_dir)
    plot_model_update(results_save, dx=dx, fig_dir=figure_dir)
    plot_loss(results_save, fig_dir=figure_dir)
    plot_quantum_recon_loss(results_save, fig_dir=figure_dir)

    # Plot Hamiltonian validation if available
    if hamiltonian_validation is not None:
        plot_hamiltonian_validation(hamiltonian_validation, fig_dir=figure_dir)

    try:
        plot_circuit(qc_paper, circuit_meta, fig_dir=figure_dir)
    except Exception as exc:
        print(f"  Circuit plot skipped: {exc}")

    print(f"Completed sweep {sweep_name}; results in {data_dir} and {figure_dir}")
    summary = {
        "sweep_name": sweep_name,
        "add_noise": add_noise,
        "noise_level": noise_level if add_noise else 0.0,
        "relative_model_error": results_save["performance_metrics"]["Relative Model Error"],
        "reg_weight": reg_weight,
        "regularization_norm": regularization_norm,
        "data_misfit": data_misfit,
        "best_loss": float(opt_results["best_loss"]),
        "final_loss": float(opt_results["final_loss"]),
    }
    return exp_dir, figure_dir, summary


def main():
    selected_names = os.getenv("SWEEP_NAMES", "").split(",")
    selected_names = [name.strip() for name in selected_names if name.strip()]
    sweeps = SWEEPS
    if selected_names:
        sweeps = [sweep for sweep in SWEEPS if sweep["name"] in selected_names]
    base_output_dir = Path(".").resolve()

    summaries = []
    for sweep in sweeps:
        _, _, summary = run_single_sweep(sweep, base_output_dir)
        summaries.append(summary)

    # Cross-noise-level robustness summary (requires >= 2 noisy runs to be
    # meaningful; e.g. the 1% / 2.5% / 5% / 10% noise sweeps below).
    noisy_summaries = [s for s in summaries if s["add_noise"]]
    if len(noisy_summaries) >= 2:
        noise_pcts = [s["noise_level"] * 100 for s in noisy_summaries]
        rel_errors = [s["relative_model_error"] for s in noisy_summaries]
        summary_dir = base_output_dir / "figures" / "noise_robustness_summary"
        plot_noise_level_summary(noise_pcts, rel_errors, fig_dir=summary_dir)


if __name__ == "__main__":
    main()
