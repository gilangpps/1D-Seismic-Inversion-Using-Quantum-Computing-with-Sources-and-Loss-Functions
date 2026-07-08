"""
src/visualization/__init__.py
Publication-quality plots for 1-D Seismic Inversion (Requirement K).

Mandatory plots (10):
  1. plot_initial_mu       — initial mu model
  2. plot_density_model    — density model
  3. plot_source           — Ricker/Gaussian source vs position
  4. plot_forward_sim      — wave propagation snapshots
  5. plot_error            — quantum reconstruction error
  6. plot_observed_vs_predicted — observed vs predicted seismic data
  7. plot_loss_history     — loss convergence
  8. plot_mu_inversion     — mu_true vs mu_recovered
  9. plot_energy           — classical PDE energy evolution
  10. plot_overlap         — quantum overlap evolution

Additional:
  plot_source_time, plot_loss, plot_model_update, plot_model_evolution, plot_circuit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.constants import ENUMS, FIGURES_DIR
from src.encoding import quantum_reconstruct


# ── 1. Initial mu model ───────────────────────────────────────────────────────

def plot_initial_mu(mu_initial, mu_true=None, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mu_initial))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mu_initial, 'o-', color='steelblue', linewidth=1.8,
            markersize=5, label=r'$\mu_{\mathrm{initial}}$')
    if mu_true is not None:
        ax.plot(np.arange(len(mu_true)), mu_true, 's--', color='red',
                linewidth=1.5, markersize=5, label=r'$\mu_{\mathrm{true}}$')
    ax.set_xlabel('Grid index')
    ax.set_ylabel(r'$\mu$ [Pa]')
    ax.set_title('Initial Elastic Modulus Model')
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'mu_initial.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 2. Density model ──────────────────────────────────────────────────────────

def plot_density_model(rho_arr, dx=1.0, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rho_arr)) * dx
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, rho_arr, 'o-', color='darkorange', linewidth=1.8, markersize=5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel(r'$\rho$ [kg/m$^3$]')
    ax.set_title('Density Model')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'density_model.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 3. Source amplitude vs position ──────────────────────────────────────────

def plot_source(source_amplitude, x, fig_dir=FIGURES_DIR, source_name='Ricker'):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, source_amplitude, 'o-', color='purple', linewidth=1.5, markersize=4)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Source amplitude')
    ax.set_title(f'{source_name} Source Amplitude vs Position')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'source.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_source_time(t_vals, source_amplitudes, fig_dir=FIGURES_DIR, source_name='Ricker'):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals, source_amplitudes, '-', color='purple', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Source amplitude')
    ax.set_title(f'{source_name} Source Wavelet vs Time')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'source_time.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 4. Forward simulation multiplot ──────────────────────────────────────────

def plot_forward_sim(fields, results, x, config, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    nx   = config['nx']
    dt   = config['dt']
    shots = config['shots']
    bcs  = config.get('bcs', {'left': 'DBC', 'right': 'DBC'})
    rho  = results['rho']
    mu   = results['mu']

    n_fields = len(fields)
    snap_idx = [0, n_fields // 4, n_fields // 2, 3 * n_fields // 4, n_fields - 1]

    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho[:nx] if len(rho) >= nx else rho
    rho_bc[0] = rho[0]; rho_bc[-1] = rho[-1]

    mu_bc = np.zeros(nx + 2)
    mu_bc[1:min(len(mu) + 1, nx + 2)] = mu[:min(len(mu), nx + 1)]
    mu_bc[0] = mu[0]; mu_bc[-1] = mu[-1]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    # Panel (a): medium properties
    ax_rho = axes[0, 0]
    ax_mu  = ax_rho.twinx()
    ax_rho.text(0.05, 0.90, ENUMS[0], transform=ax_rho.transAxes, fontsize=13)
    ax_rho.plot(x, rho_bc, color='blue', linewidth=1.5, label=r'$\rho$')
    ax_rho.set_ylabel(r'$\rho$ [kg/m$^3$]', color='blue')
    ax_rho.tick_params(axis='y', labelcolor='blue')
    ax_mu.plot(x, mu_bc, color='red', linewidth=1.5, label=r'$\mu$')
    ax_mu.set_ylabel(r'$\mu$ [Pa]', color='red')
    ax_mu.tick_params(axis='y', labelcolor='red')
    lines = ax_rho.get_legend_handles_labels()[0] + ax_mu.get_legend_handles_labels()[0]
    labels = [r'$\rho$', r'$\mu$']
    ax_mu.legend(lines, labels, loc='lower left', fontsize=8)
    ax_rho.set_xlabel('x [m]')

    np.random.seed(42)
    for pi, ti in enumerate(snap_idx):
        row = (pi + 1) // 3
        col = (pi + 1) % 3
        ax  = axes[row, col]
        ax.text(0.05, 0.90, ENUMS[pi + 1], transform=ax.transAxes, fontsize=13)
        t_val = results['times'][ti]
        cl    = fields[ti]
        qs    = quantum_reconstruct(cl, shots=shots)
        qc_r  = quantum_reconstruct(cl, shots=shots, noise_level=0.03)
        ax.plot(x, cl, 'o-', color='black', markersize=4, linewidth=0.8,
                label='Classical PDE', zorder=3)
        ax.plot(x, qs, '-', color='red', linewidth=1.2,
                label=f'Quantum sim. ({shots} shots)')
        ax.plot(x, qc_r, '-', color='blue', linewidth=1.2,
                label=f'Quantum + noise ({shots} shots)')
        ax.set_title(f"t = {t_val:.4f} s")
        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'u [m]')
        ax.set_ylim(-1.1, 1.1)
        if pi == 0:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    fig.tight_layout()
    path = fig_dir / 'forward_sim.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 5. Quantum reconstruction error ──────────────────────────────────────────

def plot_error(fields, results, config, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    shots = config['shots']
    np.random.seed(42)
    step  = max(1, len(fields) // 50)
    errs, tidx = [], []
    for i in range(1, len(fields), step):
        qr = quantum_reconstruct(fields[i], shots=shots)
        rn = np.linalg.norm(fields[i])
        errs.append(np.linalg.norm(fields[i] - qr) / (rn + 1e-30))
        tidx.append(i)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tidx, errs, color='blue', linewidth=1.0)
    me = float(np.mean(errs)) if errs else 0.0
    ax.axhline(me, color='black', linestyle='--', linewidth=0.8,
               label=f'Mean = {me:.2e}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Relative L2 error')
    ax.set_yscale('log')
    ax.set_title(f'Quantum Reconstruction Error ({shots} shots)')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    path = fig_dir / 'error.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 6. Observed vs predicted seismic data ────────────────────────────────────

def plot_observed_vs_predicted(fields_obs, fields_pred, results, config,
                               fig_dir=FIGURES_DIR, receiver_idx=None):
    """
    Plot observed (true-model) vs predicted (current-model) seismic traces.

    Shows how well the inverted model reproduces the reference data.

    Parameters
    ----------
    fields_obs  : list of arrays — reference (true-model) wavefields
    fields_pred : list of arrays — current-model wavefields
    results     : dict with 'times'
    config      : dict with 'nx'
    receiver_idx : int or None — grid index of receiver (default: nx//2)
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    nx  = config['nx']
    rec = receiver_idx if receiver_idx is not None else nx // 2

    times = results['times']
    n     = min(len(fields_obs), len(fields_pred), len(times))

    u_obs  = np.array([fields_obs[i][rec + 1] for i in range(n)])
    u_pred = np.array([fields_pred[i][rec + 1] for i in range(n)])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(times[:n], u_obs,  color='black', linewidth=1.3,
                 label='Observed (true model)')
    axes[0].plot(times[:n], u_pred, color='red',   linewidth=1.2,
                 linestyle='--', label='Predicted (inverted model)')
    axes[0].set_ylabel(r'u [m]')
    axes[0].set_title(f'Observed vs Predicted Seismic Trace (receiver x[{rec}])')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    residual = u_obs - u_pred
    axes[1].plot(times[:n], residual, color='darkorange', linewidth=1.0,
                 label='Residual (obs − pred)')
    axes[1].axhline(0, color='black', linewidth=0.6, linestyle='--')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Residual [m]')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path = fig_dir / 'observed_vs_predicted.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 7. Loss convergence ───────────────────────────────────────────────────────

def plot_loss_history(opt_results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    iters = opt_results['iteration_history']
    losses = opt_results['loss_history']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, losses, 'o-', color='darkgreen', markersize=4, linewidth=1.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Misfit loss J(μ)')
    ax.set_title('Optimization Convergence: Loss vs Iteration')
    ax.grid(True, alpha=0.25)
    if min(losses) > 0:
        ax.set_yscale('log')
    if opt_results.get('convergence_reached', False):
        ax.text(0.95, 0.95, 'Converged', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color='green')
    fig.tight_layout()
    path = fig_dir / 'loss_history.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 8. mu inversion result ────────────────────────────────────────────────────

def plot_mu_inversion(mu_true, mu_initial, mu_recovered, fig_dir=FIGURES_DIR):
    """
    Plot mu_true vs mu_initial vs mu_recovered (inversion result).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mu_true))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mu_true,      'k-o',  linewidth=2.0, markersize=5,
            label=r'$\mu_{\mathrm{true}}$')
    ax.plot(x, mu_initial,   'b--s', linewidth=1.5, markersize=5,
            label=r'$\mu_{\mathrm{initial}}$')
    ax.plot(np.arange(len(mu_recovered)), mu_recovered,
            'r-^', linewidth=1.8, markersize=5,
            label=r'$\mu_{\mathrm{recovered}}$')
    ax.set_xlabel('Grid index')
    ax.set_ylabel(r'$\mu$ [Pa]')
    ax.set_title(r'Inversion Result: $\mu_{\mathrm{true}}$ vs $\mu_{\mathrm{recovered}}$')
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'mu_inversion.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 9. Classical PDE energy ───────────────────────────────────────────────────

def plot_energy(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results['times'][1:], results['energies'],
            color='black', linewidth=1.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Classical PDE Energy [J/m]')   # Requirement I: correct label
    ax.set_title('Classical PDE Energy vs Time')
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.tight_layout()
    path = fig_dir / 'energy.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 10. Quantum overlap evolution ─────────────────────────────────────────────

def plot_overlap(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not results.get('overlaps'):
        print("  No overlap data — skipping overlap plot.")
        return None
    tvals, ovvals = zip(*results['overlaps'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tvals, ovvals, marker='o', markersize=4, color='blue',
            linewidth=1.2,
            label=r'$|\langle\psi_{\mathrm{ref}}|\psi_{\mathrm{fwd}}(t)\rangle|^2$')
    ax.axhline(float(np.mean(ovvals)), color='black', linestyle='--',
               linewidth=0.8, label=f'Mean = {np.mean(ovvals):.4f}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Squared overlap')
    ax.set_title('Quantum State Overlap vs Time')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'overlap.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: loss per timestep ─────────────────────────────────────────────

def plot_loss(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    loss_arr = results.get('loss', [])
    if not loss_arr:
        return None
    times = results['times'][1:len(loss_arr) + 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, loss_arr, color='darkgreen', linewidth=1.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Quantum Reconstruction Loss vs Time')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'loss.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: model update bar chart ────────────────────────────────────────

def plot_model_update(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_init = results.get('mu_initial', [])
    mu_upd  = results.get('mu_updated', [])
    if not len(mu_init) or not len(mu_upd):
        return None
    x_idx = np.arange(len(mu_init))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_idx - w/2, mu_init, w, label=r'$\mu_{\mathrm{initial}}$',
           color='steelblue')
    ax.bar(x_idx + w/2, mu_upd,  w, label=r'$\mu_{\mathrm{updated}}$',
           color='orangered')
    ax.set_xlabel('Index')
    ax.set_ylabel(r'$\mu$ [Pa]')
    ax.set_title('Model Update: Elastic Modulus')
    ax.legend()
    ax.grid(True, alpha=0.25, axis='y')
    fig.tight_layout()
    path = fig_dir / 'model_update.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: mu evolution over iterations ─────────────────────────────────

def plot_model_evolution(opt_results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_hist = opt_results.get('mu_history', [])
    if not mu_hist:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    step = max(1, len(mu_hist) // 10)
    for i, mu in enumerate(mu_hist):
        if i % step == 0:
            alpha = 0.4 + 0.6 * (i / max(1, len(mu_hist) - 1))
            ax.plot(mu, linewidth=1, alpha=alpha, label=f'Iter {i}')
    ax.set_xlabel('Parameter index')
    ax.set_ylabel(r'$\mu$ [Pa]')
    ax.set_title('Elastic Modulus Evolution During Inversion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'model_evolution.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: circuit diagram ───────────────────────────────────────────────

def plot_circuit(qc, circuit_meta, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    g = circuit_meta.get('group_idx', 0)
    i = circuit_meta.get('time_step_idx', 0)
    fig = qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    fig.suptitle(
        f"Time Evolution Quantum Circuit (Group {g}, Index {i})",
        fontsize=12, fontweight='bold', y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = fig_dir / 'circuit.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig
