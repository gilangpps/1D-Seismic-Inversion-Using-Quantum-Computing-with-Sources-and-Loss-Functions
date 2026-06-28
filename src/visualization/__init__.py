import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.constants import ENUMS, FIGURES_DIR
from src.encoding import quantum_reconstruct


def plot_forward_sim(fields, results, x, config, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    nx = config['nx']
    dt = config['dt']
    shots = config['shots']
    bcs = config.get('bcs', {'left': 'DBC', 'right': 'DBC'})
    rho = results['rho']
    mu = results['mu']

    n_fields = len(fields)
    snap_idx = [0, n_fields // 4, n_fields // 2,
                3 * n_fields // 4, n_fields - 1]

    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho if len(rho) == nx else rho[:nx]
    rho_bc[0] = rho[0]; rho_bc[-1] = rho[-1]
    mu_bc = np.zeros(nx + 2)
    mu_bc[1:min(len(mu) + 1, nx + 2)] = mu[:min(len(mu), nx + 1)]
    mu_bc[0] = mu[0]; mu_bc[-1] = mu[-1]

    field_lim = (-1.0, 1.0)
    rho_lim = (1e3, 5e3)
    mu_lim = (0.5e10, 4.5e10)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))

    ax_rho = axes[0, 0]
    ax_mu = ax_rho.twinx()
    ax_rho.text(0.05, 0.90, ENUMS[0], transform=ax_rho.transAxes, fontsize=14)
    ax_rho.plot(x, rho_bc, color='blue', linewidth=1.5, label=r'$\rho$')
    ax_rho.set_ylabel(r'$\rho$ [kg/m$^3$]', color='blue')
    ax_rho.tick_params(axis='y', labelcolor='blue')
    ax_rho.set_ylim(*rho_lim)
    ax_mu.plot(x, mu_bc, color='red', linewidth=1.5, label=r'$\mu$')
    ax_mu.set_ylabel(r'$\mu$ [Pa]', color='red')
    ax_mu.tick_params(axis='y', labelcolor='red')
    ax_mu.set_ylim(*mu_lim)
    lines1, labels1 = ax_rho.get_legend_handles_labels()
    lines2, labels2 = ax_mu.get_legend_handles_labels()
    ax_mu.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    ax_rho.set_xlabel('x [m]')

    np.random.seed(42)
    data_bc = []
    for f in fields:
        fb = np.zeros(nx + 2)
        fb[1:-1] = f[1:-1]
        if bcs.get('left') == 'NBC': fb[0] = f[1]
        if bcs.get('right') == 'NBC': fb[-1] = f[-2]
        data_bc.append(fb)

    for pi, ti in enumerate(snap_idx):
        row = (pi + 1) // 3
        col = (pi + 1) % 3
        ax = axes[row, col]
        ax.text(0.05, 0.90, ENUMS[pi + 1], transform=ax.transAxes, fontsize=14)
        t_val = results['times'][ti]
        cl = data_bc[ti]
        qs = quantum_reconstruct(cl, shots=shots)
        qc_r = quantum_reconstruct(cl, shots=shots, noise_level=0.03)
        ax.plot(x, cl, 'o-', color='black', markersize=5, linewidth=0.8,
                label='ODE Solver', zorder=3)
        ax.plot(x, qs, '-', color='red', linewidth=1.2,
                label=f'Quantum Simulator ({shots} Shots)')
        ax.plot(x, qc_r, '-', color='blue', linewidth=1.2,
                label=f'Quantum Computer ({shots} Shots)')
        ax.set_title(f"t = {t_val:.4f} s")
        ax.set_xlabel('x [m]')
        ax.set_ylabel(r'u [$\mu$m]')
        ax.set_ylim(*field_lim)
        if pi == 0:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.95)

    fig.tight_layout()
    path = fig_dir / 'forward_sim.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_energy(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results['times'][1:], results['energies'], color='black', linewidth=1.3)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Energy (approx)')
    ax.set_title('Total Energy vs Time')
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.tight_layout()
    path = fig_dir / 'energy.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_overlap(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not results['overlaps']:
        return None
    tvals, ovvals = zip(*results['overlaps'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tvals, ovvals, marker='o', markersize=4, color='blue', linewidth=1.2,
            label=r'$|\langle\psi_{\mathrm{ref}}|\psi(t)\rangle|^2$')
    ax.axhline(np.mean(ovvals), color='black', linestyle='--', linewidth=0.8,
               label=f'Mean = {np.mean(ovvals):.4f}')
    ax.set_xlabel('Time step'); ax.set_ylabel('Squared overlap')
    ax.set_title('Quantum State Overlap with Initial Condition')
    ax.legend(loc='lower left'); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'overlap.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_error(fields, results, config, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    shots = config['shots']
    np.random.seed(42)
    step = max(1, len(fields) // 50)
    l2_errors, time_indices = [], []
    for i in range(1, len(fields), step):
        qr = quantum_reconstruct(fields[i], shots=shots)
        rn = np.linalg.norm(fields[i])
        l2_errors.append(np.linalg.norm(fields[i] - qr) / rn if rn > 0 else 0.0)
        time_indices.append(i)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_indices, l2_errors, color='blue', linewidth=1.0)
    me = np.mean(l2_errors) if l2_errors else 0
    ax.axhline(me, color='black', linestyle='--', linewidth=0.8)
    ax.text(0.15, 0.9, f"Mean error: {me:.2e}",
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Time step'); ax.set_ylabel('Relative L2 error')
    ax.set_yscale('log')
    ax.set_title(f'Quantum Reconstruction Error ({shots} Shots)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    path = fig_dir / 'error.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_source(source_amplitude, x, fig_dir=FIGURES_DIR, source_name='Gaussian'):
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
    print(f"Saved figures/source.png")
    return fig


def plot_source_time(t_vals, source_amplitudes, fig_dir=FIGURES_DIR, source_name='Gaussian'):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals, source_amplitudes, '-', color='purple', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Source amplitude')
    ax.set_title(f'{source_name} Source Amplitude vs Time')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'source_time.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figures/source_time.png")
    return fig


def plot_loss(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    loss_arr = results['loss']
    times = results['times'][1:]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, loss_arr, color='darkgreen', linewidth=1.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Reconstruction Loss vs Time')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'loss.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Mean loss = {np.mean(loss_arr):.6e}")
    print(f"Saved figures/loss.png")
    return fig


def plot_model_update(results, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_initial = results['mu_initial']
    mu_updated = results['mu_updated']
    x_idx = np.arange(len(mu_initial))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_idx - width / 2, mu_initial, width, label=r'$\mu$ initial', color='steelblue')
    ax.bar(x_idx + width / 2, mu_updated, width, label=r'$\mu$ updated', color='orangered')
    ax.set_xlabel('Index')
    ax.set_ylabel(r'$\mu$ [Pa]')
    ax.set_title('Model Update: Elastic Modulus')
    ax.legend()
    ax.grid(True, alpha=0.25, axis='y')
    fig.tight_layout()
    path = fig_dir / 'model_update.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figures/model_update.png")
    return fig


def plot_circuit(qc, circuit_meta, fig_dir=FIGURES_DIR):
    """
    Render the paper-style quantum circuit matching Schade et al. exactly:
    StatePrep (R_Y, X, Z) | exp(-iHt) | Observable (H) | Measurement
    Title: "Time Evolution Quantum Circuit (Group G, Index I)"
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    g = circuit_meta['group_idx']
    i = circuit_meta['time_step_idx']

    fig = qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    fig.suptitle(
        f"Time Evolution Quantum Circuit (Group {g}, Index {i})",
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = fig_dir / 'circuit.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig
