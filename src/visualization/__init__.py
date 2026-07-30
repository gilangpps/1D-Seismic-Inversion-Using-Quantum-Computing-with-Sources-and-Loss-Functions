"""
src/visualization/__init__.py
Publication-quality plots for 1-D Seismic Inversion (Requirement K).

Mandatory plots (10):
  1. plot_initial_mu          — initial mu model vs true
  2. plot_density_model       — density model rho(x)
  3. plot_source              — source amplitude vs position
  4. plot_forward_sim         — wave propagation snapshots (2×3 multiplot)
  5. plot_error               — quantum reconstruction relative L2 error
  6. plot_observed_vs_predicted — observed vs predicted seismic trace + residual
  7. plot_loss_history        — inversion misfit J(mu) per iteration
  8. plot_mu_inversion        — mu_true vs mu_initial vs mu_recovered
  9. plot_energy              — classical PDE energy [J/m] vs time
  10. plot_overlap            — quantum overlap |<psi_rec|psi_ref>|^2 vs time

Additional:
  plot_source_time            — source wavelet vs time
  plot_loss                   — per-timestep INVERSION misfit J(t; mu_rec)
  plot_quantum_recon_loss     — per-timestep quantum reconstruction MSE (diagnostic)
  plot_model_update           — bar chart: mu_initial vs mu_updated
  plot_model_evolution        — mu parameter evolution across iterations
  plot_circuit                — quantum circuit diagram

Change log vs prior version
-----------------------------
* plot_loss() now plots results['loss'] which is the per-timestep INVERSION
  misfit J(t; mu_rec), not the quantum reconstruction MSE.
  Y-axis label changed to 'Inversion misfit MSE' and title updated accordingly.
* plot_quantum_recon_loss() added as a new function to separately visualise
  results['quantum_recon_loss'] (encoding quality diagnostic).
  This keeps the two quantities clearly distinct in both plots and data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.constants import ENUMS, FIGURES_DIR
from src.encoding import quantum_reconstruct


# ── 1. Initial mu model ───────────────────────────────────────────────────────

def plot_initial_mu(mu_initial, mu_true=None, dx=1.0, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth = np.arange(len(mu_initial)) * dx
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(mu_initial, depth, 'o-', color='steelblue', linewidth=1.8,
            markersize=5, label=r'$\mu_{\mathrm{initial}}$')
    if mu_true is not None:
        ax.plot(mu_true, np.arange(len(mu_true)) * dx, 's--', color='red',
                linewidth=1.5, markersize=5, label=r'$\mu_{\mathrm{true}}$')
    ax.set_ylabel('Depth z [m]')
    ax.set_xlabel(r'$\mu$ [Pa]')
    ax.set_title('Initial Elastic Modulus Model')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'mu_initial.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 2. Density model ──────────────────────────────────────────────────────────

def plot_mu_model(mu_arr, dx=1.0, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth = np.arange(len(mu_arr)) * dx
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(mu_arr, depth, 'o-', color='seagreen', linewidth=1.8, markersize=5)
    ax.set_ylabel('Depth z [m]')
    ax.set_xlabel(r'$\mu$ [Pa]')
    ax.set_title('Elastic Modulus Model')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'mu_model.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_density_model(rho_arr, dx=1.0, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth = np.arange(len(rho_arr)) * dx
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(rho_arr, depth, 'o-', color='darkorange', linewidth=1.8, markersize=5)
    ax.set_ylabel('Depth z [m]')
    ax.set_xlabel(r'$\rho$ [kg/m$^3$]')
    ax.set_title('Density Model')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'density_model.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Elastic wave velocity model, v = sqrt(mu/rho) ────────────────────────────

def plot_velocity_model(mu_arr, rho_arr, dx=1.0, fig_dir=FIGURES_DIR):
    """
    Elastic wave velocity model, v(z) = sqrt(mu(z) / rho(z)).

    NOTE ON NAMING: this project's wave equation only carries a single
    elastic modulus (mu), i.e. rho*u_tt = d/dx(mu * du/dx) (Schade et al.
    2024/2025). This is the same form as the 1-D shear/elastic wave
    equation, so v = sqrt(mu/rho) is the associated wave speed (analogous
    to Vs). It is NOT the P-wave velocity Vp = sqrt((lambda+2mu)/rho),
    since lambda (the first Lame parameter) is not part of this model.

    Depth is placed on the vertical axis, increasing downward, following
    the standard 1-D earth-model / well-log convention in geophysics.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_arr = np.asarray(mu_arr, dtype=float)
    rho_arr = np.asarray(rho_arr, dtype=float)
    v_arr = np.sqrt(mu_arr / rho_arr)
    depth = np.arange(len(v_arr)) * dx

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(v_arr, depth, 'o-', color='teal', linewidth=1.8, markersize=5)
    ax.set_ylabel('Depth z [m]')
    ax.set_xlabel(r'$v_s = \sqrt{\mu/\rho}$ [m/s]')
    ax.set_title('Shear Wave Velocity Model ($v_s$)')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'velocity_model.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_velocity_inversion(mu_true, mu_initial, mu_recovered, rho_arr,
                             dx=1.0, fig_dir=FIGURES_DIR):
    """
    Plot v_true vs v_initial vs v_recovered, where v = sqrt(mu/rho)
    (inversion result expressed as elastic wave velocity).

    Density (rho_arr) is treated as known/fixed and shared across all
    three curves, since only mu is inverted for in this project.

    Depth is placed on the vertical axis, increasing downward, following
    the standard 1-D earth-model / well-log convention in geophysics.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_true = np.asarray(mu_true, dtype=float)
    mu_initial = np.asarray(mu_initial, dtype=float)
    mu_recovered = np.asarray(mu_recovered, dtype=float)
    rho_arr = np.asarray(rho_arr, dtype=float)

    v_true = np.sqrt(mu_true / rho_arr)
    v_initial = np.sqrt(mu_initial / rho_arr)
    v_recovered = np.sqrt(mu_recovered / rho_arr)
    depth = np.arange(len(v_true)) * dx

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(v_true, depth, 'k-o', linewidth=2.0, markersize=5,
            label=r'$v_{s,\mathrm{true}}$')
    ax.plot(v_initial, depth, 'b--s', linewidth=1.5, markersize=5,
            label=r'$v_{s,\mathrm{initial}}$')
    ax.plot(v_recovered, depth, 'r-^', linewidth=1.8, markersize=5,
            label=r'$v_{s,\mathrm{recovered}}$')
    ax.set_ylabel('Depth z [m]')
    ax.set_xlabel(r'$v_s = \sqrt{\mu/\rho}$ [m/s]')
    ax.set_title(
        r'Inversion Result: $v_{s,\mathrm{true}}$ vs $v_{s,\mathrm{recovered}}$'
    )
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'velocity_inversion.png'
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


def plot_source_time(t_vals, source_amplitudes, fig_dir=FIGURES_DIR,
                     source_name='Ricker'):
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



# ── Source & receiver geometry (depth-domain, mirrors Devito-style plots) ────

def plot_source_receiver_geometry(mu_true, rho_arr=None, dx=1.0, source_idx=None,
                                   receiver_indices=None, fig_dir=FIGURES_DIR):
    """
    Show where the source and receivers sit within the 1-D depth model.

    Depth is placed on the vertical axis, increasing downward (same
    convention as plot_mu_model / plot_density_model). The background
    curve is the shear-wave velocity v_s = sqrt(mu/rho) rather than the
    raw elastic modulus mu — this matches the units/scale used in the
    Devito reference plot (velocity in m/s), so the geometry is easier to
    read together with the velocity model figures. The source is marked
    with a red star and the receivers with green dots, analogous to
    Devito's plot_velocity() source/receiver overlay.

    NOTE ON PLACEMENT: this is a strictly 1-D problem (depth is the only
    spatial axis — there is no separate lateral "x" direction like in a
    2-D Devito section). So, unlike the 2-D reference image, source and
    receivers cannot be spread out sideways; they only have a depth
    coordinate. Placing the single source at the centre of the depth
    axis (the default) is a standard, physically sensible choice for a
    single common-shot experiment — it maximises the offset to both the
    shallow and deep receivers. Pass a different `source_idx` to move it.

    Parameters
    ----------
    mu_true : array-like
        True elastic-modulus profile, length nx.
    rho_arr : array-like or None
        Density profile, length nx. If provided, the background curve is
        velocity v_s = sqrt(mu_true/rho_arr) [m/s]; if None, falls back to
        plotting mu_true [Pa] directly (legacy behaviour).
    dx : float
        Grid spacing [m].
    source_idx : int or None
        Grid index (0-based, into mu_true) of the source. Defaults to nx//2
        (the source is placed at the centre of the domain in this project).
    receiver_indices : sequence of int or None
        Grid indices (0-based, into mu_true) of the receivers. Defaults to
        every grid point (this project's inversion loss uses the full
        7-point grid, i.e. every point is a receiver).
    fig_dir : Path
        Output directory for the figure.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_true = np.asarray(mu_true, dtype=float)
    nx = len(mu_true)
    depth = np.arange(nx) * dx

    if rho_arr is not None:
        rho_arr = np.asarray(rho_arr, dtype=float)
        curve = np.sqrt(mu_true / rho_arr)
        xlabel = r'$v_s = \sqrt{\mu/\rho}$ [m/s]'
        curve_label = r'$v_{s,\mathrm{true}}$ (model)'
    else:
        curve = mu_true
        xlabel = r'$\mu$ [Pa]'
        curve_label = r'$\mu_{\mathrm{true}}$ (model)'

    if source_idx is None:
        source_idx = nx // 2
    if receiver_indices is None:
        receiver_indices = list(range(nx))

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(curve, depth, 'o-', color='lightsteelblue', linewidth=1.5,
            markersize=4, zorder=1, label=curve_label)
    ax.scatter(curve[receiver_indices], depth[receiver_indices],
               s=70, facecolors='none', edgecolors='green', linewidths=1.8,
               zorder=2, label=f'Receivers (n={len(receiver_indices)})')
    ax.scatter(curve[source_idx], depth[source_idx],
               marker='*', s=260, color='red', edgecolors='black',
               linewidths=0.6, zorder=3, label=f'Source (z={depth[source_idx]:.0f} m)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth z [m]')
    ax.set_title('Source and Receiver Geometry')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'source_receiver_geometry.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Per-iteration inversion process panels (mirrors advisor's reference figure) ─

def plot_inversion_process_grid(mu_history, rho_arr, dx=1.0, noise_level=None,
                                 fig_dir=FIGURES_DIR, n_panels=10):
    """
    Reproduce the "Initial Model / 1st Iteration / 2nd Iteration / ..."
    panel-row figure (velocity heatmap vs depth, one column per iteration)
    requested by the advisor, for either noiseless or noisy inversion runs.

    Parameters
    ----------
    mu_history : list of array-like
        mu_arr snapshot at each optimizer iteration (opt_results['mu_history']),
        index 0 = initial model, last index = final/converged model.
    rho_arr : array-like
        Density profile, length nx (used to convert mu -> v_s = sqrt(mu/rho)).
    dx : float
        Grid spacing [m].
    noise_level : float or None
        Fraction (e.g. 0.05 for 5%) of noise used for this run, for the
        title/filename. None means a noiseless run.
    fig_dir : Path
        Output directory for the figure.
    n_panels : int
        Number of panels to show (evenly sampled across mu_history if it
        contains more iterations than this).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    rho_arr = np.asarray(rho_arr, dtype=float)
    n_iters = len(mu_history)
    if n_iters <= n_panels:
        idxs = list(range(n_iters))
    else:
        idxs = sorted(set(np.linspace(0, n_iters - 1, n_panels).astype(int).tolist()))

    velocities = [np.sqrt(np.asarray(mu_history[i], dtype=float) / rho_arr) for i in idxs]
    nx = len(rho_arr)
    depth = np.arange(nx) * dx
    vmin = min(v.min() for v in velocities)
    vmax = max(v.max() for v in velocities)

    n = len(idxs)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    im = None
    for k, (ax, v, it) in enumerate(zip(axes, velocities, idxs)):
        img = v.reshape(-1, 1)
        im = ax.imshow(img, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                        extent=[0, 1, depth[-1] + dx / 2, depth[0] - dx / 2])
        title = 'Initial Model' if it == 0 else f'Iter {it}'
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        if k == 0:
            ax.set_ylabel('Depth z [m]')

    fig.subplots_adjust(right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r'$v_s$ [m/s]')

    if noise_level is not None:
        suptitle = f'Inversion Process ({noise_level * 100:.3g}% Noise)'
        fname = f'inversion_process_{noise_level * 100:.3g}pct_noise.png'
    else:
        suptitle = 'Inversion Process (Noiseless Data)'
        fname = 'inversion_process_noiseless.png'
    fig.suptitle(suptitle, y=1.02)

    path = fig_dir / fname
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_noise_level_summary(noise_pcts, relative_errors, fig_dir=FIGURES_DIR):
    """
    Summary plot: final relative model error vs. noise level, across the
    noise-robustness sweeps (e.g. 1%, 2.5%, 5%, 10%).

    Parameters
    ----------
    noise_pcts : sequence of float
        Noise levels in percent, e.g. [0, 1, 2.5, 5, 10].
    relative_errors : sequence of float
        Corresponding final ||mu_recovered - mu_true|| / ||mu_true|| (fraction,
        not percent) for each noise level.
    fig_dir : Path
        Output directory for the figure.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(noise_pcts)
    x = np.asarray(noise_pcts, dtype=float)[order]
    y = np.asarray(relative_errors, dtype=float)[order] * 100.0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, y, 'o-', color='crimson', linewidth=1.8, markersize=6)
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.1f}%', (xi, yi), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xlabel('Noise level [%]')
    ax.set_ylabel('Relative model error [%]')
    ax.set_title('Inversion Robustness vs. Noise Level')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'noise_level_summary.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 4. Forward simulation multiplot ──────────────────────────────────────────

def plot_forward_sim(fields, results, x, config, fig_dir=FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    nx    = config['nx']
    dt    = config['dt']
    shots = config['shots']
    rho   = results['rho']
    mu    = results['mu']

    n_fields = len(fields)
    snap_idx = [0, n_fields // 4, n_fields // 2,
                3 * n_fields // 4, n_fields - 1]

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
    ax_rho.plot(x, rho_bc, color='blue',  linewidth=1.5, label=r'$\rho$')
    ax_rho.set_ylabel(r'$\rho$ [kg/m$^3$]', color='blue')
    ax_rho.tick_params(axis='y', labelcolor='blue')
    ax_mu.plot(x,  mu_bc,  color='red',   linewidth=1.5, label=r'$\mu$')
    ax_mu.set_ylabel(r'$\mu$ [Pa]', color='red')
    ax_mu.tick_params(axis='y', labelcolor='red')
    lines  = (ax_rho.get_legend_handles_labels()[0]
              + ax_mu.get_legend_handles_labels()[0])
    ax_mu.legend(lines, [r'$\rho$', r'$\mu$'], loc='lower left', fontsize=8)
    ax_rho.set_xlabel('x [m]')

    np.random.seed(42)
    for pi, ti in enumerate(snap_idx):
        row = (pi + 1) // 3
        col = (pi + 1) % 3

        ax = axes[row, col]
        ax.text(
            0.05, 0.90, ENUMS[pi + 1],
            transform=ax.transAxes,
            fontsize=13
        )

        t_val = results['times'][ti]

        # Snapshot pertama
        if ti == 0:
            cl = np.zeros_like(fields[0])
            center = len(cl) // 2
            cl[center] = 1.0

            qs = cl.copy()
            qc_r = cl.copy()

        # Snapshot berikutnya
        else:
            cl = fields[ti]

            qs = quantum_reconstruct(
                cl,
                shots=shots
            )

            qc_r = quantum_reconstruct(
                cl,
                shots=shots,
                noise_level=0.01
            )

        ax.plot(
            x, cl,
            'o-',
            color='black',
            markersize=4,
            linewidth=0.8,
            label='Classical PDE',
            zorder=3
        )

        ax.plot(
            x, qs,
            '-',
            color='red',
            linewidth=1.2,
            label=f'Quantum sim. ({shots} shots)'
        )

        ax.plot(
            x, qc_r,
            '-',
            color='blue',
            linewidth=1.2,
            label=f'Quantum + noise ({shots} shots)'
        )

        ax.set_title(f"t = {t_val:.4f} s")
        ax.set_xlabel('x [m]')
        ax.set_ylabel('u [m]')

        ymin = min(np.min(cl), np.min(qs), np.min(qc_r))
        ymax = max(np.max(cl), np.max(qs), np.max(qc_r))

        margin = max(0.05, 0.1 * (ymax - ymin))
        ax.set_ylim(ymin - margin, ymax + margin)

        if pi == 0:
            ax.legend(loc='upper right', fontsize=7)

    fig.tight_layout()
    path = fig_dir / 'forward_sim.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 5. Quantum reconstruction error ──────────────────────────────────────────

def plot_error(fields, results, config, fig_dir=FIGURES_DIR):
    """
    Relative L2 error between classical field and quantum-reconstructed field.

    error(t) = ‖u_classical(t) − u_quantum(t)‖ / ‖u_classical(t)‖

    This is an encoding quality metric, not the inversion misfit.
    """
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
    Plot observed (true-model) vs predicted (recovered-model) seismic traces.

    Shows how well the inverted model reproduces the reference data.

    Parameters
    ----------
    fields_obs  : list — reference (true-model) wavefields
    fields_pred : list — recovered-model wavefields
    results     : dict with 'times'
    config      : dict with 'nx'
    receiver_idx : int or None — grid index of receiver (default: nx//2)
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    nx  = config['nx']
    rec = receiver_idx if receiver_idx is not None else nx // 2

    times = results['times']
    n     = min(len(fields_obs), len(fields_pred), len(times))

    u_obs  = np.array([fields_obs[i][rec + 1]  for i in range(n)])
    u_pred = np.array([fields_pred[i][rec + 1] for i in range(n)])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(times[:n], u_obs,  color='black', linewidth=1.3,
                 label='Observed (true model)')
    axes[0].plot(times[:n], u_pred, color='red',   linewidth=1.2,
                 linestyle='--', label='Predicted (inverted model)')
    axes[0].set_ylabel(r'u [m]')
    axes[0].set_title(
        f'Observed vs Predicted Seismic Trace (receiver x[{rec}])'
    )
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


# ── Forward data: clean vs noisy comparison ──────────────────────────────────

def plot_noisy_data_comparison(t_vals, clean_trace, noisy_trace,
                                fig_dir=FIGURES_DIR, receiver_idx=None,
                                noise_level=None, snr_db=None):
    """
    Compare clean (noiseless) vs noisy forward-simulated data at one
    receiver, plus the injected noise component on its own — analogous to
    noiseless-vs-noisy synthetic-data comparisons commonly shown in
    traveltime/waveform tomography studies.

    Parameters
    ----------
    t_vals : array_like — time axis [s]
    clean_trace : array_like — noiseless u(t) at the chosen receiver
    noisy_trace : array_like — noisy u(t) at the same receiver
    receiver_idx : int or None — grid index of the receiver (for the title)
    noise_level : float or None — noise level used (fraction), for the title
    snr_db : float or None — achieved SNR in dB, for the title
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(t_vals), len(clean_trace), len(noisy_trace))
    t_vals = np.asarray(t_vals)[:n]
    clean_trace = np.asarray(clean_trace)[:n]
    noisy_trace = np.asarray(noisy_trace)[:n]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_vals, clean_trace, color='black', linewidth=1.4,
                 label='Clean (noiseless) forward data')
    axes[0].plot(t_vals, noisy_trace, color='crimson', linewidth=1.0,
                 alpha=0.85, label='Noisy forward data (observed)')
    title = 'Forward Data: Clean vs Noisy'
    if receiver_idx is not None:
        title += f' (receiver x[{receiver_idx}])'
    axes[0].set_ylabel('u [m]')
    axes[0].set_title(title)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    subtitle_bits = []
    if noise_level is not None:
        subtitle_bits.append(f'noise level = {noise_level * 100:.1f}%')
    if snr_db is not None:
        subtitle_bits.append(f'SNR ≈ {snr_db:.1f} dB')
    if subtitle_bits:
        axes[0].text(
            0.99, 0.02, ', '.join(subtitle_bits),
            transform=axes[0].transAxes, ha='right', va='bottom', fontsize=8,
            color='dimgray',
        )

    noise_only = noisy_trace - clean_trace
    axes[1].plot(t_vals, noise_only, color='steelblue', linewidth=1.0)
    axes[1].axhline(0, color='black', linewidth=0.6, linestyle='--')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Noise [m]')
    axes[1].set_title('Injected Noise Component')
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path = fig_dir / 'forward_data_noise_comparison.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── 7. Loss convergence ───────────────────────────────────────────────────────

def plot_loss_history(opt_results, fig_dir=FIGURES_DIR):
    """
    Plot inversion misfit J(μ) per Adam iteration.

    Data source: opt_results['loss_history'] from SeismicOptimizer.
    This is the authoritative inversion convergence curve.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    iters  = opt_results['iteration_history']
    losses = opt_results['loss_history']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, losses, 'o-', color='darkgreen', markersize=4, linewidth=1.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Misfit loss $J(\mu)$')
    ax.set_title('Optimization Convergence: Inversion Loss vs Iteration')
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

def plot_mu_inversion(mu_true, mu_initial, mu_recovered, dx=1.0, fig_dir=FIGURES_DIR):
    """
    Plot mu_true vs mu_initial vs mu_recovered (inversion result comparison).

    Depth is placed on the vertical axis, increasing downward, following the
    standard 1-D earth-model / well-log convention in geophysics (depth is
    vertical, not horizontal). Depth is given in metres (grid_index * dx).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth = np.arange(len(mu_true)) * dx

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(mu_true, depth,    'k-o',  linewidth=2.0, markersize=5,
            label=r'$\mu_{\mathrm{true}}$')
    ax.plot(mu_initial, depth, 'b--s', linewidth=1.5, markersize=5,
            label=r'$\mu_{\mathrm{initial}}$')
    ax.plot(mu_recovered, np.arange(len(mu_recovered)) * dx,
            'r-^', linewidth=1.8, markersize=5,
            label=r'$\mu_{\mathrm{recovered}}$')
    ax.set_ylabel('Depth z [m]')
    ax.set_xlabel(r'$\mu$ [Pa]')
    ax.set_title(
        r'Inversion Result: $\mu_{\mathrm{true}}$ vs $\mu_{\mathrm{recovered}}$'
    )
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
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
    """
    Classical PDE energy vs time.

    E(t) = ½Σᵢ[ρᵢvᵢ² + μᵢ(∂u/∂x)ᵢ²]·dx
    Label: 'Classical PDE Energy [J/m]'  (Requirement I)
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results['times'][1:], results['energies'],
            color='black', linewidth=1.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Classical PDE Energy [J/m]')   # correct label (Req. I)
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
    """
    Quantum state overlap vs time.

    Overlap = |<ψ_fwd(t; μ_rec) | ψ_ref(t; μ_true)>|²

    Inversion quality metric: 0 for a completely wrong model, 1 for perfect
    recovery. Rises as the optimizer reduces J(μ).
    """
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


# ── Additional: per-timestep INVERSION misfit ─────────────────────────────────

def plot_loss(results, fig_dir=FIGURES_DIR):
    """
    Per-timestep inversion misfit J(t; mu_rec).

    J(t; μ_rec) = (1/N) ‖u_fwd(t; μ_rec) − u_ref(t; μ_true)‖²

    This is the INVERSION objective evaluated at each time step using the
    recovered model.  Different from quantum reconstruction MSE.
    Data source: results['loss'] (set in main.py after inversion completes).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    loss_arr = results.get('loss', [])
    if not len(loss_arr):
        print("  No inversion loss data — skipping loss plot.")
        return None
    times = results['times'][1:len(loss_arr) + 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, loss_arr, color='darkgreen', linewidth=1.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Inversion misfit MSE $J(t;\,\mu_{\mathrm{rec}})$')
    ax.set_title('Per-Timestep Inversion Misfit (Recovered Model vs Reference)')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'loss.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: quantum reconstruction MSE (encoding diagnostic) ──────────────

def plot_quantum_recon_loss(results, fig_dir=FIGURES_DIR):
    """
    Per-timestep quantum reconstruction MSE (encoding quality diagnostic).

    MSE(t) = (1/N) ‖u_classical(t) − u_quantum_reconstructed(t)‖²

    This measures how accurately amplitude encoding + shot-noise measurement
    can reconstruct the classical wavefield.  It is an encoding quality metric,
    NOT the inversion objective.
    Data source: results['quantum_recon_loss'].
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    qrl = results.get('quantum_recon_loss', [])
    if not len(qrl):
        print("  No quantum recon loss data — skipping quantum_recon_loss plot.")
        return None
    times = results['times'][1:len(qrl) + 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, qrl, color='purple', linewidth=1.3,
            label='Reference fields')
    qrl_rec = results.get('quantum_recon_loss_rec', [])
    if len(qrl_rec):
        t_rec = results['times'][1:len(qrl_rec) + 1]
        ax.plot(t_rec, qrl_rec, color='darkorange', linewidth=1.3,
                linestyle='--', label='Recovered-model fields')
        ax.legend(fontsize=9)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Quantum reconstruction MSE')
    ax.set_title('Quantum Reconstruction MSE vs Time (Encoding Diagnostic)')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / 'quantum_recon_loss.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: model update bar chart ────────────────────────────────────────

def plot_model_update(results, dx=1.0, fig_dir=FIGURES_DIR):
    """
    Bar chart: mu_initial vs mu_updated.

    Depth is placed on the vertical axis, increasing downward, following the
    standard 1-D earth-model / well-log convention in geophysics (depth is
    vertical, not horizontal). The mu axis (bottom) is plotted with actual
    depth in metres (grid_index * dx).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_init = results.get('mu_initial', [])
    mu_upd  = results.get('mu_updated', [])
    if not len(mu_init) or not len(mu_upd):
        return None
    depth = np.arange(len(mu_init)) * dx
    w = (dx if dx else 1.0) * 0.35
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(depth - w / 2, mu_init, height=w,
            label=r'$\mu_{\mathrm{initial}}$', color='steelblue')
    ax.barh(depth + w / 2, mu_upd,  height=w,
            label=r'$\mu_{\mathrm{updated}}$', color='orangered')
    ax.set_xlabel(r'$\mu$ [Pa]')
    ax.set_ylabel('Depth z [m]')
    ax.set_title('Model Update: Elastic Modulus')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
    ax.legend()
    ax.grid(True, alpha=0.25, axis='x')
    fig.tight_layout()
    path = fig_dir / 'model_update.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ── Additional: mu evolution over iterations ─────────────────────────────────

def plot_model_evolution(opt_results, dx=1.0, fig_dir=FIGURES_DIR):
    """
    mu parameter evolution across iterations.

    Depth is placed on the vertical axis, increasing downward, following the
    standard 1-D earth-model / well-log convention in geophysics (depth is
    vertical, not horizontal). The mu axis (bottom) is plotted against actual
    depth in metres (grid_index * dx).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    mu_hist = opt_results.get('mu_history', [])
    if not mu_hist:
        return None
    depth = np.arange(len(mu_hist[0])) * dx
    fig, ax = plt.subplots(figsize=(6, 8))
    step = max(1, len(mu_hist) // 10)
    for i, mu in enumerate(mu_hist):
        if i % step == 0:
            alpha = 0.4 + 0.6 * (i / max(1, len(mu_hist) - 1))
            ax.plot(mu, depth, 'o-', linewidth=1, markersize=3,
                    alpha=alpha, label=f'Iter {i}')
    ax.set_xlabel(r'$\mu$ [Pa]')
    ax.set_ylabel('Depth z [m]')
    ax.set_title('Elastic Modulus Evolution During Inversion')
    ax.invert_yaxis()          # depth increases downward (1-D earth model convention)
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
    from src.circuit import validate_paper_circuit
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    g = circuit_meta.get('group_idx', 0)
    i = circuit_meta.get('time_step_idx', 0)
    
    # Validate circuit if it's in paper diagram mode
    paper_mode = circuit_meta.get('paper_diagram_mode', False)
    if paper_mode:
        try:
            validate_paper_circuit(qc, paper_diagram_mode=True)
            print(f"  [OK] Circuit validation passed (paper diagram mode)")
        except AssertionError as e:
            print(f"  [WARNING] Circuit validation failed: {e}")
    
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




# ── Hamiltonian Validation ────────────────────────────────────────────────────

def plot_hamiltonian_validation(validation, fig_dir=FIGURES_DIR):
    """
    Plot Hamiltonian validation results: quantum exp(-iHt) vs classical PDE.

    This validates that H = i(A − Aᵀ)/2 captures elastic wave physics by
    comparing independent quantum and classical trajectories.

    DISTINCTION FROM OTHER PLOTS:
        • overlap.png: inversion quality |⟨ψ_rec|ψ_ref⟩|² (how well μ_rec fits μ_true)
        • error.png: encoding quality (amplitude encode-decode fidelity)
        • THIS PLOT: Hamiltonian physics validation (quantum vs classical evolution)

    Parameters
    ----------
    validation : dict
        Output from run_hamiltonian_validation() with keys:
        'time', 'l2_error', 'overlap', 'energy_classical', 'norm_quantum'.
    fig_dir : Path
        Output directory for figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    time = validation['time']
    l2_error = validation['l2_error']
    overlap = validation['overlap']
    energy = validation['energy_classical']
    norm_q = validation['norm_quantum']

    # Create 2×2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Hamiltonian Validation: Quantum exp(−iHt) vs Classical Leapfrog PDE',
        fontsize=14, fontweight='bold'
    )

    # ── Subplot 1: L2 Error ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(time, l2_error, color='red', linewidth=1.5, label='L2 error')
    ax.axhline(
        np.mean(l2_error[1:]), color='black', linestyle='--',
        linewidth=1.0, label=f'Mean = {np.mean(l2_error[1:]):.4f}'
    )
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Relative L2 Error: $\|u_q - u_c\| / \|u_c\|$')
    ax.set_title('Trajectory L2 Error (Quantum vs Classical)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Subplot 2: State Overlap ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(time, overlap, color='blue', linewidth=1.5, label='Overlap')
    ax.axhline(
        np.mean(overlap[1:]), color='black', linestyle='--',
        linewidth=1.0, label=f'Mean = {np.mean(overlap[1:]):.6f}'
    )
    ax.axhline(0.95, color='green', linestyle=':', linewidth=1.0, label='Target = 0.95')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'State Overlap $|\langle\psi_q|\psi_c\rangle|^2$')
    ax.set_title('Quantum State Overlap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Subplot 3: Classical Energy ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(time, energy, color='purple', linewidth=1.5, label='Classical PDE Energy')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J/m]')
    ax.set_title('Classical PDE Energy (Should be conserved)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Subplot 4: Quantum Norm ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(time, norm_q, color='orange', linewidth=1.5, label='Quantum State Norm')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, label='Expected = 1.0')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Norm $\|\psi_q\|$')
    ax.set_title('Quantum State Norm (Should be ~1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.2, np.max(norm_q) * 1.1))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = fig_dir / 'hamiltonian_validation.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return fig
