"""
src/persistence/__init__.py
Excel workbook and file persistence for 1-D Seismic Inversion.

Sheets (Requirement J):
  1. Configuration      — experiment parameters
  2. Medium             — x, rho, mu
  3. TimeSeries         — time, classical PDE energy
  4. Overlaps           — time, squared quantum overlap
  5. WaveFields         — full displacement field per time step
  6. CircuitParams      — quantum circuit metadata
  7. Source             — source amplitude vs position
  8. Loss               — per-timestep reconstruction MSE
  9. ModelUpdate        — mu_initial vs mu_updated
  10. OptimizationHistory — iteration, loss, overlap, mu per iteration
"""

import datetime
import json
import pickle

import numpy as np
import openpyxl

from src.constants import DATA_DIR


def save_experiment(fields, results, x, config, data_dir=DATA_DIR):
    """Save experiment fields and metadata to JSON + pickle."""
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_dir   = data_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # JSON config
    json.dump(config, open(exp_dir / 'configs.json', 'w', encoding='utf8'),
              indent=4, default=str)

    # Pickle results
    save_results = {
        'fields':   [f.tolist() for f in fields],
        'x':        x.tolist() if hasattr(x, 'tolist') else list(x),
        'times':    results['times'],
        'energies': results['energies'],
        'overlaps': results['overlaps'],
        'mu':       results['mu'].tolist() if hasattr(results['mu'], 'tolist')
                    else list(results['mu']),
        'rho':      results['rho'].tolist() if hasattr(results['rho'], 'tolist')
                    else list(results['rho']),
    }
    pickle.dump(save_results, open(exp_dir / 'data.pkl', 'wb'))
    print(f"  Data saved to {exp_dir}")
    return exp_dir


def save_to_excel(fields, results, x, config, circuit_meta, exp_dir):
    """Save all simulation data to Excel with required sheets."""
    path = exp_dir / 'results.xlsx'
    wb   = openpyxl.Workbook()

    # ── Sheet 1: Configuration ────────────────────────────────────────────
    ws = wb.active
    ws.title = "Configuration"
    ws.append(["Parameter", "Value"])
    for k, v in config.items():
        ws.append([str(k), str(v)])

    # ── Sheet 2: Medium ───────────────────────────────────────────────────
    ws_med = wb.create_sheet("Medium")
    ws_med.append(["x_index", "x [m]", "rho [kg/m3]", "mu [Pa]"])
    rho_arr = results['rho']
    mu_arr  = results['mu']
    for i in range(len(rho_arr)):
        ws_med.append([
            i,
            float(i * config.get('dx', 1.0)),
            float(rho_arr[i]),
            float(mu_arr[min(i, len(mu_arr) - 1)]),
        ])

    # ── Sheet 3: TimeSeries ───────────────────────────────────────────────
    ws_ts = wb.create_sheet("TimeSeries")
    ws_ts.append(["time_step", "time [s]", "classical_PDE_energy"])
    for i, E in enumerate(results['energies']):
        ws_ts.append([i + 1, float(results['times'][i + 1]), float(E)])

    # ── Sheet 4: Overlaps ─────────────────────────────────────────────────
    ws_ov = wb.create_sheet("Overlaps")
    ws_ov.append(["time [s]", "squared_overlap"])
    for t, ov in results.get('overlaps', []):
        ws_ov.append([float(t), float(ov)])

    # ── Sheet 5: WaveFields ───────────────────────────────────────────────
    ws_fld = wb.create_sheet("WaveFields")
    n_pts  = len(fields[0]) if fields else 0
    header = ["time_step", "time [s]"] + [f"u[{i}]" for i in range(n_pts)]
    ws_fld.append(header)
    for t_idx, f in enumerate(fields):
        row = [t_idx, float(results['times'][t_idx])] + [float(v) for v in f]
        ws_fld.append(row)

    # ── Sheet 6: CircuitParams ────────────────────────────────────────────
    ws_circ = wb.create_sheet("CircuitParams")
    ws_circ.append(["Parameter", "Value"])
    for k, v in circuit_meta.items():
        ws_circ.append([str(k), str(v)])

    # ── Sheet 7: Source ───────────────────────────────────────────────────
    if 'source_amplitude' in results:
        ws_src = wb.create_sheet("Source")
        ws_src.append(["x_index", "x [m]", "source_amplitude"])
        src = results['source_amplitude']
        for i in range(len(src)):
            xi = float(x[i]) if i < len(x) else float(i * config.get('dx', 1.0))
            ws_src.append([i, xi, float(src[i])])

    # ── Sheet 8: Loss ─────────────────────────────────────────────────────
    if 'loss' in results:
        ws_loss = wb.create_sheet("Loss")
        ws_loss.append(["time_step", "time [s]", "reconstruction_MSE"])
        times = results['times']
        for i, lv in enumerate(results['loss']):
            t_i = float(times[i + 1]) if i + 1 < len(times) else float(i * config.get('dt', 1.0))
            ws_loss.append([i + 1, t_i, float(lv)])

    # ── Sheet 9: ModelUpdate ──────────────────────────────────────────────
    if 'mu_initial' in results and 'mu_updated' in results:
        ws_mu = wb.create_sheet("ModelUpdate")
        mu_init = results['mu_initial']
        mu_upd  = results['mu_updated']
        mu_true = results.get('mu_true', [None] * len(mu_init))
        ws_mu.append(["index", "mu_initial [Pa]", "mu_updated [Pa]", "mu_true [Pa]"])
        for i in range(len(mu_init)):
            mt = float(mu_true[i]) if i < len(mu_true) and mu_true[i] is not None else ''
            ws_mu.append([i, float(mu_init[i]), float(mu_upd[i]), mt])

    # ── Sheet 10: OptimizationHistory ────────────────────────────────────
    opt = results.get('optimization_history', {})
    if opt:
        ws_opt = wb.create_sheet("OptimizationHistory")
        loss_hist = opt.get('loss_history', [])
        iter_hist = opt.get('iteration_history', [])
        mu_hist   = opt.get('mu_history', [])
        ov_hist   = opt.get('overlaps_history', [])

        # Build header: iteration, loss, mean_overlap, then mu_0..mu_n
        n_mu = len(mu_hist[0]) if mu_hist else 0
        ws_opt.append(
            ["iteration", "loss", "mean_overlap"] +
            [f"mu[{i}]" for i in range(n_mu)]
        )

        for idx in range(len(iter_hist)):
            it   = iter_hist[idx] if idx < len(iter_hist) else idx
            loss = float(loss_hist[idx]) if idx < len(loss_hist) else ''
            ovs  = ov_hist[idx] if idx < len(ov_hist) else []
            mean_ov = float(np.mean([o for _, o in ovs])) if ovs else ''
            mu_row  = [float(v) for v in mu_hist[idx]] if idx < len(mu_hist) else []
            ws_opt.append([it, loss, mean_ov] + mu_row)

    wb.save(path)
    print(f"  Excel saved to {path}")
    return path
