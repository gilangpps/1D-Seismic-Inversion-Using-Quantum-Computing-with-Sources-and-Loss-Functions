import datetime
import json
import pickle

import numpy as np
import openpyxl

from src.constants import DATA_DIR


def save_experiment(fields, results, x, config, data_dir=DATA_DIR):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_dir = data_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    json.dump(config, open(exp_dir / 'configs.json', 'w', encoding='utf8'), indent=4)

    save_results = {
        'fields': np.array(fields),
        'x': x,
        'times': results['times'],
        'energies': results['energies'],
        'overlaps': results['overlaps'],
        'mu': results['mu'].tolist(),
        'rho': results['rho'].tolist(),
    }
    pickle.dump(save_results, open(exp_dir / 'data.pkl', 'wb'))
    print(f"  Data saved to {exp_dir}")
    return exp_dir


def save_to_excel(fields, results, x, config, circuit_meta, exp_dir):
    """Save all simulation data to an Excel workbook with multiple sheets."""
    path = exp_dir / 'results.xlsx'
    wb = openpyxl.Workbook()

    ws_cfg = wb.active
    ws_cfg.title = "Configuration"
    ws_cfg.append(["Parameter", "Value"])
    for key, val in config.items():
        ws_cfg.append([str(key), str(val)])

    ws_med = wb.create_sheet("Medium")
    ws_med.append(["x_index", "x [m]", "rho [kg/m3]", "mu [Pa]"])
    rho_arr = results['rho']
    mu_arr = results['mu']
    for i in range(len(rho_arr)):
        ws_med.append([i, float(i * config['dx']),
                       float(rho_arr[i]),
                       float(mu_arr[min(i, len(mu_arr) - 1)])])

    ws_ts = wb.create_sheet("TimeSeries")
    ws_ts.append(["time_step", "time [s]", "energy"])
    for i, E in enumerate(results['energies']):
        ws_ts.append([i + 1, float(results['times'][i + 1]), float(E)])

    ws_ov = wb.create_sheet("Overlaps")
    ws_ov.append(["time [s]", "squared_overlap"])
    for t, ov in results['overlaps']:
        ws_ov.append([float(t), float(ov)])

    ws_fld = wb.create_sheet("WaveFields")
    header = ["time_step", "time [s]"] + [f"u[{i}]" for i in range(len(fields[0]))]
    ws_fld.append(header)
    for t_idx, f in enumerate(fields):
        row = [t_idx, float(results['times'][t_idx])] + [float(v) for v in f]
        ws_fld.append(row)

    ws_circ = wb.create_sheet("CircuitParams")
    ws_circ.append(["Parameter", "Value"])
    for key, val in circuit_meta.items():
        ws_circ.append([str(key), str(val)])

    if 'source_amplitude' in results:
        ws_src = wb.create_sheet("Source")
        ws_src.append(["x_index", "x", "source_amplitude"])
        src_amp = results['source_amplitude']
        for i in range(len(src_amp)):
            ws_src.append([i, float(x[i]), float(src_amp[i])])

    if 'loss' in results:
        ws_loss = wb.create_sheet("Loss")
        ws_loss.append(["time_step", "time", "loss"])
        times = results['times']
        for i, lv in enumerate(results['loss']):
            ws_loss.append([i + 1, float(times[i + 1]), float(lv)])

    if 'mu_initial' in results and 'mu_updated' in results:
        ws_mu = wb.create_sheet("ModelUpdate")
        ws_mu.append(["index", "mu_initial", "mu_updated"])
        mu_init = results['mu_initial']
        mu_upd = results['mu_updated']
        for i in range(len(mu_init)):
            ws_mu.append([i, float(mu_init[i]), float(mu_upd[i])])

    wb.save(path)
    print(f"  Excel saved to {path}")
    return path
