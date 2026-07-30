"""
run_lambda_sweep.py — Eksperimen variasi lambda (REG_WEIGHT) untuk Tikhonov
regularization, menghasilkan folder per-lambda dan ringkasan L-curve.

Cara pakai (default: ricker_wavelet_source, 6 nilai lambda):
    python run_lambda_sweep.py

Custom sweep / nilai lambda:
    REG_WEIGHT_VALUES="0,1e-27,1e-25,1e-23,1e-21,1e-19" \\
    LAMBDA_SWEEP_SOURCE="gaussian_source" \\
    python run_lambda_sweep.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

from main import SWEEPS, run_single_sweep
from src.constants import configure_plot_style

# ── Konfigurasi (dapat diubah lewat env var) ──────────────────────────────
SWEEP_NAME = os.getenv("LAMBDA_SWEEP_SOURCE", "ricker_wavelet_source")
LAMBDA_VALUES = [
    float(v)
    for v in os.getenv(
        "REG_WEIGHT_VALUES",
        "0,1e-27,1e-25,1e-23,1e-21,1e-19",
    ).split(",")
]
OUTPUT_ROOT = Path("runs/reg_weight_sweep")


def format_lambda(lam: float) -> str:
    """Format lambda untuk nama folder (aman di semua OS)."""
    if lam == 0.0:
        return "0e+00"
    s = f"{lam:.0e}"
    return s


def main():
    configure_plot_style()

    # Cari sweep config by name
    try:
        sweep = next(s for s in SWEEPS if s["name"] == SWEEP_NAME)
    except StopIteration:
        print(f"Error: sweep '{SWEEP_NAME}' tidak ditemukan di SWEEPS.")
        print(f"  Sweep yang tersedia: {[s['name'] for s in SWEEPS]}")
        sys.exit(1)

    rows = []
    for lam in LAMBDA_VALUES:
        tag = format_lambda(lam)
        base_dir = OUTPUT_ROOT / f"lambda_{tag}"
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 72}")
        print(f"### Menjalankan {SWEEP_NAME} dengan lambda={lam:g} -> {base_dir}")
        print(f"{'#' * 72}")
        _, _, summary = run_single_sweep(sweep, base_dir, reg_weight_override=lam)
        rows.append(summary)

    # ── Ringkasan L-curve ──────────────────────────────────────────────────
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Urutkan berdasarkan lambda
    rows.sort(key=lambda r: r["reg_weight"])

    # --- Simpan tabel ke Excel ---
    xlsx_path = summary_dir / "l_curve_summary.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "L-Curve Summary"
    headers = [
        "reg_weight",
        "regularization_norm",
        "data_misfit",
        "best_loss",
        "final_loss",
        "relative_model_error",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([
            r["reg_weight"],
            r["regularization_norm"],
            r["data_misfit"],
            r["best_loss"],
            r["final_loss"],
            r["relative_model_error"],
        ])
    wb.save(str(xlsx_path))
    print(f"\n  Saved {xlsx_path}")

    # --- Plot L-curve ---
    lam_vals = np.array([r["reg_weight"] for r in rows])
    x_vals = np.array([r["regularization_norm"] for r in rows])
    y_vals = np.array([r["data_misfit"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Log-log plot; hindari log(0) untuk lambda=0 atau norm=0
    mask = (x_vals > 0) & (y_vals > 0)
    if np.any(mask):
        ax.loglog(
            x_vals[mask], y_vals[mask],
            "o-", color="steelblue", linewidth=1.8, markersize=8,
        )

    # Label tiap titik dengan lambda-nya
    for i in range(len(rows)):
        lam_i = lam_vals[i]
        xi = x_vals[i]
        yi = y_vals[i]
        label = format_lambda(lam_i) if lam_i > 0 else "0"
        if xi > 0 and yi > 0:
            ax.annotate(
                label, (xi, yi),
                textcoords="offset points", xytext=(8, 6),
                ha="left", fontsize=8,
            )
        elif xi <= 1e-30 and yi > 0:
            ax.annotate(
                label, (x_vals[mask][0], yi),
                textcoords="offset points", xytext=(-30, -12),
                ha="left", fontsize=8, color="gray",
            )

    ax.set_xlabel(r"Regularization Norm $||\mu - \mu_{\rm prior}||^2$")
    ax.set_ylabel("Data Misfit")
    ax.set_title(f"L-Curve — {SWEEP_NAME}\n(Tikhonov regularization weight sweep)")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    lc_path = summary_dir / "l_curve.png"
    fig.savefig(str(lc_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {lc_path}")

    # --- Tampilkan ringkasan ke konsol ---
    print(f"\n{'=' * 72}")
    print(f"  L-Curve Summary — {SWEEP_NAME}")
    print(f"{'=' * 72}")
    print(f"  {'lambda':>12}  {'reg_norm':>14}  {'data_misfit':>14}  {'best_loss':>14}  {'rel_err':>10}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*10}")
    for r in rows:
        lam_str = format_lambda(r["reg_weight"])
        print(
            f"  {lam_str:>12}  {r['regularization_norm']:>14.6e}  "
            f"{r['data_misfit']:>14.6e}  {r['best_loss']:>14.6e}  "
            f"{r['relative_model_error']:>10.3%}"
        )
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
