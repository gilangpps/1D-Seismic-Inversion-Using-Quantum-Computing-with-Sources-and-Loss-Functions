"""
src/constants/__init__.py
Global constants, directory paths, and plot style configuration.
"""

import matplotlib
import pathlib

# Panel label enumerators for multi-panel publication figures
ENUMS = ['a.)', 'b.)', 'c.)', 'd.)', 'e.)', 'f.)']

# Output directories
FIGURES_DIR = pathlib.Path('figures')
DATA_DIR    = pathlib.Path('data')


def configure_plot_style():
    """
    Apply publication-quality matplotlib style settings.
    Uses DejaVu Serif if available, falls back to Liberation Serif.
    """
    try:
        matplotlib.rcParams['font.family'] = 'DejaVu Serif'
    except Exception:
        try:
            matplotlib.rcParams['font.family'] = 'Liberation Serif'
        except Exception:
            pass  # use default

    matplotlib.rcParams['font.size']        = 11
    matplotlib.rcParams['axes.labelsize']   = 12
    matplotlib.rcParams['axes.titlesize']   = 13
    matplotlib.rcParams['legend.fontsize']  = 9
    matplotlib.rcParams['figure.dpi']       = 100
    matplotlib.rcParams['savefig.dpi']      = 300
    matplotlib.rcParams['axes.grid']        = False
    matplotlib.rcParams['lines.linewidth']  = 1.5
