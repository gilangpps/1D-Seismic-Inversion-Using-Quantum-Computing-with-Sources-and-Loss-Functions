import matplotlib
import pathlib

ENUMS = ['a.)', 'b.)', 'c.)', 'd.)', 'e.)', 'f.)']
FIGURES_DIR = pathlib.Path('figures')
DATA_DIR = pathlib.Path('data')


def configure_plot_style():
    try:
        matplotlib.rcParams['font.family'] = 'Times New Roman'
    except Exception:
        matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['axes.titlesize'] = 13
    matplotlib.rcParams['legend.fontsize'] = 9
