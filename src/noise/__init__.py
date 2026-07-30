"""
src/noise/__init__.py
──────────────────────────────────────────────────────────────────────────────
Additive noise for synthetic "observed" forward data.

Real seismic recordings are never perfectly clean — they carry ambient,
instrument, and coupling noise. To test how robust the inversion is under
realistic conditions (rather than only the idealized noise-free case), this
module lets us inject zero-mean Gaussian noise into the forward-simulated
wavefield that stands in for the "observed" data, exactly as is common
practice in FWI robustness studies (compare: noiseless vs noisy inversion
runs in the tomography literature).

Only the interior grid points are perturbed — boundary (ghost) nodes are
left untouched so Dirichlet/Neumann boundary conditions stay exact.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np


def add_gaussian_noise_to_fields(fields, noise_level=0.05, seed=42,
                                  relative_to='peak'):
    """
    Add zero-mean Gaussian noise to a list of wavefield snapshots.

    Parameters
    ----------
    fields : list of np.ndarray, shape (nx+2,) each
        Clean forward-simulated wavefield snapshots (with boundary padding
        at index 0 and -1, interior physical points at [1:-1]).
    noise_level : float
        Noise standard deviation as a fraction of the reference amplitude
        of the whole dataset (default 0.05 = 5%).
    seed : int
        RNG seed, for reproducibility across runs.
    relative_to : {'peak', 'rms'}
        Whether `noise_level` scales the peak |amplitude| ('peak') or the
        RMS amplitude ('rms') of the full (all times, all interior points)
        dataset. 'peak' is the more common convention for reporting a
        noise-to-signal percentage.

    Returns
    -------
    list of np.ndarray
        Noisy copies of `fields`. Boundary nodes are copied unchanged.
    """
    rng = np.random.default_rng(seed)
    stacked = np.array([np.asarray(f)[1:-1] for f in fields])

    if relative_to == 'rms':
        ref_amp = float(np.sqrt(np.mean(stacked ** 2)))
    else:
        ref_amp = float(np.max(np.abs(stacked)))
    if ref_amp <= 0:
        ref_amp = 1.0

    sigma = noise_level * ref_amp

    noisy_fields = []
    for f in fields:
        f = np.asarray(f)
        f_noisy = f.copy()
        noise = rng.normal(0.0, sigma, size=f.shape[0] - 2)
        f_noisy[1:-1] = f[1:-1] + noise
        noisy_fields.append(f_noisy)
    return noisy_fields


def noise_summary(clean_fields, noisy_fields):
    """
    Quick diagnostic: achieved SNR (dB) and RMS noise amplitude between a
    clean and a noisy field list. Useful for logging/reporting what noise
    level was actually injected.
    """
    clean = np.array([np.asarray(f)[1:-1] for f in clean_fields])
    noisy = np.array([np.asarray(f)[1:-1] for f in noisy_fields])
    noise = noisy - clean

    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))
    snr_db = (
        10.0 * np.log10(signal_power / noise_power)
        if noise_power > 0 else float('inf')
    )
    return {
        'rms_noise': float(np.sqrt(noise_power)),
        'rms_signal': float(np.sqrt(signal_power)),
        'snr_db': float(snr_db),
    }
