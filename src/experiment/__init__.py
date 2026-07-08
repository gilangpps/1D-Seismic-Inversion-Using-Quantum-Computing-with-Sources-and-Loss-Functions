"""
src/experiment/__init__.py
──────────────────────────────────────────────────────────────────────────────
Experiment Orchestrator — Full 1-D Elastic Wave Simulation

This module runs the complete forward experiment:
    1. Set up medium (μ, ρ, initial conditions)
    2. Run classical leapfrog PDE solver with source injection (Req. B)
    3. Compute classical PDE energy at each time step (Req. I)
    4. Compute quantum-state overlap with reference fields (Bug 6 fix)
    5. Return fields, results dict, and spatial grid

SOURCE INJECTION:
    If source_params is provided, a Ricker or Gaussian wavelet is constructed
    with temporal peak at t0 = t_max/3 (inside the simulation window) and
    passed to evolve_1d_wave as source_func.  This is Requirement B.

OVERLAP (Bug 6):
    When reference_fields are provided (inversion context), the overlap is:
        |⟨ψ_fwd(t; μ_current) | ψ_ref(t; μ_true)⟩|²
    When reference_fields are None (pure forward simulation), falls back to
    IC-based overlap (diagnostic only, not an inversion metric).
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np

from src.distributions import homogeneous, spike
from src.encoding import amplitude_encode
from src.wave import (evolve_1d_wave, gaussian_source, ricker_wavelet_source,
                      set_source_peak, compute_energy)


def run_experiment_1d(
    nx: int = 7,
    dx: float = 63.0,
    dt: float = 0.005,
    steps: int = 40,
    mu_arr=None,
    rho_arr=None,
    u0=None,
    v0=None,
    source_params=None,
    measure_every: int = 4,
    shots: int = 1000,
    bc: str = 'dirichlet',
    reference_fields=None,
):
    """
    Run a 1-D elastic wave experiment with optional source injection.

    The elastic wave equation solved:
        ρ(x) ∂²u/∂t² = ∂/∂x[μ(x) ∂u/∂x] + f(x,t)

    Parameters
    ----------
    nx : int
        Number of interior grid points.
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].  Must satisfy CFL: dt ≤ dx / v_max.
    steps : int
        Number of time steps.
    mu_arr : array_like, shape (nx+1,)
        Elastic modulus.  If None, homogeneous 3e10 Pa.
    rho_arr : array_like, shape (nx,)
        Density.  If None, homogeneous 2e3 kg/m³.
    u0 : array_like, shape (nx,)
        Initial displacement (interior only).  If None, spike at centre.
    v0 : array_like, shape (nx,)
        Initial velocity.  If None, zeros.
    source_params : dict or None
        Source configuration.  Keys:
            type      : 'ricker' or 'gaussian'  (default: 'ricker')
            center    : source grid index        (default: nx//2 + 1)
            width     : spatial half-width (indices)  (default: 2)
            amplitude : peak amplitude           (default: 1.0)
        If None, no external source term (IC-only propagation).
    measure_every : int
        Compute overlap every this many time steps.
    shots : int
        Quantum measurement shots (for reconstruction diagnostics).
    bc : str
        Boundary condition: 'dirichlet' or 'neumann'.
    reference_fields : list or None
        Reference wavefield snapshots from the true model.
        If provided, overlaps compare current fields vs reference (inversion
        quality metric).  If None, IC-based overlap (diagnostic only).

    Returns
    -------
    fields : list of np.ndarray, length steps+1
        Wavefield snapshots at each time step.  Shape (nx+2,) each.
    results : dict
        energies, overlaps, times, mu, rho, field, settings.
    x : np.ndarray, shape (nx+2,)
        Spatial grid coordinates [m].
    """
    # ── Spatial grid ─────────────────────────────────────────────────────
    x = np.arange(nx + 2) * dx

    # ── Medium defaults ───────────────────────────────────────────────────
    if rho_arr is None:
        rho_arr = homogeneous(2e3, nx)
    if mu_arr is None:
        mu_arr  = homogeneous(3e10, nx + 1)
    if u0 is None:
        u0 = spike(1.0, nx, nx // 2)
    if v0 is None:
        v0 = homogeneous(0.0, nx)

    rho_arr = np.asarray(rho_arr, dtype=float)
    mu_arr  = np.asarray(mu_arr,  dtype=float)
    u0      = np.asarray(u0,      dtype=float)
    v0      = np.asarray(v0,      dtype=float)

    # ── Pad interior arrays to include ghost (boundary) nodes ────────────
    u0_bc = np.zeros(nx + 2)
    u0_bc[1:-1] = u0
    u1_bc = u0_bc.copy()          # zero initial velocity: u(0) = u(-dt)

    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho_arr
    rho_bc[0]    = rho_arr[0]
    rho_bc[-1]   = rho_arr[-1]
    rho_bc[rho_bc == 0] = float(np.mean(rho_arr)) if len(rho_arr) > 0 else 2e3

    mu_bc = np.zeros(nx + 2)
    n_mu  = len(mu_arr)
    mu_bc[1:min(n_mu + 1, nx + 2)] = mu_arr[:min(n_mu, nx + 1)]
    mu_bc[0]  = mu_arr[0]
    mu_bc[-1] = mu_arr[-1]

    # ── Source term construction (Requirement B) ──────────────────────────
    # Source peak is placed at t0 = t_max / 3 so it fires inside the window.
    # sigma_t = t_max / 12 gives ~4 periods well-resolved by dt.
    source = None
    if source_params is not None:
        t_max    = steps * dt
        t0       = t_max / 3.0
        sigma_t  = max(t_max / 12.0, 2.0 * dt)

        src_type  = source_params.get('type', 'ricker').lower()
        center    = source_params.get('center', nx // 2 + 1)
        width     = source_params.get('width', 2)
        amplitude = source_params.get('amplitude', 1.0)

        if src_type == 'ricker':
            source = ricker_wavelet_source(center, width, amplitude, nx + 2,
                                           t0=t0, sigma_t=sigma_t)
        else:
            source = gaussian_source(center, width, amplitude, nx + 2,
                                     t0=t0, sigma_t=sigma_t)

        set_source_peak(source, t0=t0, sigma_t=sigma_t)

    # ── Run leapfrog PDE solver ───────────────────────────────────────────
    fields = evolve_1d_wave(
        u0_bc, u1_bc, dx=dx, dt=dt,
        mu=mu_bc, rho=rho_bc,
        source_func=source,
        steps=steps, bc=bc,
    )

    times = [i * dt for i in range(len(fields))]

    # ── Assemble results dict ─────────────────────────────────────────────
    results = {
        'energies': [],
        'overlaps': [],
        'times':    times,
        'mu':       mu_arr,
        'rho':      rho_arr,
        'field':    {'u': np.array([f[1:-1] for f in fields])},
        'settings': {
            'nx': nx, 'dx': dx, 'dt': dt, 'steps': steps,
            'bc': bc, 'shots': shots,
        },
    }

    # ── Classical PDE energy (Requirement I) ─────────────────────────────
    # E_classical(t) = ½Σᵢ[ρᵢvᵢ² + μᵢ(∂u/∂x)ᵢ²]·dx
    for t_idx in range(1, len(fields)):
        E = compute_energy(
            fields[t_idx - 1], fields[t_idx],
            dx, dt, mu_bc, rho_bc,
        )
        results['energies'].append(float(E))

    # ── Quantum-state overlap (Bug 6 fix) ─────────────────────────────────
    if reference_fields is not None:
        _compute_reference_overlaps(fields, reference_fields,
                                    results, measure_every, dt)
    else:
        _compute_ic_overlaps(fields, results, measure_every, dt)

    return fields, results, x


# ══════════════════════════════════════════════════════════════════════════════
#  Overlap helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_reference_overlaps(fields, reference_fields, results,
                                 measure_every, dt):
    """
    Inversion quality metric:
        |⟨ψ_fwd(t; μ_current) | ψ_ref(t; μ_true)⟩|²

    Starts near 0 for a bad initial model, approaches 1 as μ_current → μ_true.
    """
    n_steps = min(len(fields), len(reference_fields))
    for t_idx in range(1, n_steps):
        if t_idx % measure_every != 0:
            continue
        sv_fwd, _, _ = amplitude_encode(fields[t_idx])
        sv_ref, _, _ = amplitude_encode(reference_fields[t_idx])
        if sv_fwd is None or sv_ref is None:
            results['overlaps'].append((t_idx * dt, 0.0))
            continue
        L = max(len(sv_fwd), len(sv_ref))
        a = np.zeros(L, dtype=complex)
        b = np.zeros(L, dtype=complex)
        a[:len(sv_fwd)] = sv_fwd
        b[:len(sv_ref)] = sv_ref
        ov = float(abs(np.vdot(a, b)) ** 2)
        results['overlaps'].append((t_idx * dt, ov))


def _compute_ic_overlaps(fields, results, measure_every, dt):
    """
    Legacy IC-based overlap: |⟨ψ(0) | ψ(t)⟩|²

    Measures wavefield dispersion from initial condition.
    Not useful as an inversion convergence metric.
    """
    ref_sv, _, _ = amplitude_encode(fields[0])
    if ref_sv is None:
        return
    for t_idx in range(1, len(fields)):
        if t_idx % measure_every != 0:
            continue
        tgt_sv, _, _ = amplitude_encode(fields[t_idx])
        if tgt_sv is None:
            results['overlaps'].append((t_idx * dt, 0.0))
            continue
        L = max(len(ref_sv), len(tgt_sv))
        a = np.zeros(L, dtype=complex)
        b = np.zeros(L, dtype=complex)
        a[:len(ref_sv)] = ref_sv
        b[:len(tgt_sv)] = tgt_sv
        ov = float(abs(np.vdot(a, b)) ** 2)
        results['overlaps'].append((t_idx * dt, ov))
