"""
src/wave/__init__.py
──────────────────────────────────────────────────────────────────────────────
1-D Elastic Wave Solver — Leapfrog Finite-Difference Scheme
Reference: Schade et al. (2024), arXiv:2312.14747

Equation solved (second-order form):
    ρ(x) ∂²u/∂t² = ∂/∂x[μ(x) ∂u/∂x] + f(x, t)

Discretised as second-order leapfrog:
    u[i,n+1] = 2·u[i,n] − u[i,n−1]
             + (dt²/ρ[i]·dx²) · [μ_{i+½}(u[i+1,n]−u[i,n]) − μ_{i-½}(u[i,n]−u[i-1,n])]
             + (dt²/ρ[i]) · f(i, n·dt)

where
    μ_{i+½} = ½(μ[i] + μ[i+1])   (arithmetic mean at half-point)

SOURCE INJECTION (Requirement B):
    The Ricker / Gaussian source enters the PDE as an additive forcing term
    f(x_i, t) at EVERY interior grid point at EVERY time step.  The temporal
    profile is centred at t0 = t_max / 3 so the source is active during the
    simulation window.  f is added to u_next AFTER the elastic stress update.

BOUNDARY CONDITIONS:
    Dirichlet (DBC): u[0] = u[N+1] = 0  (fixed ends)
    Neumann   (NBC): u[0] = u[1],  u[N+1] = u[N]  (free ends)

ENERGY:
    Classical PDE energy density:
        E(t) = ½ ρ·v² + ½ μ·(∂u/∂x)²
    where v ≈ (u[n] − u[n−1]) / dt  (centred-difference velocity).
    Returns the spatial integral ΣE·dx.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Source functions
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_source(center_idx: int, width_idx: float, amplitude: float,
                    nx_total: int, t0: float = None, sigma_t: float = None):
    """
    Gaussian source wavelet centred at (center_idx, t0).

    Spatial profile:  Gaussian bell around center_idx.
    Temporal profile: Gaussian bell centred at t0 with width sigma_t.

    Parameters
    ----------
    center_idx : int
        Grid index of source centre.
    width_idx : float
        Spatial half-width in grid indices.
    amplitude : float
        Peak amplitude [Pa/s² or displacement/s²].
    nx_total : int
        Total grid length (including boundary ghost nodes).
    t0 : float, optional
        Peak time [s].  Default: set at runtime by set_source_peak().
    sigma_t : float, optional
        Temporal width [s].  Default: set at runtime.

    Returns
    -------
    Callable S(i, t) → float
    """
    _t0     = [t0]      # mutable cell so it can be updated via closure
    _sig    = [sigma_t]

    def S(i: int, t: float) -> float:
        spatial  = np.exp(-((i - center_idx) ** 2) / (2.0 * width_idx ** 2))
        tau      = t - _t0[0]
        temporal = amplitude * np.exp(-(tau ** 2) / (2.0 * _sig[0] ** 2))
        return spatial * temporal

    S._t0  = _t0
    S._sig = _sig
    return S


def ricker_wavelet_source(center_idx: int, width_idx: float, amplitude: float,
                          nx_total: int, t0: float = None, sigma_t: float = None):
    """
    Ricker (Mexican-hat) wavelet source centred at (center_idx, t0).

    The Ricker wavelet is the second derivative of a Gaussian:
        w(t) = A · [1 − 2(τ/σ)²] · exp(−(τ/σ)²)
    where τ = t − t0.

    Parameters
    ----------
    center_idx, width_idx, amplitude, nx_total : see gaussian_source.
    t0 : float, optional
        Peak time [s].  The wavelet goes through its first positive lobe
        before t0 and its negative lobe after.
    sigma_t : float, optional
        Temporal width scale [s].

    Returns
    -------
    Callable S(i, t) → float
    """
    _t0  = [t0]
    _sig = [sigma_t]

    def S(i: int, t: float) -> float:
        spatial  = np.exp(-((i - center_idx) ** 2) / (2.0 * width_idx ** 2))
        tau      = (t - _t0[0]) / _sig[0]
        wavelet  = amplitude * (1.0 - 2.0 * tau ** 2) * np.exp(-(tau ** 2))
        return spatial * wavelet

    S._t0  = _t0
    S._sig = _sig
    return S


def set_source_peak(source_func, t0: float, sigma_t: float):
    """
    Update the temporal parameters of a source function in-place.

    Call this after computing t_max to place the source peak inside the
    simulation window:
        t0       = t_max / 3          (source fires at 1/3 of simulation)
        sigma_t  = t_max / 12         (narrow enough to be well-resolved)

    Parameters
    ----------
    source_func : callable with _t0 and _sig attributes
    t0 : float
    sigma_t : float
    """
    if hasattr(source_func, '_t0'):
        source_func._t0[0]  = t0
        source_func._sig[0] = sigma_t


# ══════════════════════════════════════════════════════════════════════════════
#  Wave solver
# ══════════════════════════════════════════════════════════════════════════════

def evolve_1d_wave(u0, u1, dx: float, dt: float, mu, rho,
                   source_func=None, steps: int = 200,
                   bc: str = 'dirichlet'):
    """
    Leapfrog finite-difference solver for the 1-D elastic wave equation.

        ρ(x) ∂²u/∂t² = ∂/∂x[μ(x) ∂u/∂x] + f(x,t)

    Parameters
    ----------
    u0 : array_like, shape (nx+2,)
        Displacement at t = 0  (includes ghost/boundary nodes).
    u1 : array_like, shape (nx+2,)
        Displacement at t = dt (same as u0 for zero initial velocity).
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].  Must satisfy CFL: dt ≤ dx / v_max.
    mu : array_like, shape ≥ (nx+1,)
        Elastic modulus at each half-point / interface [Pa].
    rho : array_like, shape ≥ (nx,)
        Density at each interior grid point [kg/m³].
    source_func : callable or None
        f(i, t) → forcing value at grid index i and time t [Pa/s² or m/s²].
        If None, pure initial-condition propagation (no source term).
    steps : int
        Number of time steps to advance (output has steps+1 snapshots,
        including the two initial conditions).
    bc : str
        Boundary condition type: 'dirichlet' (default) or 'neumann'.

    Returns
    -------
    out : list of np.ndarray, length = steps + 1
        Wavefield snapshots u[n] at each time n·dt.
        Each snapshot has shape (nx+2,).

    Notes
    -----
    The source term is injected as:
        u_next[i] += (dt² / ρ[i]) · f(i, n·dt)

    This is the correct discretisation of ρ·ü = … + f, giving:
        u_next[i] = 2u[i] − u_prev[i] + (dt²/ρ·dx²)·[stress] + (dt²/ρ)·f
    """
    nx = len(u0)

    u_prev = np.asarray(u0, dtype=float).copy()
    u_curr = np.asarray(u1, dtype=float).copy()
    out = [u_prev.copy(), u_curr.copy()]

    # Half-point elastic moduli  μ_{i+½} = ½(μ[i] + μ[i+1])
    mu_arr  = np.asarray(mu, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)

    mu_half = np.zeros(nx - 1)
    for i in range(nx - 1):
        il = min(i,     len(mu_arr) - 1)
        ir = min(i + 1, len(mu_arr) - 1)
        mu_half[i] = 0.5 * (mu_arr[il] + mu_arr[ir])

    inv_rho = np.zeros(nx)
    for i in range(nx):
        idx = min(i, len(rho_arr) - 1)
        inv_rho[i] = 1.0 / max(rho_arr[idx], 1e-30)

    dt2 = dt * dt
    inv_dx2 = 1.0 / (dx * dx)

    for step in range(1, steps):
        t_curr = step * dt
        u_next = np.zeros_like(u_curr)

        # Interior update: leapfrog + elastic stress divergence
        for i in range(1, nx - 1):
            stress = (mu_half[i]     * (u_curr[i + 1] - u_curr[i])
                    - mu_half[i - 1] * (u_curr[i]     - u_curr[i - 1]))
            u_next[i] = (2.0 * u_curr[i] - u_prev[i]
                         + dt2 * inv_rho[i] * inv_dx2 * stress)

        # Source injection: f(x_i, t) enters PDE as forcing term
        # Requirement B: the source physically drives wave propagation.
        if source_func is not None:
            for i in range(1, nx - 1):
                f_val = source_func(i, t_curr)
                u_next[i] += dt2 * inv_rho[i] * f_val

        # Boundary conditions
        if bc == 'dirichlet':
            u_next[0]  = 0.0
            u_next[-1] = 0.0
        elif bc == 'neumann':
            u_next[0]  = u_next[1]
            u_next[-1] = u_next[-2]

        out.append(u_next.copy())
        u_prev, u_curr = u_curr, u_next

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Energy computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_energy(field_prev, field_curr, dx: float, dt: float, mu, rho):
    """
    Compute classical PDE energy of the elastic wavefield.

    Energy density (discrete):
        e_kin[i] = ½ · ρ[i] · v[i]²        kinetic energy
        e_pot[i] = ½ · μ[i] · (∂u/∂x)[i]²  potential energy
        E        = Σᵢ (e_kin[i] + e_pot[i]) · dx

    Velocity approximation (backward-difference):
        v[i] ≈ (u[n][i] − u[n−1][i]) / dt

    Strain approximation (central difference, evaluated on interior):
        ∂u/∂x[i] ≈ (u[n][i+1] − u[n][i-1]) / (2·dx)

    Parameters
    ----------
    field_prev : array_like
        Wavefield at t − dt.
    field_curr : array_like
        Wavefield at t.
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].
    mu : array_like
        Elastic modulus.  Length must be ≥ len(field_curr).
    rho : array_like
        Density.  Length must be ≥ len(field_curr).

    Returns
    -------
    float
        Total classical PDE energy [J/m].
    """
    fp = np.asarray(field_prev, dtype=float)
    fc = np.asarray(field_curr, dtype=float)
    n  = len(fc)

    mu_arr  = np.asarray(mu,  dtype=float)
    rho_arr = np.asarray(rho, dtype=float)

    # Velocity (backward-difference)
    vel = (fc - fp) / dt

    # Strain (central difference, interior only; edges use forward/backward)
    strain = np.gradient(fc, dx)

    e_kin = np.zeros(n)
    e_pot = np.zeros(n)

    for i in range(n):
        ri = min(i, len(rho_arr) - 1)
        mi = min(i, len(mu_arr)  - 1)
        e_kin[i] = 0.5 * rho_arr[ri] * vel[i]    ** 2
        e_pot[i] = 0.5 * mu_arr[mi]  * strain[i] ** 2

    return float(np.sum(e_kin + e_pot) * dx)
