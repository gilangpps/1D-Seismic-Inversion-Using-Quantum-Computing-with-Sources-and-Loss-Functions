import numpy as np

from src.distributions import homogeneous, spike
from src.encoding import amplitude_encode
from src.wave import evolve_1d_wave, gaussian_source, compute_energy


def run_experiment_1d(
    nx=7, dx=1.0, dt=0.0001, steps=19,
    mu_arr=None, rho_arr=None,
    u0=None, v0=None,
    source_params=None, measure_every=5, shots=1000,
    bc='dirichlet',
    reference_fields=None,
):
    """
    Run a 1-D elastic wave experiment.

    Computes the classical wavefield, energy, and quantum-state overlap.

    Bug 6 (FIXED):
    ──────────────
    The original overlap computation was:

        ref_sv = amplitude_encode(fields[0])   ← encodes t=0 INITIAL state

        for t_idx in range(1, len(fields)):
            tgt_sv = amplitude_encode(fields[t_idx])
            ov = |⟨ref_sv | tgt_sv⟩|²          ← overlap w/ INITIAL state

    This measures how much the wavefield has DECAYED from its initial
    condition — it always decreases monotonically as the wave disperses
    from the source point.  It cannot increase during inversion because
    it does not compare current-model fields to reference (true) fields.
    It is not a useful inversion quality metric.

    CORRECT overlap for inversion:

        |⟨ψ_fwd(t; μ_current) | ψ_ref(t; μ_true)⟩|²

    This measures how close the current-model wavefield is to the
    reference (true-model) wavefield at each timestep.  It starts near
    zero for a bad initial model and approaches 1 as the optimizer
    recovers the true elastic modulus.

    If reference_fields is not provided (e.g. pure forward simulation
    without inversion context), the function falls back to the original
    IC-based overlap with a warning, so backward compatibility is maintained
    for non-optimisation use cases.

    Args:
        nx:               Number of interior grid points.
        dx:               Grid spacing [m].
        dt:               Time step [s].
        steps:            Number of time steps.
        mu_arr:           Elastic modulus (length nx+1).
        rho_arr:          Density (length nx).
        u0:               Initial displacement (length nx).
        v0:               Initial velocity (length nx).
        source_params:    Dict with keys 'center', 'width', 'amplitude'.
                          If None, no source term is applied.
        measure_every:    Measure overlap every this many steps.
        shots:            Quantum measurement shots (unused here, kept for API).
        bc:               Boundary condition ('dirichlet' or 'neumann').
        reference_fields: List of reference wavefield snapshots (length steps+1,
                          each of length nx+2).  If provided, overlaps are
                          computed against these instead of against fields[0].

    Returns:
        fields:  List of wavefield snapshots (length steps+1).
        results: Dict with keys: energies, overlaps, times, mu, rho, field,
                 settings.
        x:       Spatial grid (length nx+2).
    """
    x = np.arange(nx + 2) * dx

    if rho_arr is None:
        rho_arr = homogeneous(2e3, nx)
    if mu_arr is None:
        mu_arr = homogeneous(3e10, nx + 1)
    if u0 is None:
        u0 = spike(1, nx, nx // 2)
    if v0 is None:
        v0 = homogeneous(0, nx)

    # Pad to include ghost (boundary) nodes
    u0_bc = np.zeros(nx + 2)
    u0_bc[1:-1] = u0
    u1_bc = u0_bc.copy()

    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho_arr
    rho_bc[0] = rho_arr[0]
    rho_bc[-1] = rho_arr[-1]
    rho_bc[rho_bc == 0] = rho_arr.mean()

    mu_bc = np.zeros(nx + 2)
    mu_bc[1:min(len(mu_arr) + 1, nx + 2)] = mu_arr[:min(len(mu_arr), nx + 1)]
    mu_bc[0] = mu_arr[0]
    mu_bc[-1] = mu_arr[-1]

    # Source term
    source = None
    if source_params is not None:
        center = source_params.get('center', nx // 2 + 1)
        width  = source_params.get('width', 2)
        amp    = source_params.get('amplitude', 1.0)
        source = gaussian_source(center, width, amp, nx + 2)

    fields = evolve_1d_wave(
        u0_bc, u1_bc, dx=dx, dt=dt,
        mu=mu_bc, rho=rho_bc,
        source_func=source, steps=steps, bc=bc
    )

    times = [i * dt for i in range(len(fields))]

    results = {
        'energies': [],
        'overlaps': [],
        'times': times,
        'mu': mu_arr,
        'rho': rho_arr,
        'field': {'u': np.array([f[1:-1] for f in fields])},
        'settings': {
            'nx': nx, 'dx': dx, 'dt': dt, 'steps': steps,
            'bc': bc, 'shots': shots,
        },
    }

    # ── Energy ────────────────────────────────────────────────────────
    for t_idx in range(1, len(fields)):
        E = compute_energy(
            fields[t_idx - 1], fields[t_idx], dx, dt, mu_bc, rho_bc
        )
        results['energies'].append(E)

    # ── Overlap ───────────────────────────────────────────────────────
    # Bug 6 fix: compare against reference_fields (true model) when available.
    # Fall back to IC-based overlap only if reference_fields is None.
    if reference_fields is not None:
        # Inversion quality metric: |⟨ψ_fwd(t;μ) | ψ_ref(t;μ_true)⟩|²
        # Starts near 0 for bad model, approaches 1 as model is recovered.
        _compute_reference_overlaps(fields, reference_fields,
                                    results, measure_every, dt)
    else:
        # Legacy IC-based overlap (kept for pure forward-sim use):
        # |⟨ψ(0) | ψ(t)⟩|² — measures decay from initial condition.
        # NOT a useful inversion metric.
        _compute_ic_overlaps(fields, results, measure_every, dt)

    return fields, results, x


# ====================================================================== #
#  Overlap helpers                                                         #
# ====================================================================== #

def _compute_reference_overlaps(
    fields: list,
    reference_fields: list,
    results: dict,
    measure_every: int,
    dt: float,
) -> None:
    """
    Compute |⟨ψ_fwd(t;μ) | ψ_ref(t;μ_true)⟩|² for each measured timestep.

    This is the scientifically correct inversion quality metric.
    An overlap of 1 means the current model perfectly reproduces the
    reference wavefield at that timestep.
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


def _compute_ic_overlaps(
    fields: list,
    results: dict,
    measure_every: int,
    dt: float,
) -> None:
    """
    Legacy IC-based overlap: |⟨ψ(0) | ψ(t)⟩|².

    Measures how much the wavefield has spread from its initial condition.
    Useful as a wavefield characterisation metric but NOT as an inversion
    convergence metric.
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
