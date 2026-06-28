import numpy as np

from src.distributions import homogeneous, spike
from src.encoding import amplitude_encode
from src.wave import evolve_1d_wave, gaussian_source, compute_energy


def run_experiment_1d(nx=7, dx=1.0, dt=0.0001, steps=19,
                      mu_arr=None, rho_arr=None,
                      u0=None, v0=None,
                      source_params=None, measure_every=5, shots=1000,
                      bc='dirichlet'):
    x = np.arange(nx + 2) * dx

    if rho_arr is None:
        rho_arr = homogeneous(2e3, nx)
    if mu_arr is None:
        mu_arr = homogeneous(3e10, nx + 1)
    if u0 is None:
        u0 = spike(1, nx, nx // 2)
    if v0 is None:
        v0 = homogeneous(0, nx)

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

    source = None
    if source_params is not None:
        center = source_params.get('center', nx // 2 + 1)
        width = source_params.get('width', 2)
        amp = source_params.get('amplitude', 1.0)
        source = gaussian_source(center, width, amp, nx + 2)

    fields = evolve_1d_wave(u0_bc, u1_bc, dx=dx, dt=dt,
                            mu=mu_bc, rho=rho_bc,
                            source_func=source, steps=steps, bc=bc)

    times = [i * dt for i in range(len(fields))]
    ref_sv, _, _ = amplitude_encode(fields[0])

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

    for t_idx in range(1, len(fields)):
        E = compute_energy(fields[t_idx - 1], fields[t_idx], dx, dt,
                           mu_bc, rho_bc)
        results['energies'].append(E)

        if t_idx % measure_every == 0 and ref_sv is not None:
            tgt_sv, _, _ = amplitude_encode(fields[t_idx])
            if tgt_sv is None:
                ov = 0.0
            else:
                L = max(len(ref_sv), len(tgt_sv))
                ref_use = np.zeros(L, dtype=complex)
                ref_use[:len(ref_sv)] = ref_sv
                tgt_use = np.zeros(L, dtype=complex)
                tgt_use[:len(tgt_sv)] = tgt_sv
                ov = abs(np.vdot(ref_use, tgt_use)) ** 2
            results['overlaps'].append((t_idx * dt, ov))

    return fields, results, x
