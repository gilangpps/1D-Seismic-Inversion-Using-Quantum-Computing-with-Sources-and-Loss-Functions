import numpy as np


def evolve_1d_wave(u0, u1, dx, dt, mu, rho, source_func=None, steps=200,
                   bc='dirichlet'):
    nx = len(u0)
    if u1 is None:
        u1 = u0.copy()
    u_prev = u0.copy()
    u_curr = u1.copy()
    out = [u_prev.copy(), u_curr.copy()]

    mu_half = np.zeros(nx - 1)
    for i in range(nx - 1):
        mu_half[i] = 0.5 * (mu[min(i, len(mu) - 1)] + mu[min(i + 1, len(mu) - 1)])

    for step in range(1, steps):
        u_next = np.zeros_like(u_curr)
        for i in range(1, nx - 1):
            stress = (mu_half[i] * (u_curr[i + 1] - u_curr[i])
                      - mu_half[i - 1] * (u_curr[i] - u_curr[i - 1]))
            u_next[i] = (2 * u_curr[i] - u_prev[i]
                         + (dt ** 2 / (rho[i] * dx ** 2)) * stress)
        if bc == 'dirichlet':
            u_next[0] = 0.0
            u_next[-1] = 0.0
        elif bc == 'neumann':
            u_next[0] = u_next[1]
            u_next[-1] = u_next[-2]
        if source_func is not None:
            t = step * dt
            for i in range(nx):
                u_next[i] += dt * dt * source_func(i, t)
        out.append(u_next.copy())
        u_prev, u_curr = u_curr, u_next
    return out


def gaussian_source(center_idx, width_idx, amplitude, nx):
    def S(i, t):
        spatial = np.exp(-((i - center_idx) ** 2) / (2 * width_idx ** 2))
        temporal = amplitude * np.exp(-(t - 0.5) ** 2 / (2 * 0.05 ** 2))
        return spatial * temporal
    return S


def ricker_wavelet_source(center_idx, width_idx, amplitude, nx):
    def S(i, t):
        spatial = np.exp(-((i - center_idx) ** 2) / (2 * width_idx ** 2))
        tau = t - 0.5
        sigma = 0.05
        wavelet = amplitude * (1 - 2 * (tau / sigma) ** 2) * np.exp(-(tau / sigma) ** 2)
        return spatial * wavelet
    return S


def compute_energy(field_prev, field_curr, dx, dt, mu, rho):
    vel = (field_curr - field_prev) / dt
    grad = np.gradient(field_curr, dx)
    energy_density = 0.5 * (rho * vel ** 2 + mu[:len(grad)] * grad ** 2)
    return np.sum(energy_density) * dx
