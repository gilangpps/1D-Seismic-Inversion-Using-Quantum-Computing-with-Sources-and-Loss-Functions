import numpy as np


def build_hamiltonian(mu, rho, dx, nx):
    K = np.zeros((nx, nx))
    for i in range(nx):
        mu_r = 0.5 * (mu[min(i, len(mu) - 1)] + mu[min(i + 1, len(mu) - 1)])
        mu_l = 0.5 * (mu[max(i - 1, 0)] + mu[min(i, len(mu) - 1)])
        K[i, i] = -(mu_r + mu_l) / (rho[i] * dx ** 2)
        if i + 1 < nx:
            K[i, i + 1] = mu_r / (rho[i] * dx ** 2)
        if i - 1 >= 0:
            K[i, i - 1] = mu_l / (rho[i] * dx ** 2)

    A = np.zeros((2 * nx, 2 * nx))
    A[:nx, nx:] = np.eye(nx)
    A[nx:, :nx] = K

    n_qubits = int(np.ceil(np.log2(2 * nx)))
    dim = 2 ** n_qubits
    A_pad = np.zeros((dim, dim))
    A_pad[:2 * nx, :2 * nx] = A

    A_antisym = (A_pad - A_pad.T) / 2.0
    H = 1j * A_antisym
    return H, n_qubits, dim
