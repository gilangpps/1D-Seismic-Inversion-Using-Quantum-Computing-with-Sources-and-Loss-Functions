"""
Test: verify sqrt-symmetrization removes H null-space degeneracy.

The sqrt-symmetrized Hamiltonian H = [[0,i·S_op],[-i·S_op,0]] with
S_op = sqrtm(-K_sym) has NO null space from the identity-block structure
that plagued the dilation H = [[0,A],[A†,0]] where AA† = I.

ALL initial conditions should produce non-zero ||(H1-H2)@psi||.
This file verifies the structural fix.
"""

import numpy as np
from src.distributions import raised_cosine, homogeneous
from src.hamiltonian import build_hamiltonian, check_ic_breaks_degeneracy


def _build_ic(mode="asymmetric_spike", nx=7, dx=63.0, seed=42):
    """Build IC of the specified type."""
    rng = np.random.default_rng(seed=seed)
    if mode == "asymmetric_spike":
        spike_center = int(0.3 * nx)
        u0 = np.zeros(nx)
        u0[spike_center] = 1.0
        u0[min(spike_center + 1, nx - 1)] = 0.4
        v0 = 0.15 * np.sqrt(2.5e10 / 3e3) * rng.normal(0, 1, nx)
    elif mode == "gaussian":
        x = np.arange(nx) * dx
        x_center = x[nx // 2]
        sigma_x = 2.0 * dx
        u0 = np.exp(-0.5 * ((x - x_center) / sigma_x) ** 2)
        v0 = np.zeros(nx)
    elif mode == "multi_mode_sine":
        x_norm = np.arange(nx) / (nx - 1 + 1e-30)
        u0 = (0.5 * np.sin(2 * np.pi * x_norm) +
              0.3 * np.sin(4 * np.pi * x_norm) +
              0.2 * np.sin(6 * np.pi * x_norm))
        u0 = u0 / (np.max(np.abs(u0)) + 1e-30)
        v0 = 0.1 * np.sqrt(2.5e10 / 3e3) * np.cos(2 * np.pi * x_norm)
    else:
        raise ValueError(f"Unknown IC mode: {mode}")
    return u0, v0


def _bc_pad(arr, nx):
    """BC-pad arrays to length nx+2 (same as quantum_forward_simulate).

    Handles both:
    - rho arrays (length nx) → rho_bc[1:-1] = arr
    - mu arrays  (length nx+1) → mu_bc[1:nx+2] = arr[:nx+1]
    """
    padded = np.zeros(nx + 2)
    n = len(arr)
    if n == nx:
        padded[1:-1] = arr
    elif n == nx + 1:
        padded[1:nx + 2] = arr
    else:
        padded[1:-1] = arr[:nx]
    padded[0] = arr[0]
    padded[-1] = arr[-1]
    return padded


def _make_probe_arrays(nx, mu_low, mu_high):
    """Create BC-padded mu arrays for testing."""
    rng = np.random.default_rng(seed=42)
    mu_low_arr = homogeneous(mu_low, nx + 1)
    mu_high_arr = homogeneous(mu_high, nx + 1)
    rho_arr = raised_cosine(2e3, nx, nx - 1, 6, 2e3)
    return _bc_pad(mu_low_arr, nx), _bc_pad(mu_high_arr, nx), _bc_pad(rho_arr, nx)


def _encode_state(u0, v0, rho_bc, S_op, nx):
    """Encode [u,v] into [q;p] state for sqrt-symmetrized H."""
    sqrt_rho = np.sqrt(np.abs(rho_bc[1:-1]))
    w = sqrt_rho * u0
    q = S_op @ w
    p = sqrt_rho * v0
    return np.concatenate([q, p])


def _compute_diff_norm_encoded(u0, v0, mu_bc_low, mu_bc_high, rho_bc, dx, nx):
    """Compute ||(H1 - H2) @ psi_encoded|| with correct sqrt-symmetrization encoding."""
    H1, _, dim, phys_dim, S_op1 = build_hamiltonian(mu_bc_low, rho_bc, dx, nx)
    H2, _, _, _, S_op2 = build_hamiltonian(mu_bc_high, rho_bc, dx, nx)
    psi_enc = np.zeros(dim, dtype=complex)
    psi_enc[:phys_dim] = _encode_state(u0, v0, rho_bc, S_op1, nx)
    return float(np.linalg.norm((H1 - H2) @ psi_enc))


def _test_ic_breaks_degeneracy(mode, nx=7, dx=63.0, mu_low=0.5e10, mu_high=5.0e10):
    """Verify a given IC produces non-zero ||(H1-H2)@psi|| with sqrt-symmetrization."""
    u0, v0 = _build_ic(mode, nx=nx, dx=dx)
    mu_bc_low, mu_bc_high, rho_bc = _make_probe_arrays(nx, mu_low, mu_high)
    diff_norm = _compute_diff_norm_encoded(u0, v0, mu_bc_low, mu_bc_high, rho_bc, dx, nx)
    print(f"\n  [{mode}] ||(H1-H2)@psi|| = {diff_norm:.6e}")
    return diff_norm


def test_asymmetric_spike_breaks_degeneracy():
    """Asymmetric spike IC produces non-zero ||(H1-H2)@psi|| with sqrt-symmetrization."""
    diff_norm = _test_ic_breaks_degeneracy("asymmetric_spike")
    assert diff_norm > 1e-3, (
        f"Asymmetric spike: ||diff||={diff_norm:.3e} < 1e-3. "
        f"Expected non-zero with sqrt-symmetrized H."
    )


def test_gaussian_ic_breaks_degeneracy():
    """Gaussian IC (v0=0) also breaks degeneracy with sqrt-symmetrization.

    With the old dilation H (AA†=I), Gaussian IC with v0=0 fell in null space.
    With sqrt-symmetrized H (S_op=√(-K_sym)), ALL ICs produce mu-dependence.
    """
    diff_norm = _test_ic_breaks_degeneracy("gaussian")
    assert diff_norm > 1e-3, (
        f"Gaussian IC: ||diff||={diff_norm:.3e} < 1e-3. "
        f"Expected non-zero with sqrt-symmetrized H (no null space)."
    )


def test_multi_mode_sine_with_v0_breaks_degeneracy():
    """Multi-mode sine IC with v0 produces non-zero ||(H1-H2)@psi||."""
    diff_norm = _test_ic_breaks_degeneracy("multi_mode_sine")
    assert diff_norm > 1e-3, (
        f"Multi-mode sine+v0: ||diff||={diff_norm:.3e} < 1e-3. "
    )


def test_multi_mode_sine_v0_zero_breaks_degeneracy():
    """Multi-mode sine IC with v0=0 also breaks degeneracy with sqrt-symmetrization.

    With old dilation H, v0=0 caused null space (u eigenmodes of K).
    With sqrt-symmetrized H, the S_op block is mu-dependent for all states.
    """
    u0, _ = _build_ic("multi_mode_sine")
    v0 = np.zeros(7)
    mu_bc_low, mu_bc_high, rho_bc = _make_probe_arrays(7, 0.5e10, 5.0e10)
    diff_norm = _compute_diff_norm_encoded(u0, v0, mu_bc_low, mu_bc_high, rho_bc, 63.0, 7)
    print(f"\n  [multi_mode_sine+v0=0] ||(H1-H2)@psi|| = {diff_norm:.6e}")
    assert diff_norm > 1e-3, (
        f"Multi-mode sine v0=0: ||diff||={diff_norm:.3e} < 1e-3. "
        f"Expected non-zero with sqrt-symmetrized H (no null space)."
    )


def test_check_ic_function_accepts_good_ic():
    """check_ic_breaks_degeneracy() should pass for asymmetric spike."""
    nx = 7
    dx = 63.0

    u0, v0 = _build_ic("asymmetric_spike", nx=nx, dx=dx)
    mu_bc_low, mu_bc_high, rho_bc = _make_probe_arrays(nx, 0.5e10, 5.0e10)

    diff_norm = check_ic_breaks_degeneracy(
        mu_bc_low, mu_bc_high, rho_bc, dx, nx, u0, v0, tol=1e-3
    )
    assert diff_norm > 1e-3
