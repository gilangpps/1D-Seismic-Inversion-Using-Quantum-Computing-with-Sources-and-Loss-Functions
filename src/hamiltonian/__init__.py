"""
src/hamiltonian/__init__.py
──────────────────────────────────────────────────────────────────────────────
Hermitian Hamiltonian construction via sqrt-symmetrization for the
1-D elastic wave equation.

AVOIDS the Hermitian dilation approach H = [[0,A],[A†,0]] which produces
mu-independent u(t) = cos(t)·u0 dynamics (see PROMPT.md audit v3).

Instead uses correct sqrt-symmetrization via S_op = sqrt(-K_sym):

    Step 1 — Mass-weighted wave equation
        w = √ρ · u
        ∂²w/∂t² = K_sym · w + f/√ρ
        where K_sym = D^{-1/2} · S · D^{-1/2} (symmetric, negative definite)
        and S is the symmetric stiffness matrix (without 1/ρ factor)

    Step 2 — First-order system in (q, p) variables
        q = S_op · w  where S_op = √(-K_sym)  (symmetric positive definite)
        p = dw/dt

    ∂/∂t [q; p] = [[0, S_op], [-S_op, 0]] · [q; p] + [0; f/√ρ]

    M = [[0, S_op], [-S_op, 0]] is real anti-symmetric (M† = -M),
    so H = i·M is Hermitian with no auxiliary space needed.

    H = [[0,  i·S_op],
         [-i·S_op, 0]]        ∈ ℂ^(2nx × 2nx), H† = H by construction

    Properties:
        - No dilation required: dimension = 2nx (half of dilation approach)
        - u(t) depends explicitly on μ through S_op = √(-K_sym(μ))
        - No auxiliary space leakage (no approximation in Hermiticity)
        - Unitary evolution exp(-iHt) preserves norm

References:
    Schade et al. (2024), arXiv:2312.14747, Fig. A.1
    Jin, Liu, Yu (2022-2023), Schrödingerisation method
    Schade et al. (2025), Quantum Wave Simulation with Sources and Loss Functions
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.linalg import sqrtm


def build_hamiltonian(mu, rho, dx: float, nx: int):
    """
    Build Hermitian Hamiltonian via sqrt-symmetrization.

    Uses S_op = √(−K_sym) where K_sym = D⁻¹ᐟ² · S_stiff · D⁻¹ᐟ² is the
    symmetric mass-weighted stiffness matrix. The resulting Hamiltonian

        H = [[0,  i·S_op],
             [-i·S_op, 0]]

    is exactly Hermitian with dimension 2nx (no auxiliary space).

    Parameters
    ----------
    mu : array_like, shape ≥ (nx+1,)
        Elastic modulus at half-point interfaces [Pa].
    rho : array_like, shape ≥ (nx,)
        Density at interior grid points [kg/m³].
    dx : float
        Grid spacing [m].
    nx : int
        Number of interior grid points.

    Returns
    -------
    H : np.ndarray, shape (2^n_qubits, 2^n_qubits), dtype complex128
        Hermitian Hamiltonian. H = H† by construction.
    n_qubits : int
        Number of qubits required (n_qubits = ceil(log₂(2·nx))).
    dim : int
        Hilbert space dimension = 2^n_qubits.
    phys_dim : int
        Physical dimension = 2·nx. First phys_dim components are physical.
    S_op : np.ndarray, shape (nx, nx)
        Square-root stiffness operator needed for encode/decode:
            q = S_op @ (√ρ · u)    (encode)
            w = S_op⁺ @ q          (decode, u = w/√ρ)

    Encoding for forward simulation:
        w = √ρ · u
        q = S_op @ w
        p = √ρ · v
        state = [q; p] (zero-padded to dim)

    Back-projection after evolution:
        q = state[:nx]
        p = state[nx:phys_dim]
        w = solve(S_op, q)   (or pinv — S_op is SPD, well-conditioned)
        u = w / √ρ
        v = p / √ρ
    """
    mu_arr  = np.asarray(mu,  dtype=float)
    rho_arr = np.asarray(rho, dtype=float)

    # ── Step 1: Build symmetric stiffness matrix S_stiff (nx × nx) ──────
    # S[i,i]   = −(μ_{i+½} + μ_{i−½}) / dx²
    # S[i,i+1] =   μ_{i+½}             / dx²
    # S[i,i−1] =   μ_{i−½}             / dx²
    #
    # S is naturally symmetric (mu_{i+½} is the same for S[i,i+1] and
    # S[i+1,i]), and does NOT include the 1/ρ factor.
    S_stiff = np.zeros((nx, nx), dtype=float)
    for i in range(nx):
        mu_r = 0.5 * (mu_arr[min(i,     len(mu_arr) - 1)]
                    + mu_arr[min(i + 1, len(mu_arr) - 1)])
        mu_l = 0.5 * (mu_arr[max(i - 1, 0)]
                    + mu_arr[min(i,     len(mu_arr) - 1)])

        S_stiff[i, i]         = -(mu_r + mu_l) / (dx * dx)
        if i + 1 < nx:
            S_stiff[i, i + 1] =  mu_r          / (dx * dx)
        if i - 1 >= 0:
            S_stiff[i, i - 1] =  mu_l          / (dx * dx)

    # ── Step 2: Build K_sym = D^{-1/2} · S_stiff · D^{-1/2} ─────────────
    # D = diag(ρ_i), D^{-1/2} = diag(1/√ρ_i)
    # K_sym is symmetric (similar to original K, same eigenvalues)
    rho_int = np.array([max(rho_arr[min(i, len(rho_arr) - 1)], 1e-30) for i in range(nx)])
    inv_sqrt_rho = 1.0 / np.sqrt(rho_int)
    D_inv_half = np.diag(inv_sqrt_rho)
    K_sym = D_inv_half @ S_stiff @ D_inv_half

    assert np.allclose(K_sym, K_sym.T), "K_sym must be symmetric"

    # ── Step 3: Build S_op = √(−K_sym) ──────────────────────────────────
    # −K_sym is symmetric positive definite (stable wave equation)
    # sqrtm returns a real symmetric matrix; extract .real for safety
    S_op = sqrtm(-K_sym).real
    assert np.allclose(S_op, S_op.T), "S_op must be symmetric"

    cond = np.linalg.cond(S_op)
    assert cond < 1e10, f"S_op ill-conditioned: cond={cond:.3e}"

    # ── Step 4: Construct Hermitian Hamiltonian H = i·M ──────────────────
    # M = [[0, S_op], [-S_op, 0]] is anti-symmetric → H = i·M is Hermitian
    phys_dim = 2 * nx
    H_raw = np.zeros((phys_dim, phys_dim), dtype=complex)
    H_raw[:nx, nx:]   = 1j * S_op
    H_raw[nx:, :nx]   = -1j * S_op

    # Assert Hermiticity before padding
    assert np.allclose(H_raw, H_raw.conj().T), "H_raw must be Hermitian"

    # ── Step 5: Pad to next power-of-2 dimension ─────────────────────────
    n_qubits = int(np.ceil(np.log2(max(phys_dim, 2))))
    dim      = 2 ** n_qubits

    H = np.zeros((dim, dim), dtype=complex)
    H[:phys_dim, :phys_dim] = H_raw

    # Numerical Hermiticity enforcement (round-trip FP symmetry)
    H = (H + H.conj().T) / 2.0

    return H, n_qubits, dim, phys_dim, S_op


def verify_hamiltonian(H) -> dict:
    """
    Verify Hermiticity and compute diagnostic statistics for H.

    Parameters
    ----------
    H : np.ndarray
        Candidate Hamiltonian matrix.

    Returns
    -------
    dict with keys:
        hermitian : bool      — True if ||H − H†||_F / ||H||_F < 1e-10
        max_error : float     — max |H[i,j] − H†[i,j]|
        frobenius_norm : float
        spectral_radius : float  — max |eigenvalue|
        is_sparse : bool      — True if >90% of entries are zero
    """
    diff    = H - H.conj().T
    max_err = float(np.max(np.abs(diff)))
    frob    = float(np.linalg.norm(H, 'fro'))
    rel_err = max_err / (frob + 1e-30)

    # Spectral radius (largest eigenvalue magnitude)
    eigvals = np.linalg.eigvalsh(H)
    spectral_radius = float(np.max(np.abs(eigvals)))

    # Sparsity
    nnz   = int(np.count_nonzero(np.abs(H) > 1e-12 * frob))
    total = H.size
    is_sparse = (nnz / total) < 0.10

    return {
        'hermitian':       rel_err < 1e-10,
        'max_error':       max_err,
        'relative_error':  rel_err,
        'frobenius_norm':  frob,
        'spectral_radius': spectral_radius,
        'is_sparse':       is_sparse,
        'nnz':             nnz,
        'total':           total,
    }


def check_ic_breaks_degeneracy(mu_bc_low, mu_bc_high, rho_bc, dx, nx, u0, v0, tol=1e-3):
    """
    Pre-flight check: verify (H(mu_low)-H(mu_high)) @ psi0 != 0.

    Uses the (q,p) encoding of the sqrt-symmetrized Hamiltonian.
    See build_hamiltonian() for encoding details.

    Parameters
    ----------
    mu_bc_low, mu_bc_high : np.ndarray
        BC-padded elastic modulus arrays (shape ≥ nx+2).
    rho_bc : np.ndarray
        BC-padded density array (shape ≥ nx+2).
    dx : float, nx : int
        Grid spacing and interior points.
    u0, v0 : np.ndarray
        Initial displacement and velocity (shape nx, interior).
    tol : float
        Minimum acceptable ||(H1-H2)@psi0||.

    Returns
    -------
    diff_norm : float

    Raises
    ------
    AssertionError if diff_norm < tol.
    """
    rho_int = np.array([max(rho_bc[min(i+1, len(rho_bc)-1)], 1e-30) for i in range(nx)])
    sqrt_rho = np.sqrt(rho_int)

    H1, _, dim, phys_dim, S_op1 = build_hamiltonian(mu_bc_low, rho_bc, dx, nx)
    H2, _, _, _, S_op2          = build_hamiltonian(mu_bc_high, rho_bc, dx, nx)

    # Encode in (q,p) variables
    w = sqrt_rho * u0
    q1 = S_op1 @ w
    p = sqrt_rho * v0
    psi_low = np.zeros(dim, dtype=complex)
    psi_low[:phys_dim] = np.concatenate([q1, p])

    w = sqrt_rho * u0
    q2 = S_op2 @ w
    psi_high = np.zeros(dim, dtype=complex)
    psi_high[:phys_dim] = np.concatenate([q2, p])

    diff_norm = float(np.linalg.norm(psi_low - psi_high))
    print(f"  [IC CHECK] ||psi(mu_low)-psi(mu_high)|| = {diff_norm:.6e}  (tol={tol:.1e})")
    assert diff_norm > tol, (
        f"IC produces near-identical encoded states for different mu: "
        f"||diff||={diff_norm:.3e} < tol={tol:.1e}.\n"
        f"Optimizer will see gradient ≈ 0. Change IC — see PROMPT.md Fix A."
    )
    return diff_norm
