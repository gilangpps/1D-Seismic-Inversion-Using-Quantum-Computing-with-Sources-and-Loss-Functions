"""
src/encoding/__init__.py
──────────────────────────────────────────────────────────────────────────────
Quantum Amplitude Encoding for classical wavefield data.

Reference: Schade et al. (2024), arXiv:2312.14747, §III

REQUIREMENT E — Encoding:
    Classical field u(x) is mapped into quantum state:
        |ψ⟩ = (1/‖u‖) · Σ_i u[i] |i⟩

    where |i⟩ is the i-th computational basis state.

    Properties:
    • Normalization: ⟨ψ|ψ⟩ = 1  (valid quantum state)
    • Reversible:   u[i] = ‖u‖ · Re⟨i|ψ⟩  (exact recovery for real u)
    • Basis mapping: each grid point maps to one basis state
    • Dimension: padded to next power-of-2 for n-qubit representation

RECONSTRUCTION:
    From statevector (noiseless):
        u_rec[i] = norm · Re(ψ[i])   for i < len(u)

    From shot-noise measurement:
        Simulate projective measurement:
            p_i = |ψ[i]|²
        Estimate probabilities from counts:
            p̂_i = counts[i] / shots
        Reconstruct with sign inference from statevector:
            ψ̂[i] = sign(Re ψ[i]) · √p̂_i
        Renormalize and scale by original norm.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Amplitude encoding
# ══════════════════════════════════════════════════════════════════════════════

def amplitude_encode(field):
    """
    Encode a classical field as a normalized quantum state vector.

    Implements the mapping:
        |ψ⟩ = (1/‖u‖) · Σ_i u[i] |i⟩

    The state vector is padded to the next power-of-2 length so that
    it can be loaded into an n-qubit quantum register.

    Parameters
    ----------
    field : array_like
        Classical wavefield u[i], real or complex.

    Returns
    -------
    sv : np.ndarray, shape (2^n_qubits,), dtype complex128
        Normalized state vector |ψ⟩.  None if field is all-zero.
    n_qubits : int
        Number of qubits needed.
    norm : float
        Original field norm ‖u‖.  Required for reconstruction.

    Notes
    -----
    For reconstruction: u[i] = norm · Re(sv[i])  (exact for real fields).
    """
    vec  = np.asarray(field, dtype=np.complex128).ravel()
    norm = float(np.linalg.norm(vec))

    if norm == 0.0:
        return None, 0, 0.0

    sv = vec / norm

    L        = len(sv)
    n_qubits = max(1, (L - 1).bit_length())   # ceil(log₂(L))
    dim      = 1 << n_qubits                  # 2^n_qubits

    if dim != L:
        padded = np.zeros(dim, dtype=np.complex128)
        padded[:L] = sv
        sv = padded

    return sv, n_qubits, norm


# ══════════════════════════════════════════════════════════════════════════════
#  Quantum reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def quantum_reconstruct(field, shots: Optional[int] = None,
                        noise_level: float = 0.0):
    """
    Reconstruct a classical field from quantum measurement simulation.

    This implements the three-mode reconstruction protocol:

    Mode 1 — Statevector (noiseless, shots=None, noise_level=0):
        Returns exact classical field rescaled by norm.
        Used for deterministic gradient computation.

    Mode 2 — Shot-noise (shots given, noise_level=0):
        Simulates projective measurement via multinomial sampling.
        Reconstructed amplitude:
            ψ̂[i] = sign(Re ψ[i]) · √(counts[i] / shots)
        Models what a real quantum computer would output.

    Mode 3 — Hardware-noise (noise_level > 0):
        Adds Gaussian noise to statevector before measurement simulation.
        Models gate errors and decoherence.

    Parameters
    ----------
    field : array_like
        Classical wavefield u[i].
    shots : int or None
        Number of measurement shots.  None = noiseless statevector.
    noise_level : float
        Standard deviation of additive Gaussian noise on statevector.
        0.0 = no noise.

    Returns
    -------
    np.ndarray, shape (len(field),)
        Reconstructed classical field.
    """
    sv, n_q, norm = amplitude_encode(field)

    if sv is None:
        return np.zeros(len(field), dtype=float)

    nx = len(field)

    # Mode 1: noiseless statevector reconstruction
    if shots is None and noise_level == 0.0:
        return np.real(sv[:nx]) * norm

    # Mode 3: hardware noise on statevector
    if noise_level > 0.0:
        noise      = np.random.normal(0.0, noise_level, len(sv))
        sv_noisy   = sv + noise
        sv_norm    = np.linalg.norm(sv_noisy)
        sv_noisy   = sv_noisy / (sv_norm + 1e-30)
    else:
        sv_noisy   = sv

    # Mode 2: shot-noise measurement simulation
    if shots is not None:
        probs = np.abs(sv_noisy) ** 2
        probs = probs / (probs.sum() + 1e-30)          # renormalise

        counts = np.random.multinomial(shots, probs)
        p_est  = counts / shots

        # Sign inference from statevector (required: real part)
        signs  = np.sign(np.real(sv_noisy))
        signs[signs == 0] = 1.0                        # break ties

        sv_est = signs * np.sqrt(np.maximum(p_est, 0.0))
        sv_norm = np.linalg.norm(sv_est)
        if sv_norm > 0.0:
            sv_est = sv_est / sv_norm

        return np.real(sv_est[:nx]) * norm

    # Fallback: noisy statevector without shot measurement
    return np.real(sv_noisy[:nx]) * norm
