import numpy as np


def amplitude_encode(field):
    vec = np.array(field, dtype=np.complex128).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None, 0, 0.0
    vec = vec / norm
    L = len(vec)
    n_qubits = (L - 1).bit_length()
    target_len = 1 << n_qubits
    if target_len != L:
        padded = np.zeros(target_len, dtype=np.complex128)
        padded[:L] = vec
        vec = padded
    return vec, n_qubits, norm


def quantum_reconstruct(field, shots=None, noise_level=0.0):
    sv, n_q, norm = amplitude_encode(field)
    if sv is None:
        return np.zeros_like(field)
    nx = len(field)
    if shots is None and noise_level == 0.0:
        return np.real(sv[:nx]) * norm
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(sv))
        sv_noisy = sv + noise
        sv_noisy = sv_noisy / np.linalg.norm(sv_noisy)
    else:
        sv_noisy = sv
    if shots is not None:
        probs = np.abs(sv_noisy) ** 2
        probs = probs / probs.sum()
        counts = np.random.multinomial(shots, probs)
        p_est = counts / shots
        signs = np.sign(np.real(sv_noisy))
        sv_est = signs * np.sqrt(np.maximum(p_est, 0.0))
        sv_norm = np.linalg.norm(sv_est)
        if sv_norm > 0:
            sv_est = sv_est / sv_norm
        return np.real(sv_est[:nx]) * norm
    return np.real(sv_noisy[:nx]) * norm
