import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit import transpile

from src.hamiltonian import build_hamiltonian


def build_paper_circuit(u0, v0, mu, rho, dx, dt, time_step_idx, nx,
                        observable=None, group_idx=0):
    H_mat, n_qubits, dim = build_hamiltonian(mu, rho, dx, nx)

    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    u_weighted = sqrt_rho * u0
    v_weighted = inv_sqrt_rho * v0
    state = np.concatenate([u_weighted, v_weighted])
    psi = np.zeros(dim)
    psi[:len(state)] = state
    psi_norm = np.linalg.norm(psi)
    if psi_norm > 0:
        psi = psi / psi_norm

    t_evo = time_step_idx * dt
    U = expm(-1j * H_mat * t_evo)

    if observable is None:
        obs = ['I'] * n_qubits
        obs[0] = 'X'
        if n_qubits > 2:
            obs[2] = 'X'
    else:
        obs = list(observable)

    qc = QuantumCircuit(n_qubits, n_qubits,
                        name=f"TimeEvo_G{group_idx}_I{time_step_idx}")

    nonzero_idx = np.where(np.abs(psi) > 1e-10)[0]

    if len(nonzero_idx) == 1:
        idx = nonzero_idx[0]
        for q in range(n_qubits):
            if (idx >> q) & 1:
                qc.x(q)
        if np.real(psi[idx]) < 0:
            qc.z(0)
    elif len(nonzero_idx) == 2:
        i0, i1 = nonzero_idx
        a0, a1 = float(np.real(psi[i0])), float(np.real(psi[i1]))
        theta = 2 * np.arctan2(abs(a1), abs(a0))
        diff_bits = i0 ^ i1
        common_bits = i0 & i1
        sign_neg = (a1 < 0)
        for q in range(n_qubits):
            if (diff_bits >> q) & 1:
                qc.ry(-theta if sign_neg else theta, q)
                break
        for q in range(n_qubits):
            if (common_bits >> q) & 1:
                qc.x(q)
                if sign_neg:
                    qc.z(q)
    else:
        prep_sub = QuantumCircuit(n_qubits)
        prep_sub.initialize(psi.tolist(), range(n_qubits))
        prep_transpiled = transpile(
            prep_sub, basis_gates=['ry', 'rz', 'cx', 'x', 'z', 'h', 'id'],
            optimization_level=2,
        )
        for inst in prep_transpiled.data:
            if inst.operation.name not in ('reset', 'barrier', 'id'):
                qc.append(inst.operation, inst.qubits, inst.clbits)
    qc.barrier()

    evo_gate = UnitaryGate(U, label=f'exp(-i{t_evo:.4g}H)\n{t_evo:.4g}')
    qc.append(evo_gate, range(n_qubits))
    qc.barrier()

    for i, p in enumerate(obs):
        if p == 'X':
            qc.h(i)
        elif p == 'Y':
            qc.sdg(i)
            qc.h(i)
    qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))

    obs_str = ''.join(obs)
    metadata = {
        'n_qubits': n_qubits,
        'dim': dim,
        'time': t_evo,
        'time_step_idx': time_step_idx,
        'group_idx': group_idx,
        'observable': obs_str,
        'psi_norm': psi_norm,
        'hamiltonian_shape': list(H_mat.shape),
    }
    return qc, metadata
