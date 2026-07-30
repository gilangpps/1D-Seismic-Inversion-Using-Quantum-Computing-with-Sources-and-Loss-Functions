"""
src/circuit/__init__.py
──────────────────────────────────────────────────────────────────────────────
Quantum Circuit Builder — Hamiltonian Time Evolution

Reference: Schade et al. (2024), arXiv:2312.14747, Fig. A.1
           Schade et al. (2025), Quantum Wave Simulation with Sources and Loss Functions

REQUIREMENT F — Circuit structure:
    The circuit implements the Schrödinger evolution:
        |ψ(t)⟩ = exp(−iHt) |ψ(0)⟩

    where H is the physical Hamiltonian from src/hamiltonian.

    Four sections (matching paper Fig. A.1):
    ┌──────────────┐ ░ ┌──────────────┐ ░ ┌────────────┐ ░ ┌─────────┐
    │ State Prep   ├─░─┤  exp(−iHt)   ├─░─┤ Observable ├─░─┤ Measure │
    └──────────────┘ ░ └──────────────┘ ░ └────────────┘ ░ └─────────┘

    1. State Preparation:
       Encodes |ψ(0)⟩ = amplitude_encode([√ρ·u, (1/√ρ)·v]) using
       Ry, X, Z gates (or full initialize for multi-component states).

    2. Time Evolution:
       Exact matrix exponential U = expm(−iHt) as UnitaryGate.
       H is built by build_hamiltonian(μ, ρ, dx, nx).

    3. Observable Rotation:
       Hadamard gates on selected qubits for X-basis measurement.

    4. Measurement:
       Projective measurement of all qubits in computational basis.

PHYSICAL ENCODING:
    The state vector encodes both displacement and velocity:
        |ψ⟩ ∝ [√ρ·u₁, ..., √ρ·uₙ, (1/√ρ)·v₁, ..., (1/√ρ)·vₙ]

    This mass-weighted encoding ensures the inner product
        ⟨ψ₁|ψ₂⟩ = Σᵢ ρᵢ·u₁ᵢ·u₂ᵢ + Σᵢ (1/ρᵢ)·v₁ᵢ·v₂ᵢ
    approximates the physical energy inner product.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from src.hamiltonian import build_hamiltonian


def build_paper_circuit(u0, v0, mu, rho, dx: float, dt: float,
                        time_step_idx: int, nx: int,
                        observable=None, group_idx: int = 0,
                        paper_diagram_mode: bool = False):
    """
    Build the Schade et al. style quantum circuit for elastic wave evolution.

    Parameters
    ----------
    u0 : array_like, shape (nx,)
        Initial displacement field (interior grid points only).
    v0 : array_like, shape (nx,)
        Initial velocity field.
    mu : array_like, shape ≥ (nx+1,)
        Elastic modulus [Pa].
    rho : array_like, shape ≥ (nx,)
        Density [kg/m³].
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].
    time_step_idx : int
        Which time step to evolve to.  Evolution time t = time_step_idx × dt.
    nx : int
        Number of interior grid points.
    observable : list of str or None
        Pauli basis for each qubit: 'X', 'Y', 'Z', or 'I'.
        'X' → Hadamard gate before measurement.
        'Y' → Sdg + Hadamard before measurement.
        'Z' / 'I' → no rotation (computational basis).
        Default: ['X', 'Z', 'X', 'Z'] pattern.
    group_idx : int
        Label index for circuit naming.
    paper_diagram_mode : bool, default False
        If True, build circuit with deterministic paper-style state preparation:
            - q0 (q1_0): RY(-1.7)
            - q2 (q1_2): X + Z
        This ensures a stable, reproducible circuit diagram for visualization.
        If False, use physics-based adaptive state preparation from the
        actual displacement/velocity fields.

    Returns
    -------
    qc : QuantumCircuit
        Complete circuit: StatePrep | TimeEvolution | Observable | Measure.
    metadata : dict
        n_qubits, dim, time, observable, hamiltonian_shape, etc.
    """
    # ── 1. Build Hamiltonian (dilated) ───────────────────────────────────
    H_mat, n_qubits, dim, phys_dim, _ = build_hamiltonian(mu, rho, dx, nx)

    # ── 2. Build initial state vector ────────────────────────────────────
    # Mass-weighted encoding: [√ρ·u, (1/√ρ)·v] in physical subspace
    # Embed into dilated space: [physical_state, auxiliary_zeros]
    u0_arr  = np.asarray(u0,  dtype=float)
    v0_arr  = np.asarray(v0,  dtype=float)
    rho_arr = np.asarray(rho, dtype=float)[:nx]

    sqrt_rho     = np.sqrt(rho_arr)
    inv_sqrt_rho = 1.0 / np.maximum(sqrt_rho, 1e-30)

    u_weighted = sqrt_rho     * u0_arr[:nx]
    v_weighted = inv_sqrt_rho * v0_arr[:nx]

    state    = np.concatenate([u_weighted, v_weighted])  # length 2nx
    
    # Embed into dilated Hilbert space
    psi      = np.zeros(dim, dtype=complex)
    psi[:min(len(state), phys_dim)] = state[:min(len(state), phys_dim)]
    psi_norm = np.linalg.norm(psi)

    if psi_norm > 1e-15:
        psi = psi / psi_norm
    else:
        # Fallback: zero velocity, unit displacement at centre
        psi = np.zeros(dim, dtype=complex)
        psi[nx // 2] = 1.0

    # ── 3. Compute time evolution unitary ────────────────────────────────
    t_evo = float(time_step_idx) * dt
    U     = expm(-1j * H_mat * t_evo)

    # ── 4. Observable pattern ────────────────────────────────────────────
    if observable is None:
        obs = ['X' if (i % 2 == 0) else 'Z' for i in range(n_qubits)]
    else:
        obs = list(observable)
        if len(obs) < n_qubits:
            obs += ['I'] * (n_qubits - len(obs))

    # ── 5. Build circuit ─────────────────────────────────────────────────
    qc = QuantumCircuit(n_qubits, n_qubits,
                        name=f"TimeEvo_G{group_idx}_I{time_step_idx}")

    # Section 1: State Preparation
    _add_state_prep(qc, psi, n_qubits, paper_diagram_mode=paper_diagram_mode)
    qc.barrier()

    # Section 2: Time Evolution  exp(−iHt)
    evo_gate = UnitaryGate(U, label=f'exp(-i{t_evo:.4g}H)')
    qc.append(evo_gate, range(n_qubits))
    qc.barrier()

    # Section 3: Observable rotation
    for i, p in enumerate(obs[:n_qubits]):
        if p == 'X':
            qc.h(i)
        elif p == 'Y':
            qc.sdg(i)
            qc.h(i)
        # 'Z' and 'I': no gate needed

    qc.barrier()

    # Section 4: Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # ── 6. Metadata ──────────────────────────────────────────────────────
    metadata = {
        'n_qubits':          n_qubits,
        'dim':               dim,
        'time':              t_evo,
        'time_step_idx':     time_step_idx,
        'group_idx':         group_idx,
        'observable':        ''.join(obs[:n_qubits]),
        'psi_norm':          float(psi_norm),
        'hamiltonian_shape': list(H_mat.shape),
        'evolution_type':    'expm(-iHt) unitary gate',
        'paper_diagram_mode': paper_diagram_mode,
    }

    return qc, metadata


def _add_state_prep(qc: QuantumCircuit, psi: np.ndarray, n_qubits: int,
                    paper_diagram_mode: bool = False):
    """
    Add state-preparation gates to encode |ψ⟩.

    Parameters
    ----------
    qc : QuantumCircuit  (modified in-place)
    psi : np.ndarray, shape (2^n_qubits,)
        Normalized state vector to prepare.
    n_qubits : int
        Number of qubits in the circuit.
    paper_diagram_mode : bool, default False
        If True, use deterministic paper-style gate pattern:
            - q0: RY(-1.7)
            - q2: X + Z
        This creates a stable, reproducible circuit diagram for figures.
        If False, use adaptive encoding based on the actual state vector.

    Notes
    -----
    In paper_diagram_mode, the circuit is built for visualization fidelity,
    not physical accuracy. The gates are placed deterministically to match
    the intended Schade et al. Figure A.1 layout.

    In physics mode (paper_diagram_mode=False), the gates adapt to the
    actual state amplitudes:
        - Single non-zero amplitude: encode index as computational basis (X gates)
        - Two non-zero amplitudes: RY rotation on the differing qubit
        - General state: use Qiskit initialize + transpile to basis gates
    """
    if paper_diagram_mode:
        # Deterministic paper-style pattern:
        # q0 gets RY(-1.7), q2 gets X and Z
        # This creates the intended figure layout regardless of physics state
        if n_qubits >= 1:
            qc.ry(-1.7, 0)
        if n_qubits >= 3:
            qc.x(2)
            qc.z(2)
        return

    # Physics-based adaptive state preparation
    nonzero_idx = np.where(np.abs(psi) > 1e-10)[0]

    if len(nonzero_idx) == 0:
        # All-zero state (should not happen after normalization)
        return

    if len(nonzero_idx) == 1:
        # Computational basis state: encode index in binary
        idx = nonzero_idx[0]
        for q in range(n_qubits):
            if (idx >> q) & 1:
                qc.x(q)
        if np.real(psi[idx]) < 0:
            qc.z(0)
        return

    if len(nonzero_idx) == 2:
        # Two-amplitude state: RY rotation
        i0, i1  = nonzero_idx
        a0      = float(np.real(psi[i0]))
        a1      = float(np.real(psi[i1]))
        theta   = 2.0 * np.arctan2(abs(a1), abs(a0))
        diff    = i0 ^ i1
        common  = i0 & i1

        # Find the differing qubit: the position of the single set bit in diff
        # OLD BROKEN: q_diff = int(np.log2(diff + 1))  # Wrong for non-power-of-2
        # NEW CORRECT: Extract the bit position directly
        if diff > 0:
            # Find position of the rightmost set bit
            q_diff = (diff & -diff).bit_length() - 1
        else:
            q_diff = 0

        if a1 < 0:
            qc.ry(-theta, q_diff)
        else:
            qc.ry(theta, q_diff)

        # Set common bits
        for q in range(n_qubits):
            if (common >> q) & 1:
                qc.x(q)

        if a1 < 0:
            qc.z(q_diff)
        return

    # General case: use Qiskit initialize + transpile to basis gates
    prep = QuantumCircuit(n_qubits)
    prep.initialize(psi.tolist(), range(n_qubits))
    prep_t = transpile(
        prep,
        basis_gates=['ry', 'rz', 'cx', 'x', 'z', 'h', 'id'],
        optimization_level=0,  # Use level 0 to minimize gate reordering
    )
    for inst in prep_t.data:
        if inst.operation.name not in ('reset', 'barrier', 'id'):
            qc.append(inst.operation, inst.qubits, inst.clbits)


def validate_paper_circuit(qc: QuantumCircuit, paper_diagram_mode: bool = False):
    """
    Validate that the circuit has the expected paper-style gate pattern.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to validate.
    paper_diagram_mode : bool
        If True, enforce strict paper pattern validation.
        If False, skip validation (physics mode allows adaptive gates).

    Raises
    ------
    AssertionError
        If paper_diagram_mode is True and the circuit doesn't match
        the expected pattern:
            - q0 should have RY(-1.7)
            - q2 should have X and Z
            - q0 and q1 should NOT have stray X gates from state prep
    """
    if not paper_diagram_mode:
        # Physics mode: no validation needed
        return

    # Extract state preparation gates (before first barrier)
    gate_counts = {i: [] for i in range(qc.num_qubits)}
    for instruction in qc.data:
        if instruction.operation.name == 'barrier':
            break
        if instruction.operation.name not in ['barrier', 'measure']:
            for q in instruction.qubits:
                q_idx = qc.find_bit(q).index
                gate_counts[q_idx].append(instruction.operation.name)

    # Check expected pattern
    has_ry_on_q0 = any('ry' in g.lower() for g in gate_counts[0])
    has_x_on_q0_only_from_ry = 'x' not in gate_counts[0]
    has_x_on_q1 = 'x' in gate_counts[1]
    has_x_on_q2 = 'x' in gate_counts[2] if len(gate_counts) > 2 else False
    has_z_on_q2 = 'z' in gate_counts[2] if len(gate_counts) > 2 else False

    errors = []
    if not has_ry_on_q0:
        errors.append("q0 missing RY(-1.7) gate")
    if not has_x_on_q0_only_from_ry:
        errors.append("q0 has unwanted X gate (should only have RY)")
    if has_x_on_q1:
        errors.append("q1 has unwanted X gate (should be clean)")
    if not has_x_on_q2:
        errors.append("q2 missing X gate")
    if not has_z_on_q2:
        errors.append("q2 missing Z gate")

    if errors:
        raise AssertionError(
            f"Paper circuit validation failed:\n  " + "\n  ".join(errors)
        )

