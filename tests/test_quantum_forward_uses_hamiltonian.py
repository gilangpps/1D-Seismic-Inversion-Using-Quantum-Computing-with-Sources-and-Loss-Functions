"""
Structural test: verify quantum_forward_simulate calls build_hamiltonian().

This is NOT a numerical test — it checks the source code structure to
prevent silent regressions where the quantum engine is replaced with
real-valued expm(A*t) (classical) evolution without updating the interface.

See AUDIT_REPORT.md "Regresi: Quantum Engine Ter-Regresi Jadi Klasik"
"""

import inspect
from src.optimization.objective import SeismicObjective


def test_quantum_forward_calls_build_hamiltonian():
    src = inspect.getsource(SeismicObjective.quantum_forward_simulate)
    assert "build_hamiltonian" in src, (
        "quantum_forward_simulate does not call build_hamiltonian() — "
        "likely regression to real-valued expm(A*t) evolution"
    )
    assert "1j" in src, (
        "quantum_forward_simulate does not contain complex evolution (1j) — "
        "likely regression to real-valued expm(A*t)"
    )
    assert "sqrt-symmetrized" in src, (
        "quantum_forward_simulate missing sqrt-symmetrization reference"
    )


def test_validation_uses_hamiltonian():
    from src.experiment.validate_hamiltonian import run_hamiltonian_validation
    src = inspect.getsource(run_hamiltonian_validation)
    assert "build_hamiltonian" in src, (
        "run_hamiltonian_validation does not call build_hamiltonian()"
    )
    assert "1j" in src, (
        "run_hamiltonian_validation does not contain complex evolution (1j)"
    )
