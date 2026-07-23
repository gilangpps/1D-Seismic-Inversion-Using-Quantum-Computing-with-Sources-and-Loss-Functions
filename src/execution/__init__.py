"""
src/execution/__init__.py
Circuit execution on AerSimulator backend.
"""

from qiskit            import transpile
from qiskit_aer        import AerSimulator


def execute_circuit(qc, shots: int = 1000) -> dict:
    """
    Transpile and execute a Qiskit circuit on the AerSimulator.

    Parameters
    ----------
    qc    : QuantumCircuit
    shots : int  — number of measurement shots

    Returns
    -------
    dict  — measurement counts {bitstring: count}
    """
    sim        = AerSimulator()
    transpiled = transpile(qc, backend=sim)
    job        = sim.run(transpiled, shots=shots)
    return job.result().get_counts()
