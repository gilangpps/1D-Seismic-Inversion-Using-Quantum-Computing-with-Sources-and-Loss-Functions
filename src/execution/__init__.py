from qiskit import transpile
from qiskit_aer import AerSimulator


def execute_circuit(qc, shots=1000):
    sim = AerSimulator()
    transpiled = transpile(qc, backend=sim)
    job = sim.run(transpiled, shots=shots)
    return job.result().get_counts()
