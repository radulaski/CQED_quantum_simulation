import numpy as np
from qiskit import Aer, execute, QuantumCircuit


def cs(N: int, g: float, kappa: float, initial_populations, t: float) -> float:
    D = np.emath.sqrt(-16 * N * g**2 + kappa**2)
    m = 1 - np.exp(-kappa * t / 4) * ((kappa / D) * np.sinh(D * t / 4) + np.cosh(D * t / 4))
    return initial_populations - m * initial_populations.sum() / N


def thetas(N: int, g: float, kappa: float, initial_populations, t: float) -> list[float]:
    c = cs(N, g, kappa, initial_populations, t)
    theta_1 = np.arccos(c[0])

    theta = []
    for i in range(1, N):
        denom = np.sin(theta_1) * np.prod(np.cos(theta))
        theta.append(0 if denom == 0 else np.arcsin(c[i] / denom))
    
    return np.real([theta_1] + theta)


def construct_circuit(N, g, kappa, t):
    c0 = np.zeros(N)
    c0[0] = 1

    theta = thetas(N, g, kappa, c0, t)

    qc = QuantumCircuit(N + 1, N + 1)
    qc.x(0)
    qc.cry(2 * theta[0], 0, N)
    qc.cx(N, 0)
    
    for i in range(N - 1):
        qc.cry(2 * theta[i + 1], N, i + 1)
        qc.cx(i + 1, N)

    qc.measure_all(add_bits=False)

    return qc


def run_qmarina_on_simulator(N: int, g: float, kappa: float, times: np.ndarray, shots: int = 10_000):
    circuits = [construct_circuit(N, g, kappa, t) for t in times]
    simulator = Aer.get_backend('statevector_simulator')
    counts = execute(circuits, backend=simulator, shots=shots).result().get_counts()

    populations = {
        "Emitter 1": [c.get('0001', 0) / shots for c in counts],
        "Emitter 2": [c.get('0010', 0) / shots for c in counts],
        "Emitter 3": [c.get('0100', 0) / shots for c in counts],
        "Cavity + Environment": [c.get('1000', 0) / shots for c in counts],
    }
    return populations
