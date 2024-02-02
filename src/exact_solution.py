import pandas as pd
from qutip import basis, destroy, mesolve, qeye, tensor


def a_op(N: int):
    """
    Constructs the photon annihilation operator
    for the Tavis-Cummings system with N two-level systems:
    a ⊗ I_1 ⊗ … ⊗ I_i ⊗ … ⊗ I_N
    where I_i is the identity operator on the ith two-level system,
    and a is the photon annihilation operator.
    """
    return tensor(destroy(2), *[qeye(2) for _ in range(N)])


def sigma_minus_op(N: int, i: int):
    """
    Constructs the emitter lowering operator for the Tavis-Cummings system with N two-level systems:
    I_c ⊗ I_1 ⊗ … ⊗ σ_i ⊗ … ⊗ I_N
    where σ_i is the lowering operator on the ith two-level system.
    """
    return tensor(
        qeye(2),
        *[qeye(2) for _ in range(i - 1)],
        destroy(2),
        *[qeye(2) for _ in range(N - i)],
    )


def tavis_cummings_hamiltonian(N: int, g: float):
    a = a_op(N)

    H = (
        a.dag() * a
        + sum(sigma_minus_op(N, i).dag() * sigma_minus_op(N, i) for i in range(1, N + 1))
        + g * sum(a.dag() * sigma_minus_op(N, i) + a * sigma_minus_op(N, i).dag() for i in range(1, N + 1))
    )

    return H


def one_excited_emitter_state(N: int, i: int):
    return tensor(
        basis(2, 0),
        *[basis(2, 0) for _ in range(i - 1)],
        basis(2, 1),
        *[basis(2, 0) for _ in range(N - i)],
    )


def solve_master_equation(N: int, g: float, kappa: float, initial_state, times) -> pd.DataFrame:
    H = tavis_cummings_hamiltonian(N, g)
    c_op = (kappa ** 0.5) * a_op(N)
    e_ops = [a_op(N).dag() * a_op(N), *[sigma_minus_op(N, i).dag() * sigma_minus_op(N, i) for i in range(1, N + 1)]]
    result = mesolve(H, tlist=times, rho0=initial_state, c_ops=c_op, e_ops=e_ops)

    populations = pd.DataFrame(
        {
            "Emitter 1": result.expect[1],
            "Emitter 2": result.expect[2],
            "Emitter 3": result.expect[3],
            "Cavity + Environment": result.expect[0] + 1 - sum(result.expect),
        },
        index=pd.Index(times, name="Time"),
    )
    return populations
