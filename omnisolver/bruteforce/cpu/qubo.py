from functools import lru_cache
from typing import Protocol

import numba
import numpy as np


@numba.njit
def _sigma(x):
    """Map binary variable to spin variable."""
    return 1 - 2 * x


class Qubo(Protocol):
    """Quadratic Unconstrained Binary Optimization problem.

    This Protocol serves as the definition of QUBO-related operations needed for
    implementing exhaustive-search on CPU.

    Currently, the only implementation is DenseQubo. However, in the future we might want to
    experiment e.g. with some sparse implementation.

    The operations defined in this protocol originate in:
    M. Tao et al., "A Work-Time Optimal Parallel Exhaustive Search Algorithm for the QUBO and the
    Ising model, with GPU implementation"
    2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW),
    2020, pp. 557-566, doi: 10.1109/IPDPSW50202.2020.00098.

    The following definitions are used in the docstrings of public methods:

    E(q): energy of given state q (computed w.r.t. to this QUBO's matrix)
    f_k(q): state obtained by flipping k-th bit of q
    delta_k(q): E(f_k(q)) - E(q)

    Note that unlike the original paper mentioned above, in our implementation energy is computed
    by iterating over coefficients only in the upper triangle of QUBO matrix, and hence some
    formulas are a bit different.
    """

    def energy(self, state: np.ndarray) -> float:
        """Compute E(state).

        :param state: state which energy is to be computed. For performance reasons, it is not
         checked if the state has correct length.
        :return: E(state)
        """

    def energy_diff(self, state, k) -> float:
        """Compute delta_k(state) = E(f_k(state)) - E(state) in Theta(N) time.

        :param state: state for which energy difference should be computed. For performance
         reasons, it is not checked if the state has correct length.
        :param k: index of bit to be flipped. For performance reasons it is not checked
         if k lies in correct range.
        :return: delta_k(state)
        """

    def adjust_energy_diff(self, state, k, energy_diff_k, j) -> float:
        """Compute delta_k(f_j(state)) given delta_k(state) in Theta(1) time.

        :param state: state for which energy difference adjustment should be computed. For
         performance reasons, it is not checked if the state has correct length.
        :param k: index of bit such that delta_k(state) is known. For performance reasons it is not
         checked if k lies in the correct range.
        :param energy_diff_k: delta_k(state)
        :param j: index of bit to be flipped. For performance reasons it is not checked if j lies
         in the correct range.
        :return: delta_k(f_j(state))
        """

    @property
    def num_variables(self) -> int:
        """Number of variables in this model."""
        raise NotImplementedError()


class DenseQubo:
    """Basic dense QUBO for use in CPU-based exhaustive search algorithm.

    This class implements Qubo protocol. See its docstring for more elaborate description of public
    methods.
    """

    def __init__(self, q_mat: np.ndarray) -> None:
        """Initialize new DenseQubo instance.

        :param q_mat: coefficient matrix. It should be symmetric, but for performance reasons
         this assumption is not verified. For safe initialization use `qubo_from_matrix(q_mat)`
        """
        self.q_mat = q_mat

    def energy(self, state: np.ndarray) -> float:
        """Compute E(state)."""
        energy = 0.0
        for i in range(self.q_mat.shape[0]):
            for j in range(i, self.q_mat.shape[0]):
                energy += state[i] * state[j] * self.q_mat[i, j]

        return energy

    def energy_diff(self, state, k) -> float:
        """Compute delta_k(state) = E(f_k(state)) - E(state) in Theta(N) time."""
        # fmt: off
        return (
            (self.q_mat[k, :] * state * _sigma(state[k])).sum() +
            self.q_mat[k, k] * (1 - state[k])
        )
        # fmt: on

    def adjust_energy_diff(self, state, k, energy_diff_k, j) -> float:
        """Compute delta_k(f_j(state)) given delta_k(state) in Theta(1) time."""
        if k == j:
            return -energy_diff_k
        return energy_diff_k + self.q_mat[k, j] * _sigma(state[j]) * _sigma(state[k])

    @property
    def num_variables(self) -> int:
        return self.q_mat.shape[0]


@lru_cache
def _create_dense_qubo_cls(spec):
    return numba.experimental.jitclass(spec)(DenseQubo)


def qubo_from_matrix(q_mat: np.ndarray) -> Qubo:
    """Create jit-compiled QUBO instance from given symmetric matrix.

    .. note::
       Currently this function always returns jit-compiled DenseQubo instance. However, in the
       future this might change to a more polymorphic behaviour (e.g. if we decide to implement
       sparse QUBO. For this reason, using `qubo_from_matrix` should be preferred method of
       constructing QUBO instance.

    :param q_mat: symmetric coefficient matrix
    :return: an instance of QUBO
    :raises ValueError: if passed array is not a symmetric matrix.
    """
    if not np.array_equal(q_mat, q_mat.T):
        raise ValueError("QUBO matrix needs to be symmetric.")

    spec = (("q_mat", numba.typeof(q_mat)),)

    qubo_cls = _create_dense_qubo_cls(spec)

    return qubo_cls(q_mat)
