from functools import lru_cache

import numba
import numpy as np


class DenseQubo:
    def __init__(self, q_mat: np.ndarray):
        self.q_mat = q_mat

    def energy(self, state: np.ndarray) -> float:
        energy = 0.0
        for i in range(self.q_mat.shape[0]):
            for j in range(i, self.q_mat.shape[0]):
                energy += state[i] * state[j] * self.q_mat[i, j]

        return energy

    def energy_diff(self, state, bit_to_flip) -> float:
        sigma = 1 - 2 * state[bit_to_flip]

        return (self.q_mat[bit_to_flip, :] * state * sigma).sum() + self.q_mat[
            bit_to_flip, bit_to_flip
        ] * (1 - state[bit_to_flip])


@lru_cache
def _create_qubo_cls(spec):
    return numba.experimental.jitclass(spec)(DenseQubo)


def qubo_from_matrix(q_mat) -> DenseQubo:
    if not np.array_equal(q_mat, q_mat.T):
        raise ValueError("QUBO matrix needs to be symmetric.")

    spec = (("q_mat", numba.typeof(q_mat)),)

    qubo_cls = _create_qubo_cls(spec)

    return qubo_cls(q_mat)
