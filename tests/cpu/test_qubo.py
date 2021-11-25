import numpy as np
import pytest

from omnisolver.bruteforce.cpu.qubo import qubo


@pytest.fixture
def q_mat():
    return np.array(
        [
            [-16, 4, 12, -4],
            [4, 15, -16, 4],
            [12, -16, 0, 3],
            [-4, 4, 3, -5],
        ]
    )


class TestQubo:
    @pytest.mark.parametrize(
        "q_mat",
        [
            np.array([[1.0, -1.0, 0.5], [0.5, 2.0, 3.0]]),
            np.array([[-1.0, 2.0], [-2.0, 0.5]]),
        ],
    )
    def test_cannot_be_initialized_with_non_symmetric_coefficient_matrix(self, q_mat):
        with pytest.raises(ValueError):
            qubo(q_mat)

    @pytest.mark.parametrize(
        "state, expected_energy",
        [
            (np.zeros(4, dtype=np.int8), 0),  # sanity check
            (np.ones(4, dtype=np.int8), -3),  # equal to sum of upper triangle
            (np.array([0, 1, 0, 1], dtype=np.int8), 14),
        ],
    )
    def test_correctly_computes_energy_given_state(self, q_mat, state, expected_energy):
        assert qubo(q_mat).energy(state) == expected_energy
