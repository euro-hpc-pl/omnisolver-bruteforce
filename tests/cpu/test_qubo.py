import numpy as np
import pytest

from omnisolver.bruteforce.cpu.qubo import qubo_from_matrix


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
            qubo_from_matrix(q_mat)

    @pytest.mark.parametrize(
        "state, expected_energy",
        [
            (np.zeros(4, dtype=np.int8), 0),  # sanity check
            (np.ones(4, dtype=np.int8), -3),  # equal to sum of upper triangle
            (np.array([0, 1, 0, 1], dtype=np.int8), 14),
        ],
    )
    def test_correctly_computes_energy_given_state(self, q_mat, state, expected_energy):
        assert qubo_from_matrix(q_mat).energy(state) == expected_energy

    @pytest.mark.parametrize(
        "state, bit_to_flip",
        [
            (np.zeros(4, dtype=np.int8), 3),
            (np.zeros(4, dtype=np.int8), 0),
            (np.array([1, 1, 0, 1], dtype=np.int8), 3),
            (np.array([0, 1, 1, 0], dtype=np.int8), 0),
            (np.array([0, 1, 1, 0], dtype=np.int8), 1),
            (np.ones(4, dtype=np.int8), 0),
            (np.ones(4, dtype=np.int8), 3),
        ],
    )
    def test_correctly_computes_energy_diff_given_state_and_bit_to_flip(
        self, q_mat, state, bit_to_flip
    ):
        flipped = state.copy()
        flipped[bit_to_flip] = 1 - flipped[bit_to_flip]
        qubo = qubo_from_matrix(q_mat)
        assert qubo.energy_diff(state, bit_to_flip) == qubo.energy(
            flipped
        ) - qubo.energy(state)

    @pytest.mark.parametrize(
        "state, k, j",
        [
            (np.zeros(4, dtype=np.int8), 3, 3),
            (np.zeros(4, dtype=np.int8), 0, 2),
            (np.array([1, 1, 0, 1], dtype=np.int8), 3, 1),
            (np.array([0, 1, 1, 0], dtype=np.int8), 0, 0),
            (np.array([0, 1, 1, 0], dtype=np.int8), 1, 1),
            (np.ones(4, dtype=np.int8), 0, 1),
            (np.ones(4, dtype=np.int8), 3, 2),
        ],
    )
    def test_correctly_adjusts_energy_diff_when_another_bit_is_flipped(
        self, q_mat, state, k, j
    ):
        qubo = qubo_from_matrix(q_mat)
        flipped = state.copy()
        flipped[j] = 1 - flipped[j]

        energy_diff_k = qubo.energy_diff(state, k)

        assert qubo.energy_diff(flipped, k) == qubo.adjust_energy_diff(
            state, k, energy_diff_k, j
        )

    @pytest.mark.parametrize("num_variables", [3, 8, 11])
    def test_correctly_defines_its_number_of_variables(self, num_variables):
        rng = np.random.default_rng(42)

        q_mat = rng.random((num_variables, num_variables))
        q_mat += q_mat.T

        qubo = qubo_from_matrix(q_mat)
        assert qubo.num_variables == num_variables
