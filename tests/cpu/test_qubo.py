import numpy as np
import pytest

from omnisolver.bruteforce.cpu.qubo import qubo


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
