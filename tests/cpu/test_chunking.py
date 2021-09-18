import numpy as np
import pytest

from omnisolver.bruteforce.cpu.chunking import initial_state_for_chunk


def _fixed_width_binary_repr(number, width):
    bits = []
    for _ in range(width):
        bits.append(number % 2)
        number //= 2
    return bits


@pytest.mark.parametrize(
    "num_bits, num_bits_in_chunk, chunk_id",
    [(30, 20, 1), (8, 4, 0), (40, 32, 1), (40, 32, 128)],
)
class TestInitialStateForChunk:
    def test_contains_all_zeros_for_expected_number_of_first_bits(
        self, num_bits, num_bits_in_chunk, chunk_id
    ):
        initial_state = initial_state_for_chunk(num_bits, num_bits_in_chunk, chunk_id)
        np.testing.assert_array_equal(initial_state[:num_bits_in_chunk], 0)

    def test_contains_binary_representation_of_chunk_id_on_last_bits(
        self, num_bits, num_bits_in_chunk, chunk_id
    ):
        initial_state = initial_state_for_chunk(num_bits, num_bits_in_chunk, chunk_id)
        remaining_bits = num_bits - num_bits_in_chunk
        np.testing.assert_array_equal(
            initial_state[-remaining_bits:],
            _fixed_width_binary_repr(chunk_id, remaining_bits),
        )
