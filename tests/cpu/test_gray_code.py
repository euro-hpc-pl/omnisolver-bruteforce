"""Test cases for omnisolver.bruteforce.cpu.gray_code module."""
import numpy as np
import pytest

from omnisolver.bruteforce.cpu.gray_code import (
    advance_focus_vector,
    gray_code_index,
    nth_gray_number,
)


@pytest.mark.parametrize(
    "binary_number, gray_code",
    [
        (0, "0"),
        (1, "1"),
        (2, "11"),
        (3, "10"),
        (4, "110"),
        (5, "111"),
        (6, "101"),
        (7, "100"),
    ],
)
class TestConversionBetweenBinaryAndGray:
    def test_nth_gray_code_is_computed_correctly(self, binary_number, gray_code):
        assert bin(nth_gray_number(binary_number))[2:] == gray_code

    def test_gray_code_index_is_computed_correctly(self, binary_number, gray_code):
        assert binary_number == gray_code_index(int(gray_code, 2))


@pytest.mark.parametrize("num_bits", [10, 12, 16])
class TestBijectionBetweenGrayAndBinary:
    def test_nth_gray_code_inverts_gray_code_index(self, num_bits):
        assert all(gray_code_index(nth_gray_number(n)) == n for n in range(2**num_bits))

    def test_gray_code_index_inverts_nth_gray_code(self, num_bits):
        assert all(nth_gray_number(gray_code_index(n)) == n for n in range(2**num_bits))


def _binary_array_to_number(arr):
    return np.sum(arr * 2 ** np.arange(len(arr)))


@pytest.mark.parametrize("num_bits", [4, 6, 8, 10])
def test_iterating_algorithm_l_yields_all_gray_codes(num_bits):
    # Idea of this test: iterate Algorithm L to obtain all consecutive bit flips
    # for Gray code. Apply the switches in order to produce all possible numbers
    # in Gray code order. Determine if this is gives the same results as generating
    # Gray code explicitly.
    focus_vector = np.arange(num_bits + 1, dtype=np.int8)
    state = np.zeros(num_bits, dtype=np.int8)
    produced_numbers = [_binary_array_to_number(state)]

    for _ in range(2**num_bits - 1):
        bit_to_flip = focus_vector[0] % num_bits
        state[bit_to_flip] = 1 - state[bit_to_flip]
        advance_focus_vector(focus_vector)
        produced_numbers.append(_binary_array_to_number(state))

    np.testing.assert_array_equal(
        produced_numbers, [nth_gray_number(n) for n in range(2**num_bits)]
    )
