"""Test cases for omnisolver.bruteforce.cpu.gray_code module."""
import pytest

from omnisolver.bruteforce.cpu.gray_code import gray_code_index, nth_gray_number


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
        assert all(
            gray_code_index(nth_gray_number(n)) == n for n in range(2 ** num_bits)
        )

    def test_gray_code_index_inverts_nth_gray_code(self, num_bits):
        assert all(
            nth_gray_number(gray_code_index(n)) == n for n in range(2 ** num_bits)
        )
