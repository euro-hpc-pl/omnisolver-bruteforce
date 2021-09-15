import numba
import numpy as np


@numba.njit(numba.int64(numba.int64))
def nth_gray_number(n: int) -> int:
    return n ^ (n >> 1)


@numba.njit(numba.int64(numba.int64))
def gray_code_index(gray_code: int) -> int:
    mask = gray_code
    while mask:
        mask >>= 1
        gray_code ^= mask
    return gray_code


@numba.njit(numba.void(numba.int8[:]))
def advance_focus_vector(f_vec: np.ndarray):
    j = f_vec[0]
    f_vec[0] = 0
    f_vec[j] = f_vec[j + 1]
    f_vec[j + 1] = j + 1
