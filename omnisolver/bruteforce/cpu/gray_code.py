import numba


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
