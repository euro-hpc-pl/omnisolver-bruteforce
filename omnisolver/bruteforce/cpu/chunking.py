import numba
import numpy as np


@numba.njit
def initial_state_for_chunk(num_bits, num_bits_in_chunk, chunk_index):
    state = np.zeros(num_bits, dtype=np.int8)
    remaining_bits = num_bits - num_bits_in_chunk
    for i in range(remaining_bits):
        state[-remaining_bits + i] = chunk_index & 1
        chunk_index >>= 1
    return state
