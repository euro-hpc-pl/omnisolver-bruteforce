from omnisolver.bruteforce.cpu.qubo import Qubo


def find_lowest_state_in_chunk(
    qubo: Qubo, num_fixed_bits: int, chunk_index: int
) -> int:
    """Find the lowest energy state among states with given most significant bits fixed.

    .. warning::
        This function returns a single state, even in case of degeneracy.

    :param qubo: QUBO model defining energy of states
    :param num_fixed_bits: number of fixed bits
    :param chunk_index: integer defining fixed bits in fixed part. For instance, if three bits
     are fixed and chunk index is 3, the considered states all end with [1, 1, 0].
    :returns: integer M such that N-th first bits of M describe the lowest energy state
     in the processed chunk, where N is the system size.
    """
    # current_state = np.zeros(qubo.num_variables, dtype=np.int8)
    # energy_updates = np.zeros(qubo.num_variables)

    # Flip last bits of current_state, while updating (hehe) energy_updates table

    # Enumerate states in Gray code order, each time flip a bit and update energy_updates
    # Update minimum if necessary

    # Return the minimum
    return 0  # temporary, to pass mypy
