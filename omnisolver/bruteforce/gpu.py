import typing

import numpy as np
from dimod import Sampler, SampleSet, Vartype

from omnisolver.bruteforce.ext.gpu import gpu_search


def _convert_int_to_sample(val, num_variables):
    sample = {}
    for i in range(num_variables):
        sample[i] = val % 2
        val //= 2
    return sample


class BruteforceGPUSampler(Sampler):
    def sample(self, bqm, num_states, suffix_size, grid_size, block_size, dtype=np.float32):
        if bqm.vartype == Vartype.SPIN:
            return self.sample(
                bqm.change_vartype("BINARY", inplace=False),
                num_states,
                suffix_size,
                grid_size,
                block_size,
            ).change_vartype("SPIN", inplace=False)

        bqm, mapping = bqm.relabel_variables_as_integers()

        qubo_mat = np.zeros((bqm.num_variables, bqm.num_variables), dtype=dtype)

        for (i, j), coef in bqm.quadratic.items():
            qubo_mat[i, j] += coef
            qubo_mat[j, i] += coef

        for i, coef in bqm.linear.items():
            qubo_mat[i, i] = coef

        states_out = np.zeros(num_states, dtype=np.uint64)
        energies_out = np.zeros(num_states, dtype=dtype)

        gpu_search(
            qubo_mat,
            num_states,
            states_out,
            energies_out,
            grid_size,
            block_size,
            suffix_size,
        )

        samples = [_convert_int_to_sample(state, bqm.num_variables) for state in states_out]

        result = SampleSet.from_samples(samples, bqm.vartype, energies_out + bqm.offset)

        return result.relabel_variables(
            {value: key for key, value in mapping.items()}, inplace=False
        )

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        return {
            "num_states": [],
            "suffix_size": [],
            "grid_size": [],
            "block_size": [],
            "dtype": [],
        }

    @property
    def properties(self) -> typing.Dict[str, typing.Any]:
        return {}
