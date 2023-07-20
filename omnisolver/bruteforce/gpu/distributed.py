import typing
from itertools import product
from time import perf_counter

import numpy as np
import ray
from dimod import Sampler, Vartype, append_variables, concatenate

from .sampler import BruteforceGPUSampler


@ray.remote(num_gpus=1)
def _solve_subproblem(bqm, num_states, fixed_vars, suffix_size, grid_size, block_size, dtype):
    new_bqm = bqm.copy()
    new_bqm.fix_variables(fixed_vars)

    sampler = BruteforceGPUSampler()
    result = sampler.sample(new_bqm, num_states, suffix_size, grid_size, block_size, dtype)

    return append_variables(result, fixed_vars)


class DistributedBruteforceGPUSampler(Sampler):
    def sample(
        self, bqm, num_states, num_fixed_vars, suffix_size, grid_size, block_size, dtype=np.float32
    ):
        if bqm.vartype == Vartype.SPIN:
            return self.sample(
                bqm.change_vartype("BINARY", inplace=False),
                num_states,
                num_fixed_vars,
                suffix_size,
                grid_size,
                block_size,
            ).change_vartype("SPIN", inplace=False)

        bqm, mapping = bqm.relabel_variables_as_integers()

        start_counter = perf_counter()

        subproblems = [
            {i: v for i, v in enumerate(vals)} for vals in product([0, 1], repeat=num_fixed_vars)
        ]

        refs = [
            _solve_subproblem.remote(
                bqm, num_states, fixed_vars, suffix_size, grid_size, block_size, dtype
            )
            for fixed_vars in subproblems
        ]

        solve_time_in_seconds = perf_counter() - start_counter

        subsolutions = [ray.get(ref) for ref in refs]
        result = concatenate(subsolutions).truncate(num_states)
        result.info["solve_time_in_seconds"] = solve_time_in_seconds
        return result

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
