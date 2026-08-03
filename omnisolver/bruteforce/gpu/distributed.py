import typing
from itertools import product
from time import perf_counter

import numpy as np
import ray
from dimod import BQM, Sampler, Vartype, append_variables, concatenate

from .sampler import (
    MAX_KERNEL_NUM_VARIABLES,
    BruteforceGPUSampler,
    normalize_dtype,
    validate_kernel_problem_size,
)

#: Default number of partial results combined by a single merge task. ``None`` selects the
#: direct merge, in which the controller collects every partial result and concatenates them
#: in one go.
#:
#: The direct merge is the default because it is measurably the cheaper of the two in the
#: regime this solver is normally used in. Measured for ``num_states=1`` (see
#: ``benchmarks/exp_controller_cost.py`` in the article repository), the direct merge wins up
#: to roughly 10 ** 4 subproblems, because the concatenation itself is cheap and performing it
#: in a hierarchy of remote tasks replaces a cheap local operation by several rounds of task
#: scheduling. Above that the hierarchy wins, since its depth grows only logarithmically while
#: the direct merge has to fetch every partial result: at 2 ** 16 subproblems the two cost
#: about 4.3 s and 5.9 s respectively, against a 4.8 s floor set by Ray's task scheduling
#: alone. Setting an integer also bounds the memory the controller needs to at most
#: ``merge_batch_size`` partial results at a time, which matters when ``num_states`` is large
#: or the partial results are big.
DEFAULT_MERGE_BATCH_SIZE = None


@ray.remote(num_gpus=1)
def _solve_subproblem(
    bqm,
    num_states,
    fixed_vars,
    suffix_size,
    grid_size,
    block_size,
    num_steps_per_kernel,
    partial_diff_buffer_depth,
    dtype,
):
    bqm = BQM.from_serializable(bqm)
    new_bqm = bqm.copy()
    new_bqm.fix_variables(fixed_vars)

    sampler = BruteforceGPUSampler()
    result = sampler.sample(
        new_bqm,
        num_states,
        suffix_size,
        grid_size,
        block_size,
        num_steps_per_kernel,
        partial_diff_buffer_depth,
        dtype,
    )

    return append_variables(result, fixed_vars)


@ray.remote
def _merge_partial_results(num_states, *partial_results):
    """Combine partial sample sets, keeping only the num_states lowest energy samples.

    Truncating at every level of the merge hierarchy is safe: the globally lowest
    num_states samples are necessarily contained in the union of the per-branch lowest
    num_states samples.
    """
    if len(partial_results) == 1:
        return partial_results[0]
    return concatenate(partial_results).truncate(num_states)


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class DistributedBruteforceGPUSampler(Sampler):
    def sample(
        self,
        bqm,
        num_states,
        num_fixed_vars,
        suffix_size,
        grid_size,
        block_size,
        num_steps_per_kernel=1,
        partial_diff_buffer_depth=1,
        dtype=np.float32,
        merge_batch_size=DEFAULT_MERGE_BATCH_SIZE,
    ):
        """Solve Binary Quadratic Model by distributing an exhaustive search over Ray workers.

        The search is split by fixing num_fixed_vars variables to all of their
        2 ** num_fixed_vars possible assignments. Each assignment defines an independent
        subproblem over the remaining variables, which is dispatched as a Ray task and solved
        by the GPU kernel on one worker. There is no communication between workers during the
        search; the partial results are combined afterwards by a hierarchy of merge tasks.

        :param bqm: Binary Quadratic Model instance to solve.
        :param num_states: number of lowest energy states to compute.
        :param num_fixed_vars: number k of variables to fix, giving 2 ** k independent
            subproblems over ``bqm.num_variables - k`` variables each.
        :param suffix_size: exponent l such that 2 ** l configurations are kept in the GPU
            working set at a time (see :meth:`BruteforceGPUSampler.sample`).
        :param grid_size: number of blocks for the custom kernels.
        :param block_size: number of threads per block for the custom kernels.
        :param num_steps_per_kernel: number of chunks processed by a single kernel launch.
        :param partial_diff_buffer_depth: depth of the incremental energy difference buffers.
        :param dtype: datatype to use, either np.float32 or np.float64, or one of the names
            accepted by :func:`normalize_dtype`. Forwarded unchanged to the workers.
        :param merge_batch_size: ``None`` (the default) collects every partial result on the
            controller and concatenates them in one go. An integer instead merges them in a
            hierarchy of Ray tasks, at most ``merge_batch_size`` at a time, which bounds the
            memory the controller needs at the cost of several rounds of task scheduling; see
            :data:`DEFAULT_MERGE_BATCH_SIZE` for when that trade is worth making.
        :raises ValueError: if num_fixed_vars or merge_batch_size is out of range, if the
            resulting subproblems cannot be enumerated by the kernels, or if dtype is not a
            supported precision.
        :returns: sample set containing num_states samples. Its ``info`` dictionary reports, in
            seconds, ``dispatch_time_in_seconds`` (submitting all subproblems),
            ``solve_time_in_seconds`` (dispatch plus completion of every subproblem search),
            ``merge_time_in_seconds`` (combining the partial results and fetching the final
            one) and ``total_time_in_seconds`` (the sum of the latter two), together with
            ``num_subproblems`` and ``num_merge_rounds``.
        """
        dtype = normalize_dtype(dtype)

        if not 0 <= num_fixed_vars < bqm.num_variables:
            raise ValueError(
                f"num_fixed_vars has to satisfy 0 <= num_fixed_vars < {bqm.num_variables} "
                f"(the number of variables of bqm), but got {num_fixed_vars}."
            )
        if merge_batch_size is not None and merge_batch_size < 2:
            raise ValueError(
                f"merge_batch_size has to be at least 2, or None for a direct merge, "
                f"but got {merge_batch_size}."
            )
        validate_kernel_problem_size(bqm.num_variables - num_fixed_vars, suffix_size)

        if bqm.vartype == Vartype.SPIN:
            return self.sample(
                bqm.change_vartype("BINARY", inplace=False),
                num_states=num_states,
                num_fixed_vars=num_fixed_vars,
                suffix_size=suffix_size,
                grid_size=grid_size,
                block_size=block_size,
                num_steps_per_kernel=num_steps_per_kernel,
                partial_diff_buffer_depth=partial_diff_buffer_depth,
                dtype=dtype,
                merge_batch_size=merge_batch_size,
            ).change_vartype("SPIN", inplace=False)

        bqm, mapping = bqm.relabel_variables_as_integers()

        start_counter = perf_counter()

        subproblems = [
            {i: v for i, v in enumerate(vals)} for vals in product([0, 1], repeat=num_fixed_vars)
        ]

        serialized_bqm = ray.put(bqm.to_serializable())
        refs = [
            _solve_subproblem.remote(
                serialized_bqm,
                num_states,
                fixed_vars,
                suffix_size,
                grid_size,
                block_size,
                num_steps_per_kernel,
                partial_diff_buffer_depth,
                dtype,
            )
            for fixed_vars in subproblems
        ]

        dispatch_time_in_seconds = perf_counter() - start_counter

        # Wait for the search phase to complete before merging, so that the cost of the
        # search and the cost of the controller can be reported separately.
        ray.wait(refs, num_returns=len(refs))
        solve_time_in_seconds = perf_counter() - start_counter

        merge_start_counter = perf_counter()
        num_merge_rounds = 0
        if merge_batch_size is None:
            partial_results = ray.get(refs)
            result = (
                partial_results[0]
                if len(partial_results) == 1
                else concatenate(partial_results).truncate(num_states)
            )
        else:
            while len(refs) > 1:
                refs = [
                    _merge_partial_results.remote(num_states, *batch)
                    for batch in _batched(refs, merge_batch_size)
                ]
                num_merge_rounds += 1
            result = ray.get(refs[0])
        merge_time_in_seconds = perf_counter() - merge_start_counter

        result.info.update(
            {
                "dispatch_time_in_seconds": dispatch_time_in_seconds,
                "solve_time_in_seconds": solve_time_in_seconds,
                "merge_time_in_seconds": merge_time_in_seconds,
                "total_time_in_seconds": solve_time_in_seconds + merge_time_in_seconds,
                "num_subproblems": len(subproblems),
                "num_merge_rounds": num_merge_rounds,
            }
        )
        return result.relabel_variables(mapping)

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        return {
            "num_states": [],
            "num_fixed_vars": [],
            "suffix_size": [],
            "grid_size": [],
            "block_size": [],
            "num_steps_per_kernel": [],
            "partial_diff_buffer_depth": [],
            "dtype": ["dtypes"],
            "merge_batch_size": [],
        }

    @property
    def properties(self) -> typing.Dict[str, typing.Any]:
        return {
            "dtypes": [np.float32, np.float64],
            "max_num_variables_per_worker": MAX_KERNEL_NUM_VARIABLES,
        }
