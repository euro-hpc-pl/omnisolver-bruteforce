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

#: Number of partial results combined by a single merge task when the hierarchical strategy is
#: used. Eight is the value the shipped measurements were taken with.
DEFAULT_MERGE_BATCH_SIZE = 8

#: Number of subproblems above which the hierarchical merge becomes cheaper than collecting
#: every partial result on the controller and concatenating once.
#:
#: Measured with ``num_states=1`` (see ``benchmarks/exp_controller_cost.py`` in the article
#: repository): the direct merge grows as O(P^0.97) and overtakes the hierarchy below this
#: threshold, where the concatenation is cheap enough that performing it in remote tasks only
#: adds rounds of scheduling; above it the hierarchy's logarithmic depth wins, reaching 3.9 s
#: against 7.4 s at 2 ** 16 subproblems. Either side of the threshold the two differ by under a
#: millisecond up to a few hundred subproblems, so the choice only matters at large k. The
#: hierarchy additionally bounds the memory the controller needs, which is why an explicit
#: ``merge_batch_size`` always selects it.
MERGE_TREE_THRESHOLD = 64


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
        # A simple slice keeps black and flake8 (E203) from disagreeing about the colon.
        stop = start + batch_size
        yield items[start:stop]


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
        merge_batch_size=None,
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
        :param merge_batch_size: ``None`` (the default) picks a merge strategy from the number
            of subproblems, using :data:`MERGE_TREE_THRESHOLD`: at or below the threshold every
            partial result is collected on the controller and concatenated in one go, above it
            they are reduced by a hierarchy of Ray tasks combining
            :data:`DEFAULT_MERGE_BATCH_SIZE` of them at a time. An integer forces the hierarchy
            with that batch size, which also bounds the memory the controller needs; a value of
            at least ``2 ** num_fixed_vars`` reproduces the single-shot merge of release 0.0.5.
        :raises ValueError: if num_fixed_vars or merge_batch_size is out of range, if the
            resulting subproblems cannot be enumerated by the kernels, or if dtype is not a
            supported precision.
        :returns: sample set containing num_states samples. Its ``info`` dictionary reports, in
            seconds, ``dispatch_time_in_seconds`` (submitting all subproblems),
            ``solve_time_in_seconds`` (dispatch plus completion of every subproblem search),
            ``merge_time_in_seconds`` (combining the partial results and fetching the final
            one) and ``total_time_in_seconds`` (the sum of the latter two), together with
            ``num_subproblems``, ``num_merge_rounds`` and ``merge_strategy``.
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
        use_hierarchy = merge_batch_size is not None or len(refs) > MERGE_TREE_THRESHOLD
        batch_size = merge_batch_size if merge_batch_size is not None else DEFAULT_MERGE_BATCH_SIZE
        if not use_hierarchy:
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
                    for batch in _batched(refs, batch_size)
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
                "merge_strategy": "hierarchical" if use_hierarchy else "direct",
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
