# SPDX-FileCopyrightText: 2021-2026 The Omnisolver developers
#
# SPDX-License-Identifier: Apache-2.0

import typing
from time import perf_counter

import numpy as np
from dimod import Sampler, SampleSet, Vartype

from omnisolver.bruteforce.ext.gpu import gpu_search, gpu_search_ground_only

#: Widest configuration the kernels can enumerate, set by the 64-bit state word.
MAX_KERNEL_NUM_VARIABLES = 64

#: Names accepted by :func:`normalize_dtype` in addition to NumPy dtypes.  The CLI passes
#: ``"float"``/``"double"`` as plain strings; without this mapping NumPy would resolve both
#: of them to ``float64``, silently disabling the single-precision fast path.
_DTYPE_ALIASES = {
    "float": np.float32,
    "float32": np.float32,
    "single": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


def normalize_dtype(dtype) -> type:
    """Resolve a user-supplied dtype specification to ``np.float32`` or ``np.float64``.

    :param dtype: either one of ``np.float32``/``np.float64`` (or an equivalent NumPy dtype
        object), or one of the names ``"float"``, ``"float32"``, ``"single"``, ``"double"``,
        ``"float64"``. Note that ``"float"`` means *single* precision here, matching the
        ``--dtype`` choices of the command line interface, and deliberately not NumPy's
        interpretation of the same string.
    :raises ValueError: if the specification does not denote a supported precision.
    :returns: ``np.float32`` or ``np.float64``.
    """
    if isinstance(dtype, str):
        try:
            return _DTYPE_ALIASES[dtype.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Expected one of "
                f"{sorted(_DTYPE_ALIASES)} or np.float32/np.float64."
            ) from None
    resolved = np.dtype(dtype).type
    if resolved not in (np.float32, np.float64):
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected np.float32 or np.float64.")
    return resolved


def validate_kernel_problem_size(num_variables: int, suffix_size: int) -> None:
    """Check that a problem of given size can be enumerated by the CUDA kernels.

    The kernels keep each configuration in a single 64-bit word and iterate over
    ``2 ** (num_variables - suffix_size)`` chunks of ``2 ** suffix_size`` configurations,
    so both bounds have to hold before any kernel is launched. Violating them would
    otherwise produce silently truncated or wrong results rather than an error.

    :param num_variables: number of variables handed to the kernel.
    :param suffix_size: number of variables forming the resident enumeration chunk.
    :raises ValueError: if the problem is too wide, or the suffix does not fit in it.
    """
    if num_variables > MAX_KERNEL_NUM_VARIABLES:
        raise ValueError(
            f"The GPU kernels hold each configuration in a {MAX_KERNEL_NUM_VARIABLES}-bit word, "
            f"so at most {MAX_KERNEL_NUM_VARIABLES} variables can be enumerated in one kernel, "
            f"but got {num_variables}. Use DistributedBruteforceGPUSampler with "
            f"num_fixed_vars >= {num_variables - MAX_KERNEL_NUM_VARIABLES} to split the problem."
        )
    if not 0 < suffix_size <= num_variables:
        raise ValueError(
            f"suffix_size has to satisfy 0 < suffix_size <= {num_variables} "
            f"(the number of variables seen by the kernel), but got {suffix_size}."
        )


def _convert_int_to_sample(val, num_variables):
    sample = {}
    for i in range(num_variables):
        sample[i] = val % 2
        val //= 2
    return sample


class BruteforceGPUSampler(Sampler):
    def sample(
        self,
        bqm,
        num_states,
        suffix_size,
        grid_size,
        block_size,
        num_steps_per_kernel=16,
        partial_diff_buffer_depth=1,
        dtype=np.float32,
    ):
        """Solve Binary Quadratic Model using exhaustive (bruteforce) search on the GPU.

        :param bqm: Binary Quadratic Model instance to solve.
        :param num_states: number of lowest energy states to compute.
        :param suffix_size: exponent l such that 2 ** l is the number of configurations kept
            in the GPU working set at a time; the search sweeps 2 ** (N - l) such chunks,
            where N is the number of variables of bqm.
        :param grid_size: number of blocks for the custom kernels. Note that this parameter
            does not affect the grid on which the Thrust kernels are launched.
        :param block_size: number of threads per block for custom kernels. Note that this
            parameter does not affect the grid on which the Thrust kernels are launched.
        :param num_steps_per_kernel: number of chunks processed by a single kernel launch.
        :param partial_diff_buffer_depth: depth of the incremental energy difference buffers.
        :param dtype: datatype to use, either np.float32 or np.float64, or one of the names
            accepted by :func:`normalize_dtype` (in particular `"float"` means *single*
            precision). The default is np.float32, which on most GPUs is significantly faster
            than 64-bit floating point numbers. For `num_states == 1`, the GPU backend
            automatically enables compensated updates and periodic numeric re-anchoring when
            using np.float32 on large models (at least 40 variables) to remove drift from long
            chains of incremental energy updates. Note that np.float64 is therefore a different
            numerical path, not merely a slower one.
        :raises ValueError: if dtype is not a supported precision, or if the problem size and
            suffix_size are not enumerable by the kernels (see
            :func:`validate_kernel_problem_size`).
        :returns: sample set containing num_states samples.
        """
        dtype = normalize_dtype(dtype)
        validate_kernel_problem_size(bqm.num_variables, suffix_size)

        if bqm.vartype == Vartype.SPIN:
            return self.sample(
                bqm.change_vartype("BINARY", inplace=False),
                num_states=num_states,
                suffix_size=suffix_size,
                grid_size=grid_size,
                block_size=block_size,
                num_steps_per_kernel=num_steps_per_kernel,
                partial_diff_buffer_depth=partial_diff_buffer_depth,
                dtype=dtype,
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

        start_counter = perf_counter()

        if num_states == 1:  # Shortcut if we are only looking for a ground state
            gpu_search_ground_only(
                qubo_mat,
                states_out,
                energies_out,
                grid_size,
                block_size,
                suffix_size,
                num_steps_per_kernel,
                partial_diff_buffer_depth,
            )
        else:
            gpu_search(
                qubo_mat,
                num_states,
                states_out,
                energies_out,
                grid_size,
                block_size,
                suffix_size,
            )

        solve_time_in_seconds = perf_counter() - start_counter

        samples = [_convert_int_to_sample(state, bqm.num_variables) for state in states_out]

        result = SampleSet.from_samples(
            samples,
            bqm.vartype,
            energies_out + bqm.offset,
            info={"solve_time_in_seconds": solve_time_in_seconds},
        )

        return result.relabel_variables(mapping, inplace=False)

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        return {
            "num_states": [],
            "suffix_size": [],
            "grid_size": [],
            "block_size": [],
            "num_steps_per_kernel": [],
            "partial_diff_buffer_depth": [],
            "dtype": ["dtypes"],
        }

    @property
    def properties(self) -> typing.Dict[str, typing.Any]:
        return {
            "dtypes": [np.float32, np.float64],
            "max_num_variables": MAX_KERNEL_NUM_VARIABLES,
        }
