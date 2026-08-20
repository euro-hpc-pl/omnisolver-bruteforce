# SPDX-FileCopyrightText: 2021-2026 The Omnisolver developers
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Ray-based distributed sampler.

These tests deliberately use instances small enough for :class:`dimod.ExactSolver`, so that
the fixed-variable decomposition, the hierarchical merge and the forwarding of solver
arguments to the workers can be checked against a known-correct answer.
"""

import numpy as np
import pytest
from dimod import BQM, ExactSolver
from numba import cuda

ray = pytest.importorskip("ray", reason="ray is required by the distributed sampler")

from omnisolver.bruteforce.gpu.distributed import (  # noqa: E402  (after importorskip)
    DistributedBruteforceGPUSampler,
)

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA-compatible GPU is required")

KERNEL_ARGS = {
    "suffix_size": 8,
    "grid_size": 64,
    "block_size": 64,
    "num_steps_per_kernel": 8,
    "partial_diff_buffer_depth": 2,
}


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def random_bqm(num_variables, vartype, rng):
    linear = {i: float(coef) for i, coef in enumerate(rng.uniform(-2, 2, size=num_variables))}
    quadratic = {
        (i, j): float(coef)
        for (i, j), coef in zip(
            [(i, j) for i in range(num_variables) for j in range(i + 1, num_variables)],
            rng.uniform(-1, 1, size=(num_variables - 1) * num_variables // 2),
        )
    }
    return BQM(linear, quadratic, 0.0, vartype=vartype)


@pytest.mark.parametrize("num_fixed_vars", [0, 1, 2])
def test_ground_state_matches_exact_solver(num_fixed_vars):
    bqm = random_bqm(16, "BINARY", np.random.default_rng(2026))

    result = DistributedBruteforceGPUSampler().sample(
        bqm, num_states=1, num_fixed_vars=num_fixed_vars, **KERNEL_ARGS
    )
    expected = ExactSolver().sample(bqm).first

    assert result.first.energy == pytest.approx(expected.energy, abs=5e-4)
    assert bqm.energy(result.first.sample) == pytest.approx(expected.energy, abs=5e-4)


@pytest.mark.parametrize("merge_batch_size", [2, 4, 16])
def test_low_energy_spectrum_matches_exact_solver_for_any_merge_hierarchy(merge_batch_size):
    """The hierarchical merge must return the globally lowest states, not per-branch ones.

    With merge_batch_size below the number of subproblems the partial results are combined in
    several rounds, which is the regime that a merge dropping candidates would break.
    """
    num_states = 8
    bqm = random_bqm(14, "BINARY", np.random.default_rng(7))

    result = DistributedBruteforceGPUSampler().sample(
        bqm,
        num_states=num_states,
        num_fixed_vars=3,
        merge_batch_size=merge_batch_size,
        **KERNEL_ARGS,
    )
    expected = np.sort(ExactSolver().sample(bqm).record.energy)[:num_states]

    assert len(result) == num_states
    assert np.sort(result.record.energy) == pytest.approx(expected, abs=5e-4)


def test_dtype_is_forwarded_to_the_workers_for_spin_models():
    """Regression test: the SPIN -> BINARY recursion used to drop the requested dtype.

    A float64 search resolves the energy far more tightly than float32 can, so a tolerance
    that only double precision can meet detects the argument being silently dropped.
    """
    bqm = random_bqm(14, "SPIN", np.random.default_rng(11))

    result = DistributedBruteforceGPUSampler().sample(
        bqm, num_states=1, num_fixed_vars=1, dtype=np.float64, **KERNEL_ARGS
    )
    expected = ExactSolver().sample(bqm).first

    assert result.first.energy == pytest.approx(expected.energy, abs=1e-9)


def test_reported_timings_separate_the_search_from_the_merge():
    bqm = random_bqm(14, "BINARY", np.random.default_rng(13))

    result = DistributedBruteforceGPUSampler().sample(
        bqm, num_states=4, num_fixed_vars=2, merge_batch_size=2, **KERNEL_ARGS
    )

    info = result.info
    assert info["num_subproblems"] == 4
    assert info["num_merge_rounds"] == 2  # 4 -> 2 -> 1 with merge_batch_size=2
    assert 0 < info["dispatch_time_in_seconds"] <= info["solve_time_in_seconds"]
    assert info["merge_time_in_seconds"] > 0
    assert info["total_time_in_seconds"] == pytest.approx(
        info["solve_time_in_seconds"] + info["merge_time_in_seconds"]
    )


class TestArgumentValidation:
    """Invalid configurations have to raise rather than return a silently wrong answer."""

    def test_rejects_subproblems_wider_than_the_64_bit_state_word(self):
        bqm = random_bqm(70, "BINARY", np.random.default_rng(17))

        with pytest.raises(ValueError, match="64-bit word"):
            DistributedBruteforceGPUSampler().sample(
                bqm, num_states=1, num_fixed_vars=2, **KERNEL_ARGS
            )

    @pytest.mark.parametrize("num_fixed_vars", [-1, 14])
    def test_rejects_out_of_range_num_fixed_vars(self, num_fixed_vars):
        bqm = random_bqm(14, "BINARY", np.random.default_rng(19))

        with pytest.raises(ValueError, match="num_fixed_vars"):
            DistributedBruteforceGPUSampler().sample(
                bqm, num_states=1, num_fixed_vars=num_fixed_vars, **KERNEL_ARGS
            )

    def test_rejects_unsupported_dtype(self):
        bqm = random_bqm(14, "BINARY", np.random.default_rng(23))

        with pytest.raises(ValueError, match="Unsupported dtype"):
            DistributedBruteforceGPUSampler().sample(
                bqm, num_states=1, num_fixed_vars=1, dtype=np.int32, **KERNEL_ARGS
            )
