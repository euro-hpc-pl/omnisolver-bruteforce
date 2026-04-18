import os

import numpy as np
import pytest
from dimod import BQM, ExactSolver
from numba import cuda

from omnisolver.bruteforce.gpu import BruteforceGPUSampler


pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA-compatible GPU is required")


def random_bqm(
    num_variables,
    vartype,
    offset,
    rng,
    linear_range=(-2, 2),
    quadratic_range=(-1, 1),
):
    linear_low, linear_high = linear_range
    quad_low, quad_high = quadratic_range
    linear = {
        i: coef
        for i, coef in zip(range(num_variables), rng.uniform(linear_low, linear_high, size=num_variables))
    }
    quadratic = {
        (i, j): coef
        for (i, j), coef in zip(
            [(i, j) for i in range(num_variables) for j in range(i + 1, num_variables)],
            rng.uniform(quad_low, quad_high, size=(num_variables - 1) * num_variables // 2),
        )
    }
    return BQM(linear, quadratic, offset, vartype=vartype)


def create_bqms():
    rng = np.random.default_rng(1234)
    return [
        random_bqm(num_variables, vartype, offset, rng)
        for num_variables in [26, 28, 30]
        for vartype in ["SPIN", "BINARY"]
        for offset in [0, -5, 2.5]
    ]


@pytest.mark.parametrize("bqm", create_bqms())
@pytest.mark.parametrize("num_states", [100, 500])
@pytest.mark.parametrize("suffix_size", [21, 22, 24])
@pytest.mark.parametrize("grid_size", [2**10, 2**11])
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("dtype", [np.float32])
def test_samples_returned_by_sampler_have_correct_energies(
    bqm, num_states, suffix_size, grid_size, block_size, dtype
):
    sampler = BruteforceGPUSampler()
    result = sampler.sample(
        bqm,
        num_states,
        suffix_size,
        grid_size,
        block_size,
        dtype=dtype,
    )

    assert all(
        bqm.energy(entry.sample) == pytest.approx(entry.energy, abs=1e-3) for entry in result.data()
    )


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_ground_only_energy_matches_cpu_exact_recompute(seed):
    rng = np.random.default_rng(seed)
    bqm = random_bqm(
        20,
        "BINARY",
        0.0,
        rng,
        linear_range=(-4, 4),
        quadratic_range=(-2, 2),
    )

    sampler = BruteforceGPUSampler()
    result = sampler.sample(
        bqm,
        num_states=1,
        suffix_size=12,
        grid_size=2**8,
        block_size=128,
        num_steps_per_kernel=32,
        partial_diff_buffer_depth=2,
        dtype=np.float32,
    )

    first = result.first
    assert bqm.energy(first.sample) == pytest.approx(first.energy, abs=5e-4)


def test_ground_only_matches_exact_solver_on_small_problem():
    rng = np.random.default_rng(2026)
    bqm = random_bqm(
        18,
        "BINARY",
        0.25,
        rng,
        linear_range=(-3, 3),
        quadratic_range=(-1.5, 1.5),
    )

    sampler = BruteforceGPUSampler()
    gpu_result = sampler.sample(
        bqm,
        num_states=1,
        suffix_size=11,
        grid_size=2**8,
        block_size=128,
        num_steps_per_kernel=32,
        partial_diff_buffer_depth=2,
        dtype=np.float32,
    )
    gpu_first = gpu_result.first

    exact_result = ExactSolver().sample(bqm)
    exact_first = exact_result.first

    assert gpu_first.energy == pytest.approx(exact_first.energy, abs=5e-4)
    assert bqm.energy(gpu_first.sample) == pytest.approx(exact_first.energy, abs=5e-4)


@pytest.mark.skipif(
    os.getenv("OMNISOLVER_RUN_PERF_GUARD") != "1",
    reason="Performance guard is opt-in (set OMNISOLVER_RUN_PERF_GUARD=1).",
)
def test_ground_only_perf_guard_representative_workload():
    rng = np.random.default_rng(4242)
    bqm = random_bqm(
        24,
        "BINARY",
        0.0,
        rng,
        linear_range=(-2, 2),
        quadratic_range=(-1, 1),
    )
    max_seconds = float(os.getenv("OMNISOLVER_PERF_MAX_SECONDS", "10.0"))

    sampler = BruteforceGPUSampler()
    result = sampler.sample(
        bqm,
        num_states=1,
        suffix_size=14,
        grid_size=2**9,
        block_size=128,
        num_steps_per_kernel=64,
        partial_diff_buffer_depth=2,
        dtype=np.float32,
    )

    assert result.info["solve_time_in_seconds"] <= max_seconds
