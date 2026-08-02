# SPDX-FileCopyrightText: 2021-2026 The Omnisolver developers
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the stabilized single-precision ground-state path.

The compensated updates and the periodic exact re-anchoring of the ground-state kernel are
only enabled for ``np.float32`` and for problems of at least 40 variables, so none of the
tests operating on small instances exercise them. The tests below therefore run a full
exhaustive search at N = 40; each run takes a few seconds on a recent data-centre or
high-end consumer GPU and is marked ``slow``.
"""

import numpy as np
import pytest
from dimod import BQM
from numba import cuda

from omnisolver.bruteforce.gpu import BruteforceGPUSampler

pytestmark = [
    pytest.mark.skipif(not cuda.is_available(), reason="CUDA-compatible GPU is required"),
    pytest.mark.slow,
]

#: Smallest size for which the kernel enables the stabilization mechanisms.
STABILIZATION_THRESHOLD = 40

#: Launch geometry small enough to fit any CUDA-capable device used for testing while still
#: enumerating 2 ** 40 configurations in a few seconds.
KERNEL_ARGS = {
    "suffix_size": 22,
    "grid_size": 4096,
    "block_size": 512,
    "partial_diff_buffer_depth": 2,
}


def random_bqm(num_variables, rng):
    linear = {i: float(coef) for i, coef in enumerate(rng.uniform(-2, 2, size=num_variables))}
    quadratic = {
        (i, j): float(coef)
        for (i, j), coef in zip(
            [(i, j) for i in range(num_variables) for j in range(i + 1, num_variables)],
            rng.uniform(-1, 1, size=(num_variables - 1) * num_variables // 2),
        )
    }
    return BQM(linear, quadratic, 0.0, vartype="BINARY")


@pytest.fixture(scope="module")
def bqm():
    return random_bqm(STABILIZATION_THRESHOLD, np.random.default_rng(1234))


@pytest.mark.parametrize("num_steps_per_kernel", [4096, 512])
def test_reported_ground_state_energy_does_not_drift_from_exact_recomputation(
    bqm, num_steps_per_kernel
):
    """The energy reported by the float32 fast path must survive 2 ** 40 incremental updates.

    Without the compensated updates and the periodic re-anchoring, the incrementally
    accumulated energy drifts away from the true energy of the returned configuration; the
    difference between the reported energy and a from-scratch float64 recomputation is what
    detects that.
    """
    result = BruteforceGPUSampler().sample(
        bqm,
        num_states=1,
        num_steps_per_kernel=num_steps_per_kernel,
        dtype=np.float32,
        **KERNEL_ARGS,
    )

    reported_energy = result.first.energy
    recomputed_energy = bqm.energy(result.first.sample)
    assert reported_energy == pytest.approx(recomputed_energy, abs=1e-4)


def test_ground_state_is_independent_of_the_re_anchoring_cadence(bqm):
    """Changing num_steps_per_kernel changes when re-anchoring happens, but not the answer."""
    sampler = BruteforceGPUSampler()
    results = [
        sampler.sample(
            bqm,
            num_states=1,
            num_steps_per_kernel=num_steps_per_kernel,
            dtype=np.float32,
            **KERNEL_ARGS,
        ).first
        for num_steps_per_kernel in (4096, 512)
    ]

    assert results[0].sample == results[1].sample
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-5)
