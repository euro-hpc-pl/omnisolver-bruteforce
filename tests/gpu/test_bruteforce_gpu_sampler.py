import numpy as np
import pytest
from dimod import BQM

from omnisolver.bruteforce.gpu import BruteforceGPUSampler


def random_bqm(num_variables, vartype, offset, rng):
    linear = {i: rng.uniform(-2, 2) for i in range(num_variables)}
    quadratic = {
        (i, j): rng.uniform(-1, 1)
        for i in range(num_variables)
        for j in range(i + 1, num_variables)
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
@pytest.mark.parametrize("num_states", [100, 1000])
@pytest.mark.parametrize("suffix_size", [22, 24, 26])
@pytest.mark.parametrize("grid_size", [2**10, 2**11])
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("dtype", [np.float32])
def test_samples_returned_by_sampler_have_correct_energies(
    bqm, num_states, suffix_size, grid_size, block_size, dtype
):
    sampler = BruteforceGPUSampler()
    result = sampler.sample(bqm, num_states, suffix_size, grid_size, block_size, dtype)

    assert all(
        bqm.energy(entry.sample) == pytest.approx(entry.energy, abs=1e-3) for entry in result.data()
    )
