from dimod import BQM
from numpy.random import default_rng

from omnisolver.bruteforce.gpu import BruteforceGPUSampler
from omnisolver.bruteforce.gpu.distributed import DistributedBruteforceGPUSampler


def random_bqm(num_variables, vartype, offset, rng):
    linear = {
        i: coef for i, coef in zip(range(num_variables), rng.uniform(-2, 2, size=num_variables))
    }
    quadratic = {
        (i, j): coef
        for (i, j), coef in zip(
            [(i, j) for i in range(num_variables) for j in range(i + 1, num_variables)],
            rng.uniform(-1, -1, size=(num_variables - 1) * num_variables // 2),
        )
    }
    return BQM(linear, quadratic, offset, vartype=vartype)


NUM_VARIABLES = 40


def main():
    sampler = DistributedBruteforceGPUSampler()
    sampler2 = BruteforceGPUSampler()
    rng = default_rng(1234)

    bqm = random_bqm(NUM_VARIABLES, "BINARY", 0.0, rng)

    import time

    start = time.perf_counter()
    result = sampler.sample(
        bqm, num_states=100, num_fixed_vars=1, suffix_size=25, grid_size=1024, block_size=1024
    )
    duration = time.perf_counter() - start
    distributed_en = [entry.energy for entry in result.data()]

    print(f"Distributed finished in : {duration}s")

    start = time.perf_counter()
    result2 = sampler2.sample(
        bqm, num_states=100, suffix_size=25, grid_size=2**12, block_size=512
    )
    duration = time.perf_counter() - start
    single_en = [entry.energy for entry in result2.data()]
    print(f"Single finished in: {duration}s")

    print(max(abs(en1 - en2) for en1, en2 in zip(distributed_en, single_en)))


if __name__ == "__main__":
    main()
