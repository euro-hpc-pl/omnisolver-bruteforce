
![Logo](https://raw.githubusercontent.com/euro-hpc-pl/omnisolver/master/logo.png)
*Bruteforce (a.k.a. exhaustive search) Plugin for [Omnisolver](https://github.com/euro-hpc-pl/omnisolver)*

Solve Ising and QUBO instances by exhaustive search on CUDA-enabled GPUs, either on a single
device or distributed over many GPUs and hosts with [Ray](https://www.ray.io/). Because every
one of the `2 ** N` configurations is enumerated, the returned optimum is *certified*, which
makes the plugin useful as a ground-truth reference for heuristic and quantum-inspired
solvers.

**Documentation:** <https://euro-hpc-pl.github.io/omnisolver-bruteforce>

## Installation

The `omnisolver-bruteforce` package requires a working CUDA installation. It is distributed
as a source archive only — there are no wheels — so the CUDA extension is compiled at
install time against the toolkit found on the target machine.

First set the `CUDAHOME` environment variable to your CUDA installation location, e.g.:

```shell
# Remember, your actual location may vary!
export CUDAHOME=/usr/local/cuda
```

and then run:

```shell
# Single-GPU sampler only
pip install omnisolver-bruteforce

# Include ray, required by the distributed sampler
pip install "omnisolver-bruteforce[distributed]"
```

During build, this package will:
- Prefer `CUDAHOME/bin/nvcc` (if `CUDAHOME` is set), so multi-CUDA environments are deterministic.
- Add NVCC flag `-allow-unsupported-compiler` via `NVCC_PREPEND_FLAGS` to reduce host compiler compatibility build failures.

> **Warning**
> If you don't set the `CUDAHOME` directory, an attempt will be made to deduce it based on the location of your `nvcc` compiler.
> However, this process might not work in all the cases and should not be relied on.

Supported versions: Python 3.10 and 3.11; CUDA Toolkit 12.4 and 12.5 are exercised in
continuous integration.

### Troubleshooting (multiple CUDA toolchains)

If your system has multiple CUDA toolchains (for example HPC SDK and system CUDA), set:

```shell
export CUDAHOME=/usr/local/cuda
```

before installing so build uses `CUDAHOME/bin/nvcc`.

## Command line usage

```text
usage: omnisolver bruteforce-gpu [-h] [--output OUTPUT] [--vartype {SPIN,BINARY}] [--num_states NUM_STATES]
                                 [--suffix_size SUFFIX_SIZE] [--grid_size GRID_SIZE] [--block_size BLOCK_SIZE]
                                 [--num_steps_per_kernel NUM_STEPS_PER_KERNEL]
                                 [--partial_diff_buffer_depth PARTIAL_DIFF_BUFFER_DEPTH] [--dtype {float,double}]
                                 input

Bruteforce (a.k.a. exhaustive search) sampler using a CUDA-enabled GPU

positional arguments:
  input                 Path of the input BQM file in COO format. If not specified, stdin is used.

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Path of the output file. If not specified, stdout is used.
  --vartype {SPIN,BINARY}
                        Variable type
  --num_states NUM_STATES
                        Size of the low energy spectrum to compute. A value of 1 selects the faster
                        ground-state-only code path
  --suffix_size SUFFIX_SIZE
                        Number of suffix bits enumerated in the resident chunk, i.e. 2 ** suffix_size
                        configurations are kept in the GPU working set at a time
  --grid_size GRID_SIZE
                        Number of blocks in grid running bruteforce kernels
  --block_size BLOCK_SIZE
                        Number of threads in each block running bruteforce kernels
  --num_steps_per_kernel NUM_STEPS_PER_KERNEL
                        Number of chunks processed by a single kernel launch (ground-state-only path)
  --partial_diff_buffer_depth PARTIAL_DIFF_BUFFER_DEPTH
                        Depth of the incremental energy difference buffers (ground-state-only path)
  --dtype {float,double}
                        Data type to use: 'float' for single precision (default, enables the stabilized
                        fast path) or 'double' for double precision
```

## Python usage

```python
import numpy as np
from dimod.serialization import coo
from omnisolver.bruteforce.gpu import BruteforceGPUSampler

with open("instance.txt") as fd:
    bqm = coo.load(fd, vartype="SPIN")

result = BruteforceGPUSampler().sample(
    bqm, num_states=1, suffix_size=23, grid_size=8192, block_size=1024,
    num_steps_per_kernel=8192, partial_diff_buffer_depth=10, dtype=np.float32,
)
print(result.first.energy, result.info["solve_time_in_seconds"])
```

For the multi-GPU sampler, start Ray on the participating nodes and use
`DistributedBruteforceGPUSampler` with the additional `num_fixed_vars` parameter; see the
[user guide](https://euro-hpc-pl.github.io/omnisolver-bruteforce) and `examples/distributed.py`.

## Development

```shell
pip install -e ".[distributed]"
pip install pytest
pytest tests -m "not slow"   # drop the marker filter to also run the N >= 40 searches
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citing

If you used the Omnisolver package or one of its plugins, please cite:

```text
@article{omnisolver2023,
    title = {Omnisolver: An extensible interface to Ising spin–glass and QUBO solvers},
    journal = {SoftwareX},
    volume = {24},
    pages = {101559},
    year = {2023},
    doi = {10.1016/j.softx.2023.101559},
    author = {Konrad Jałowiecki and {\L}ukasz Pawela},
}
```

The CUDA kernel underlying this plugin was introduced in:

```text
@article{jalowiecki2021brute,
    title = {Brute-forcing spin-glass problems with CUDA},
    journal = {Computer Physics Communications},
    volume = {260},
    pages = {107728},
    year = {2021},
    doi = {10.1016/j.cpc.2020.107728},
    author = {Konrad Jałowiecki and Marek M. Rams and Bart{\l}omiej Gardas},
}
```
