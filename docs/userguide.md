# User guide

Throughout this guide, $N$ is the number of binary variables of the instance (equivalently,
the number of Ising spins), so an exhaustive search enumerates all $2^N$ configurations, and
$k$ is the number of variables the distributed sampler freezes to split the work.

## Installation

The package is distributed as a **source archive only**: there are no wheels, because the
CUDA extension is compiled against the toolkit present on the target machine. A working
CUDA installation is therefore required at install time.

```shell
# Make the build deterministic on machines with several toolchains
export CUDAHOME=/usr/local/cuda

# Single-GPU sampler only
pip install omnisolver-bruteforce

# Include ray, required by the distributed sampler
pip install "omnisolver-bruteforce[distributed]"
```

If `CUDAHOME` is not set, the build tries to deduce it from the location of `nvcc`. That
heuristic is not reliable on systems carrying both a vendor HPC SDK and a system CUDA, so
set the variable explicitly. The build also adds `-allow-unsupported-compiler` through
`NVCC_PREPEND_FLAGS`, which relaxes the host-compiler version check.

Supported versions: Python 3.10 and 3.11; CUDA Toolkit 12.4 and 12.5 are exercised in
continuous integration.

## Single-GPU usage

```python
import numpy as np
from dimod.serialization import coo
from omnisolver.bruteforce.gpu import BruteforceGPUSampler

with open("instance.txt") as fd:
    bqm = coo.load(fd, vartype="SPIN")

result = BruteforceGPUSampler().sample(
    bqm,
    num_states=1,
    suffix_size=23,
    grid_size=8192,
    block_size=1024,
    num_steps_per_kernel=8192,
    partial_diff_buffer_depth=10,
    dtype=np.float32,
)
print(result.first.energy, result.info["solve_time_in_seconds"])
```

The same sampler is reachable from the command line, where Omnisolver registers it as
`bruteforce-gpu`:

```shell
omnisolver bruteforce-gpu --vartype SPIN --num_states 1 --suffix_size 23 instance.txt
```

## Multi-GPU usage

Start Ray on the head and worker nodes, telling each how many GPUs it owns:

```shell
# Head node
ray stop --force
ray start --head --port=6379 --num-gpus=4

# Every worker node
ray stop --force
ray start --address='<HEAD_NODE_IP>:6379' --num-gpus=4
```

Then use the distributed sampler, whose only additional parameter is `num_fixed_vars`:

```python
from omnisolver.bruteforce.gpu.distributed import DistributedBruteforceGPUSampler

result = DistributedBruteforceGPUSampler().sample(
    bqm,
    num_states=1,
    num_fixed_vars=3,      # 2 ** 3 = 8 independent subproblems
    suffix_size=23,
    grid_size=8192,
    block_size=1024,
    num_steps_per_kernel=8192,
    partial_diff_buffer_depth=10,
)
```

Each subproblem is a Ray task requesting one GPU, so with fewer GPUs than subproblems Ray
simply queues them. This is worth remembering when benchmarking: the number of subproblems
is set by `num_fixed_vars` and is completely independent of how many devices are available.

## Choosing the parameters

`num_states`
: Size of the low-energy spectrum to return. `num_states=1` selects a separate, considerably
  faster code path that tracks only the ground state, and is the right choice whenever a
  certified optimum is all that is needed.

`num_fixed_vars` ($k$)
: Number of frozen variables, giving $2^k$ subproblems of $N-k$ variables. Set it so that
  every worker gets at least one subproblem — typically $2^k$ equal to the number of GPUs.
  Raising $k$ by one halves the work per subproblem and doubles the number of results to
  merge, so it should not be increased past the point where the merge starts to matter.

`suffix_size`
: Number of variables forming the resident enumeration chunk: $2^{\texttt{suffix\_size}}$
  configurations are held in the GPU working set at a time and the search sweeps
  $2^{N-\texttt{suffix\_size}}$ such chunks. It is bounded by device memory and by the L2
  budget; 23 is a good value on a 96 GB H100. **Larger problems should be reached by raising
  `num_fixed_vars`, not `suffix_size`.**

`grid_size`, `block_size`
: CUDA launch geometry for the custom kernels. More resident threads generally means higher
  throughput; 8192 blocks of 1024 threads works well on data-centre GPUs.

`num_steps_per_kernel`
: How many chunks a single kernel launch processes. Larger values amortize launch overhead;
  the value also determines the granularity at which periodic re-anchoring can occur.

`partial_diff_buffer_depth`
: Depth of the incremental energy-difference buffers. Costs
  $\texttt{depth} \times 2^{\texttt{suffix\_size}}$ elements of device memory.

### Size limits

The kernels keep each configuration in a single 64-bit word, so

* `BruteforceGPUSampler` supports $N \le 64$;
* `DistributedBruteforceGPUSampler` supports $N - k \le 64$, i.e. $N \le 64 + k$.

Both samplers validate this and raise `ValueError` rather than returning a silently
truncated answer.

## Precision and numerical stability

The ground-state path accumulates energies incrementally, which over $2^N$ updates can
accumulate roundoff. For `dtype=np.float32` and kernels seeing at least 40 variables, the
backend therefore enables compensated updates, periodic exact re-anchoring of the energies
and a final refresh of the best-state buffer. This happens automatically and does not change
the API.

Two consequences worth knowing:

* `np.float64` is a **different numerical path**, not merely a slower one — the stabilization
  applies to single precision only.
* The returned *configuration* is an exact bit string. If a certified numerical value is
  needed, recompute the energy from the configuration in double precision:

```python
best = result.first
exact_energy = bqm.energy(best.sample)   # float64, from scratch
```

`dtype` accepts either NumPy types (`np.float32`, `np.float64`) or the names `"float"`,
`"float32"`, `"single"`, `"double"`, `"float64"`. Note that `"float"` means *single*
precision, matching the `--dtype` choices of the CLI.

## Reported timings

Both samplers put timings into `result.info`. The single-GPU sampler reports
`solve_time_in_seconds`, covering the CUDA search itself and excluding host-side assembly of
the QUBO matrix and decoding of the results. The distributed sampler additionally separates
the phases:

| Key | Meaning |
|---|---|
| `dispatch_time_in_seconds` | submitting all $2^k$ subproblem tasks |
| `solve_time_in_seconds` | dispatch plus completion of every subproblem search |
| `merge_time_in_seconds` | combining the partial results and fetching the final one |
| `total_time_in_seconds` | the sum of the two above |
| `num_subproblems`, `num_merge_rounds` | shape of the merge hierarchy |

Partial results are combined by a hierarchy of Ray tasks rather than on the controller, so
the merge cost grows with the *depth* of the hierarchy, $\log_{\texttt{merge\_batch\_size}}
2^k$, rather than linearly with the number of subproblems. Passing a `merge_batch_size` of at
least $2^k$ recovers the single-shot merge used up to release 0.0.5, which is useful when
comparing against older measurements.
