---
hide:
  - navigation
---

<p align="center">
    <a href="https://github.com/euro-hpc-pl/omnisolver"><img src="assets/logo-large.png" alt="Omnisolver"></a>
</p>

---

<h1></h1>

<p align="center">
<em>Exhaustive search plugin for <a href="https://github.com/euro-hpc-pl/omnisolver">Omnisolver</a></em>
</p>

**Source code:** <https://github.com/euro-hpc-pl/omnisolver-bruteforce>

**Framework documentation:** <https://euro-hpc-pl.github.io/omnisolver>

---

`omnisolver-bruteforce` solves Ising and QUBO instances by **exhaustive search on
CUDA-enabled GPUs**. It enumerates all $2^N$ configurations of an $N$-variable instance and
therefore returns a *certified* optimum rather than a heuristic one, which makes it useful
as a ground-truth reference for annealers, heuristics and quantum-inspired solvers.

The plugin provides two samplers:

| Sampler | Scope | Use when |
|---|---|---|
| `BruteforceGPUSampler` | one node, one GPU | a single device has enough memory and the expected wall-clock time is acceptable |
| `DistributedBruteforceGPUSampler` | many GPUs, many nodes, via [Ray](https://www.ray.io/) | the problem no longer fits a single device within a practical time frame |

The distributed sampler fixes $k$ variables, turning the search into $2^k$ independent
subproblems of $N-k$ variables that are dispatched across the available GPUs and merged
afterwards.

## Quickstart

<!-- termynal -->

```
# Point the build at your CUDA installation
$ export CUDAHOME=/usr/local/cuda
# Install the plugin (add [distributed] for the multi-GPU sampler)
$ pip install "omnisolver-bruteforce[distributed]"
---> 100%
Successfully installed omnisolver-bruteforce
# Create an instance file in COOrdinate format
$ echo "0 1 1.0
> 1 2 1.0
> 2 0 1.0" > instance.txt
# Run the solver
$ omnisolver bruteforce-gpu --vartype SPIN --num_states 1 instance.txt
0,1,2,energy,num_occurrences
1,-1,1,-1.0,1
```

## Scope and limitations

Exhaustive search costs $O(2^N)$ regardless of how many GPUs are used: each added variable
doubles the runtime. The plugin lowers the constant factor and makes the search numerically
dependable at large $N$, but it cannot change that scaling. Treat it as a **verification and
certification backend**, not as a routine optimizer.

Concrete bounds:

* each subproblem is held in a 64-bit word, so a single kernel enumerates at most
  **64 variables**; the distributed sampler therefore reaches $N \le 64 + k$;
* `suffix_size` is bounded by the working-set budget of the device;
* the stabilized single-precision path engages from 40 variables per kernel upwards.

See the [user guide](userguide.md) for details and for how to choose the parameters.

## Citing

If you used this plugin in your research, please cite both the framework and the CUDA
kernel it builds on:

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
