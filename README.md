
![Logo](https://raw.githubusercontent.com/euro-hpc-pl/omnisolver/master/logo.png)
*Bruteforce (a.k.a. exhaustive search) Plugin for [Omnisolver](https://github.com/euro-hpc-pl/omnisolver)*


## Installation 

Currently, the `omnisolver-bruteforce` package requires working CUDA installation.

To install the plugin first set the `CUDAHOME` environmental library to your CUDA instalaltion location, e.g.:

```shell
# Rmember, your actual location  may vary!
export CUDAHOME=/usr/local/cuda
```

and then run:

```shell
pip install omnisolver-bruteforce
```

> **Warning**
> If you don't set the `CUDAHOME` directory, an attempt will be made to deduce it based on the location of your `nvcc` compiler.
> However, this process might not work in all the cases and should not be relied on.

## Command line usage

```text
usage: omnisolver bruteforce-gpu [-h] [--output OUTPUT] [--vartype {SPIN,BINARY}] [--num_states NUM_STATES] [--suffix_size SUFFIX_SIZE] [--grid_size GRID_SIZE]
                                 [--block_size BLOCK_SIZE] [--dtype {float,double}]
                                 input

Bruteforce (a.k.a exhaustive search) sampler using CUA-enabled GPU

positional arguments:
  input                 Path of the input BQM file in COO format. If not specified, stdin is used.

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Path of the output file. If not specified, stdout is used.
  --vartype {SPIN,BINARY}
                        Variable type
  --num_states NUM_STATES

```

## Citing

If you used the Omnisolver package or one of its plugins, please cite:

```text
@misc{https://doi.org/10.48550/arxiv.2112.11131,
  doi = {10.48550/ARXIV.2112.11131},
  
  url = {https://arxiv.org/abs/2112.11131},
  
  author = {Jałowiecki, Konrad and Pawela, Łukasz},
  
  keywords = {Software Engineering (cs.SE), Quantum Physics (quant-ph), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences},
  
  title = {Omnisolver: an extensible interface to Ising spin glass solvers},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
