# Welcome to omnisolver-pt documentation!

The `omnisolver-bruteforce` is Omnisolver plugin implementing the bruteforce (a.k.a exhaustive 
search) algorithm.

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

## Documentation
```{toctree}
:maxdepth: 1
```
