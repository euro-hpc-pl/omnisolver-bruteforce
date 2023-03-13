# distutils: language=c++
from libc.stdint cimport uint64_t

#from numpy cimport uint64_t


ctypedef fused real:
    double
    float


cdef extern from "bruteforce_gpu.h":
    void search[T](
        T* qubo,
        int N,
        uint64_t num_states,
        uint64_t* states_out,
        T* energies_out,
        int block_per_grid,
        int threads_per_block,
        int suffix_size
    )


def gpu_search(
    real[:,:] qubo,
    uint64_t num_states,
    uint64_t[::1] states_out,
    real[::1] energies_out,
    int grid_size,
    int block_size,
    int suffix_size
):
    search(
        &qubo[0, 0],
        qubo.shape[0],
        num_states,
        &states_out[0],
        &energies_out[0],
        grid_size,
        block_size,
        suffix_size
    )
