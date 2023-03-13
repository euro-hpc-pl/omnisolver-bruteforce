template <typename T>
void search(
    T* qubo,
    int N,
    uint64_t num_states,
    uint64_t* states_out,
    T* energies_out,
    int block_per_grid,
    int threads_per_block,
    int suffix_size
);

template <typename T>
void search(
    T* qubo,
    int N,
    T* energies_out,
    uint64_t* states_out,
    uint64_t num_states_in_chunk,
    int blocks_per_grid,
    int threads_per_block
);
