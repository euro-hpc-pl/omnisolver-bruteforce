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
void search_ground_only(
    T* qubo,
    int N,
    uint64_t* states_out,
    T* energies_out,
    int blocks_per_grid,
    int threads_per_block,
    int suffix_size
);
