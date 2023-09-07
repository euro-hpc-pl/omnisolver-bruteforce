// Implementation of exhaustive search using a CUDA--enabled GPU.
#include <bit>
#include <random>
#include <stdexcept>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "vectors.h"

// Basic error checking (used only for kernel launches)
// It should throw an instance of std::runtime_error containing the decoded
// cuda runtime error message and also filename and line at which the error
// occurred.
#define cuda_error_check(code) { cuda_assert((code), __FILE__, __LINE__); }

inline void cuda_assert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      std::stringstream what;
      what << "File: " << __FILE__ << " at " << __LINE__ << ", error: " << cudaGetErrorString(code);
      throw std::runtime_error(what.str());
   }
}

// Compute i-th Gray code.
__host__ __device__
uint64_t gray(uint64_t index) { return index ^ (index >> 1); }

// Find First bit Set in a number.
// Following common convention for this function it returns bits indexed from 1
// (and not as usually from 0). For the special case of number=0 ffs returns 0.
// Examples:
//     ffs(7) = 1
//     ffs(10) = 2
//     ffs(24) = 4
int ffs(uint64_t number) {
    if(number == 0) {
        return 0;
    }
    int result = 1;
    while((number & 1) != 1) {
        number >>= 1;
        result++;
    }

    return result;
}

// Given a QUBO and a state, compute how the energy changes if the i-th bit of state
// is flipped.
// Specifically, E(.) is energy function of the qubo, then this function computes
// E(s) - E(s'), where s' is the state after flipping i-th bit of s.
// One can easily verify that to compute the difference above we need only the
// i-th row of QUBO, and hence we already pass this row instead of computign the offset.
template <typename T>
__host__ __device__ __forceinline__ T energy_diff(T* qubo_row, int N, uint64_t state, int i) {
    int qi = (state >> i) & 1; // This is the i-th bit of state
    T total = qubo_row[i];     // Start accumulating from the linear term

    uint32_t word = state & 0xFFFFFFFF;

    // Go through each bit of state
    for(int j = 0; j < 32 && j < N; j++) {
        if(i != j && (word % 2)) { // except the one to flip, it's already accounted for
            total += qubo_row[j];
        }
        word /= 2; //>>= 1; // move to next bit
    }

    word = state >> 32;
    for(int j=32; j < N; j++) {
        if(i != j && (word % 2)) { // except the one to flip, it's already accounted for
            total += qubo_row[j];
        }
        word /= 2; // move to next bit
    }

    // When flipping from 0 to 1 there is nothing to do, otherwise we need
    // to multiply the total by -1.
    return (2 * qi - 1) * total;
}

template <typename T>
__device__ __forceinline__ T energy_diff_reduced(T* qubo_row, int N, int offset, uint64_t state, int i, T common_part) {
    int qi = (state >> i) & 1; // This is the i-th bit of state
    T total = common_part;     // Start accumulating from the linear term

    uint32_t word = (state >> offset) & 0xFFFFFFFF;

    // Go through each bit of state
    for(int j = offset; j < 32 + offset && j < N; j++) {
        if(word % 2) total += qubo_row[j];
        word /= 2; //>>= 1; // move to next bit
    }

    word = state >> (32+offset);
    for(int j=32+offset; j < N; j++) {
        if(word % 2) total += qubo_row[j];
        word /= 2; // move to next bit
    }

    // When flipping from 0 to 1 there is nothing to do, otherwise we need
    // to multiply the total by -1.
    return (2 * qi - 1) * total;
}

// Initialize first states and energies for further processing.
// The idea is as follows. Each state (which is N-bit number) is (logically) split into
// two parts:
// l-bit part specific to state
// k-bit part that changes in all states in the same way for all states in given iteration.
// This function computes energy of a state with some l-bits set according to index,
// and k least significant bits set to 0.
template <typename T>
__device__ void _init(T* qubo, int N, uint64_t* states, T* energies, uint64_t index) {
    T energy = 0.0;
    uint64_t state = 0, suffix = index, offset = 0;

    // Since we don't pass l or k, we cannot infer how many bits are fixed.
    // Instead we enumerate bits from the most significant one.
    // For instance, if index=12, which is 110 in binary, we will set the three
    // most significant bits of state to 011.
    while(suffix > 0) {
        // Find first set (note: not our function, this one is CUDA's intrinsic)
        uint64_t bit_in_suffix = __ffsll(suffix);
        // Compute bit index from the most significant position
        // Since we reduce suffix in each iteration, we have to store the offset
        // travelled so far in the `offset` variable.
        uint64_t bit_to_flip = N - bit_in_suffix - offset;
        // Update energy
        energy -= energy_diff(qubo + N * bit_to_flip, N, state, bit_to_flip);
        // Actually flip bit
        state = state | (1L << bit_to_flip);
        // Move to next bit in suffix
        suffix >>= bit_in_suffix;
        // Move the offset
        offset += bit_in_suffix;
    }

    // Finally store both state and energy in the global memory.
    energies[index] = energy;
    states[index] = state;
}

// Kernel that basically launched the _init function above for all states and energies
template <typename T>
__global__ void init(
    T* qubo,
    int N,
    T* energies,
    uint64_t* states,
    uint64_t states_in_chunk
) {
    for(uint64_t i=blockIdx.x * blockDim.x + threadIdx.x; i < states_in_chunk; i += blockDim.x * gridDim.x) {
        _init(qubo, N, states, energies, i);
    }
}

// A single step of bruteforce algorithm.
// Recall that we split all states (logically) into l-most significant fixed bits
// and k-least significant bits that are modified, one at a time, in each iteration.
// This kernel flips one of those k-significant bits.
template <typename T>
__global__ void single_step(
    T* qubo,
    int N,
    T* energies,
    uint64_t* states,
    int bit_to_flip,
    uint64_t states_in_chunk
) {
    T* qubo_row = qubo + N * bit_to_flip;

    for(uint64_t i=blockIdx.x * blockDim.x + threadIdx.x; i < states_in_chunk; i += blockDim.x * gridDim.x) {
        uint64_t state = states[i];
        T energy = energies[i];
        energies[i] = energy - energy_diff(qubo_row, N, state, bit_to_flip);
        states[i] = state ^ (1L << bit_to_flip);
    }
}


template <typename T>
__device__ __forceinline__ T* row_ptr(T* array_2d, uint64_t n_cols, uint64_t row_id) {
    return array_2d + n_cols * row_id;
}


// Kernel performing whole ground state (and only ground state) computation in a single step
template <typename T>
__global__ void find_ground(
    T* qubo,
    int N,
    T* energies,
    uint64_t* states,
    T* best_energies,
    uint64_t* best_states,
    uint64_t suffix_size,
    uint64_t num_iterations,
    int* bit_buffer,
    T* prefix_diff_buffer,
    T* partial_diff_buffer,
    int partial_diff_buffer_depth
) {
    uint64_t chunk_size = 1L << suffix_size;
    T tmp_energy, candidate_energy;
    uint64_t tmp_state, candidate_state;
    int bit_to_flip;
    int qi;

    for(uint64_t i=blockIdx.x * blockDim.x + threadIdx.x; i < chunk_size; i += blockDim.x * gridDim.x) {
        candidate_energy = best_energies[i];
        tmp_energy = energies[i];
        candidate_state = best_states[i];
        tmp_state = states[i];
        for(uint64_t j=0; j < num_iterations; j++) {
            bit_to_flip = bit_buffer[j];
            if(bit_to_flip < partial_diff_buffer_depth) {
                qi = (tmp_state >> bit_to_flip) & 1;
                tmp_energy = tmp_energy - (2 * qi - 1) * (
                    row_ptr(partial_diff_buffer, chunk_size, bit_to_flip)[i] +
                    prefix_diff_buffer[j]
                );
            } else {
                tmp_energy -= energy_diff_reduced(
                    row_ptr(qubo, N, bit_to_flip),
                    N,
                    N-suffix_size,
                    tmp_state,
                    bit_to_flip,
                    prefix_diff_buffer[j]
                );
            }

            tmp_state = tmp_state ^ (1L << bit_to_flip);
            if(tmp_energy < candidate_energy) {
                candidate_energy = tmp_energy;
                candidate_state = tmp_state;
            }
        }
        states[i] = tmp_state;
        energies[i] = tmp_energy;
        best_states[i] = candidate_state;
        best_energies[i] = candidate_energy;
    }
}


template <typename T>
__global__ void _initialize_partial_diff_buffer(
    T* qubo,
    int N,
    uint64_t* states,
    uint64_t suffix_size,
    T* partial_diff_buffer,
    int partial_diff_buffer_depth
) {
    uint64_t chunk_size = 1L << suffix_size;
    uint64_t shifted_state, tmp_state;
    T total;
    T* qubo_row;

    for(uint64_t i=blockIdx.x * blockDim.x + threadIdx.x; i < chunk_size; i += blockDim.x * gridDim.x) {
        shifted_state = states[i] >> (N - suffix_size);
        for(int bit_to_flip=0; bit_to_flip < partial_diff_buffer_depth; bit_to_flip++) {
            total = 0;
            tmp_state = shifted_state;
            qubo_row = qubo + N * bit_to_flip;
            for(int k=N-suffix_size; k < N; k++) {
                if(tmp_state % 2) total += qubo_row[k];
                tmp_state /= 2;
            }

            partial_diff_buffer[bit_to_flip * chunk_size + i] = total;
        }
    }
}


// Predicate used in thrust::copy_if
// Given a tuple (state, energy), copy this copule iff energy < threshold.
template <typename T>
struct energy_less_then {
    energy_less_then(T threshold) : threshold(threshold) {};
    T threshold;

    __host__ __device__
    inline bool operator () (const thrust::tuple<uint64_t, T>& pair) {
        return thrust::get<1>(pair) < threshold;
    }
};

// The pièce de résistance of this module, the search function.
// QUBO: N x N matrix of floats or doubles (depending on the template parameter T)
// num_states: requested size of the low energy spectrum
// states_out, energies_out: output buffers, should be of size num_states
// grid_size, block_size: definition of grid on which init and single_step kernels will
//   be launched. Warning: other kernels might be launched by thrust, and we cannot control
//   their grid (and frankly, thrust's judgement is probably better then our judgement)
// suffix_size: the number l determining how many most significant bits in each state are
//   fixed. Since there are 2 ** l possible configurations of l-bits, the temporary buffers
//   for energies and states will also have 2 ** l items.
template <typename T>
void search(
    T* qubo,
    int N,
    uint64_t num_states,
    uint64_t* states_out,
    T* energies_out,
    int grid_size,
    int block_size,
    int suffix_size
) {
    // chunk_size = 2 ** suffix_size
    uint64_t chunk_size = 1 << suffix_size;
    // Device vectors with qubo
    device_vector<T> qubo_gpu(qubo, qubo + N * N);
    // Device vectors for energies in current iteration
    // and best_energies found so far.
    // Observe that best_energies has size num_states + chunk_size
    // The first num_states are currently best found energies. The second part
    // can accommodate the chunk_size "candidate" energies.
    energy_vector<T> energies(chunk_size), best_energies(num_states + chunk_size);
    // Analogously, device vectors for states.
    state_vector states(chunk_size), best_states(num_states + chunk_size);

    // For easier iteration: tuple iterators over current and best spectrum
    auto current_spectrum_it = zip(states, energies);
    auto best_spectrum_it = zip(best_states, best_energies);

    // Number of chunks s.t. num_chunk * chunk_size = 2 ** N.
    uint64_t num_chunks = 1 << (N - suffix_size);

    // Initialize and check if it succeeded.
    init<<<grid_size, block_size>>>((T*)qubo_gpu, N, (T*)energies, states, chunk_size);
    cuda_error_check(cudaGetLastError());
    cudaDeviceSynchronize();

    // Sort the initial spectrum by the energies
    thrust::sort_by_key(energies.begin(), energies.end(), states.begin());

    // The initial states and energies are becoming the first approximation of
    // the low energy spectrum. We copy first num_states of them into best_spectrum_it.
    thrust::copy(
        current_spectrum_it, current_spectrum_it + num_states, best_spectrum_it
    );

    // Iterate in chunks in gray code order.
    for(uint64_t i = 1; i < num_chunks; i++) {
        // To compute bit to flip compute two consecutive gray codes and see at which
        // bit they differ. Subtract 1 because ffs counts from 1.
        int bit_to_flip = ffs(gray(i) ^ gray(i - 1)) - 1;
        // Perform single step of energy computation and check if it succeeded.
        single_step<<<grid_size, block_size>>>(
            (T*)qubo_gpu, N, (T*)energies, states, bit_to_flip, chunk_size
        );
        cuda_error_check(cudaGetLastError());


        // Find the maximum energy in current approximation if low enrgy spectrum.
        T threshold = *thrust::max_element(
            thrust::device, best_energies.begin(), best_energies.begin() + num_states
        );

        // From the currently computed chunk, the only candidates to enter the low energy
        // spectrum are the ones that have energy < threshold.
        // Hence, we copy only those states and energies into the range directly being
        // the current approximation of low energy spectrum.
        auto candidates_it = best_spectrum_it + num_states;

        // We also store the position at which we placed the last candidate state.
        // This way, we can sometimes sort much smaller range.
        auto end = thrust::copy_if(
            thrust::device,
            current_spectrum_it,
            current_spectrum_it + chunk_size,
            candidates_it,
            energy_less_then<T>(threshold)
        );

        // This sort effectively combines the current approximation and candidates.
        thrust::sort_by_key(
            best_energies.begin(),
            best_energies.begin() + num_states + (end - candidates_it),
            best_states.begin()
        );
    }

    // After all chunks are processed, copy the result to the output arrays.
    thrust::copy(best_states.begin(), best_states.begin() + num_states, states_out);
    thrust::copy(best_energies.begin(), best_energies.begin() + num_states, energies_out);
}

// Explicit template instantiations, necessary for use with Cython.
template void search(
    double* qubo,
    int N,
    uint64_t num_states,
    uint64_t* states_out,
    double* energies_out,
    int grid_size,
    int block_size,
    int suffix_size
);

template void search(
    float* qubo,
    int N,
    uint64_t num_states,
    uint64_t* states_out,
    float* energies_out,
    int grid_size,
    int block_size,
    int suffix_size
);

template <typename T>
void search_ground_only(
    T* qubo,
    int N,
    uint64_t* states_out,
    T* energies_out,
    int grid_size,
    int block_size,
    int suffix_size,
    int num_steps_per_kernel,
    int partial_diff_buffer_depth
) {
    uint64_t chunk_size = 1L << suffix_size;
    device_vector<T> qubo_gpu(qubo, qubo + N * N);

    energy_vector<T> energies(chunk_size);
    state_vector states(chunk_size);
    energy_vector<T> best_energies(chunk_size);
    state_vector best_states(chunk_size);

    device_vector<int> bit_flip_buffer(num_steps_per_kernel);
    energy_vector<T> prefix_diff_buffer(num_steps_per_kernel);
    energy_vector<T> partial_diff_buffer(partial_diff_buffer_depth * chunk_size);

    std::vector<int> src_bit_flip_buffer(num_steps_per_kernel);
    std::vector<T> src_prefix_diff_buffer(num_steps_per_kernel);

    uint64_t last_gray_code, next_gray_code;
    uint64_t counter;
    int64_t base_state;
    T base_energy;

    init<<<grid_size, block_size>>>((T*)qubo_gpu, N, (T*)energies, states, chunk_size);
    cuda_error_check(cudaGetLastError());
    cudaDeviceSynchronize();

    best_energies = energies;
    best_states = states;

    _initialize_partial_diff_buffer<<<grid_size, block_size>>>((T*)qubo_gpu, N, states, suffix_size, (T*) partial_diff_buffer, partial_diff_buffer_depth);
    cuda_error_check(cudaGetLastError());
    cudaDeviceSynchronize();

    counter = 0;
    last_gray_code = 0;

    base_energy = 0;
    base_state = 0;
    for(uint64_t i = 0; i < (1L << (N - suffix_size)) / num_steps_per_kernel; i++) {
        for(uint64_t j = 0; j < num_steps_per_kernel; j++) {
            counter += 1;
            next_gray_code = gray(counter);
            int bit_to_flip = ffs(last_gray_code ^ next_gray_code)-1;
            src_bit_flip_buffer[j] = bit_to_flip;
            src_prefix_diff_buffer[j] = qubo[(N + 1) * bit_to_flip];
            for(int k = 0; k < N - suffix_size; k++) {
               if(((base_state >> k) % 2) && k != bit_to_flip) {
                  src_prefix_diff_buffer[j] += qubo[N * bit_to_flip + k];
               }
            }

            last_gray_code = next_gray_code;
            base_state ^= (1L << src_bit_flip_buffer[j]);
        }
        bit_flip_buffer = src_bit_flip_buffer;
        prefix_diff_buffer = src_prefix_diff_buffer;
        cuda_error_check(cudaGetLastError());
        cudaDeviceSynchronize();
        find_ground<<<grid_size, block_size>>>(
            (T*)qubo_gpu,
            N,
            (T*)energies,
            states,
            (T*)best_energies,
            best_states,
            suffix_size,
            num_steps_per_kernel,
            (int*) bit_flip_buffer,
            (T*) prefix_diff_buffer,
            (T*)partial_diff_buffer,
            partial_diff_buffer_depth
        );

    }

    cuda_error_check(cudaGetLastError());
    cudaDeviceSynchronize();
    thrust::sort_by_key(best_energies.begin(), best_energies.end(), best_states.begin());
    thrust::copy(best_states.begin(), best_states.begin() + 1, states_out);
    thrust::copy(best_energies.begin(), best_energies.begin() + 1, energies_out);
}

template void search_ground_only(
    float* qubo,
    int N,
    uint64_t* states_out,
    float* energies_out,
    int grid_size,
    int block_size,
    int suffix_size,
    int num_steps_per_kernel,
    int partial_diff_buffer_depth
);

template void search_ground_only(
    double* qubo,
    int N,
    uint64_t* states_out,
    double* energies_out,
    int grid_size,
    int block_size,
    int suffix_size,
    int num_steps_per_kernel,
    int partial_diff_buffer_depth
);
