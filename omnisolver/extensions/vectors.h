#ifndef _VECTORS_H
#define _VECTORS_H
// Types representing vectors, derived from thrust::device_vector<T>
//
// Rationale:
// Thrust device vectors compared to raw CUDA pointers offer simplified memory allocation and are
// far easier to use with thrust algorithms, However, when calling kernels, it is necessary to use
// pointers of vectors' underlying data. The types implemented here offer an implicit conversion
// operator, which is more elegant than using thrust::raw_pointer_cast(vector.data()) over and
// over again.
#include <thrust/tuple.h>
#include <thrust/device_vector.h>

template <typename T>
using base_vec = thrust::device_vector<T>;

template <typename T>
class device_vector: public base_vec<T> {
    using base_vec<T>::base_vec;
    public:
        operator T*() { return thrust::raw_pointer_cast(this->data()); };
};

using state_vector = device_vector<uint64_t>;

template <typename T>
using energy_vector = device_vector<T>;


template <typename T>
auto zip(state_vector& states, device_vector<T>& energies) {
    return thrust::make_zip_iterator(thrust::make_tuple(states.begin(), energies.begin()));
}
#endif
