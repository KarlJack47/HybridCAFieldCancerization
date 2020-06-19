#ifndef __ARRAY_FUNCTIONS_H__
#define __ARRAY_FUNCTIONS_H__

template <class T>
__host__ __device__ unsigned get_indexes(T val, T *L, unsigned *idx,
                                         unsigned N)
{
    unsigned count = 0, i;

    for (i = 0; i < N; i++) {
        if (L[i] == val) {
            idx[count] = i;
            count++;
        }
    }

    return count;
}

// Bitonic Sort Functions
__host__ __device__ unsigned greatest_power_of_two_less_than(unsigned n)
{
    int k = 1;

    while (k > 0 && k < n) k <<= 1;

    return (unsigned) k >> 1;
}

template<class T>
__host__ __device__ void exchange(T *L, unsigned i, unsigned j)
{
    T t = L[i];

    L[i] = L[j];
    L[j] = t;
}

template<class T>
__host__ __device__ void compare(T *L, unsigned i, unsigned j, bool dir)
{
    if (dir==(L[i] > L[j])) exchange(L, i, j);
}

template<class T>
__host__ __device__ void bitonic_merge(T *L, int lo, unsigned N, bool dir)
{
    unsigned i, m;

    if (N > 1) {
        m = greatest_power_of_two_less_than(N);
        for (i = lo; i < lo+N-m; i++) compare(L, i, i+m, dir);
        bitonic_merge(L, lo, m, dir);
        bitonic_merge(L, lo+m, N-m, dir);
    }
}

template<class T>
__host__ __device__ void bitonic_sort(T *L, int lo, unsigned N, bool dir)
{
    unsigned m;

    if (N > 1) {
        m = N/2;
        bitonic_sort(L, lo, m, !dir);
        bitonic_sort(L, lo+m, N-m, dir);
        bitonic_merge(L, lo, N, dir);
    }
}

template<class T>
__device__ unsigned get_rand_idx(T *L, const unsigned N,
                                 curandState_t *rndState, unsigned *idx=NULL)
{
    curandState_t localState = *rndState;
    double rnd = curand_uniform_double(&localState);
    T sum = 0.0;
    T *sorted = (T*)malloc(N*sizeof(T));
    unsigned *idxT = NULL, i;
    int count = -1, chosen = -1;

    if (idx == NULL) idxT = (unsigned*)malloc(N*sizeof(unsigned));
    else idxT = idx;

    for (i = 0; i < N; i++) sorted[i] = 1000000000.0 * L[i];
    bitonic_sort(sorted, 0, N, false);
	for (i = 0; i < N; i++) sum += sorted[i];
	rnd *= sum;

    for (i = 0; i < N; i++) {
        rnd -= sorted[i];
        if (rnd < 0) {
            if (idx == NULL) {
                count = get_indexes(sorted[i] / 1000000000.0, L, idxT, N);
                rnd = curand_uniform_double(&localState);
                chosen = idxT[(unsigned) ceil(rnd * (double) count) % count];
                free(idxT); idxT = NULL;
                break;
            } else {
                count = get_indexes(sorted[i] / 1000000000.0, L, idx, N);
                break;
            }
        }
    }

    rnd = curand_uniform_double(&localState);
    *rndState = localState;
    free(sorted); sorted = NULL;

    if (idx == NULL) free(idxT);

    if (chosen != -1) return chosen;
    else if (count != -1) return count;

    return (unsigned) ceil(rnd * (double) N) % N;
}

#endif // __ARRAY_FUNCTIONS_H__