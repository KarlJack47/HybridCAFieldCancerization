#ifndef __GENERAL_H__
#define __GENERAL_H__

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <sys/time.h>
#include "error_check.h"

#define BLOCK_SIZE 16

/* Start Array Functions */
#ifndef MAX_IDX
#define MAX_IDX
__device__ int max_idx( float *L, int *location, int N ) {
        float max = L[0];
        int count = 0;

        for (int i = 1; i < N; i++) {
                if (L[i] > max) {
                        max = L[i];
                }
        }

        for (int i = 0; i < N; i++) {
                if (L[i] == max) {
                        location[count] = i;
                        count++;
                }
        }

        return count;
}
#endif

#ifndef GET_INDEXES
#define GET_INDEXES
__device__ int get_indexes(float val, float *L, int *idx, int N) {
        int count = 0;
        for (int i = 0; i < N; i++) {
                if (L[i] == val) {
                        idx[count] = i;
                        count++;
                }
        }

        return count;
}
#endif


// Bitonic Sort Functions
#ifndef EXCHANGE
#define EXCHANGE
__device__ void exchange(float *L, int i, int j) {
        float t = L[i];
        L[i] = L[j];
        L[j] = t;
}
#endif

#ifndef COMPARE
#define COMPARE
__device__ void compare(float *L, int i, int j, bool dir) {
        if (dir==(L[i]>L[j]))
                exchange(L, i, j);
}
#endif

#ifndef BITONIC_MERGE
#define BITONIC_MERGE
__device__ void bitonic_merge(float *L, int lo, int N, bool dir ) {
        if (N > 1) {
                int m = N/2;
                for (int i = lo; i < lo+m; i++)
                        compare(L, i, i+m, dir);
                bitonic_merge(L, lo, m, dir);
                bitonic_merge(L, lo+m, m, dir);
        }
}
#endif

#ifndef BITONIC_SORT
#define BITONIC_SORT
__device__ void bitonic_sort(float *L, int lo, int N, bool dir ) {
        if (N > 1) {
                int m = N/2;
                bitonic_sort(L, lo, m, true);
                bitonic_sort(L, lo+m, m, false);
                bitonic_merge(L, lo, N, dir);
        }
}
#endif
/* end array functions */

/* Start functions around random number generation */
#ifndef SET_SEED
#define SET_SEED
// Set random seed for host
void set_seed( void ) {
        timespec seed;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &seed);
        srand(seed.tv_nsec);
}
#endif

#ifndef INIT_CURAND
#define INIT_CURAND
// Set random seed for gpu
__global__ void init_curand(unsigned int seed, curandState_t* states) {
        unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &states[id]);
}
#endif
/* end functions around random number generation */

#ifndef NUMDIGITS
#define NUMDIGITS
int numDigits(int x) {
        x = abs(x);
        return (x < 10 ? 1 :
               (x < 100 ? 2 :
               (x < 1000 ? 3 :
               (x < 10000 ? 4 :
               (x < 100000 ? 5 :
               (x < 1000000 ? 6 :
               (x < 10000000 ? 7 :
               (x < 100000000 ? 8 :
               (x < 1000000000 ? 9 :
                10)))))))));
}
#endif

#endif // __GENERAL_H__
