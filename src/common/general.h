#ifndef __GENERAL_H__
#define __GENERAL_H__

#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "error_check.h"

#define NTHREADS 4
#define BLOCK_SIZE 16

__managed__ double carcinogen_mutation_map[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

__managed__ int index_map[11*12] = {0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    9, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1,
				    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    2, 1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    2, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				    2, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1};

#define PHENOTYPE_INCR 0.01f
__managed__ double upreg_phenotype_map[11*4] = {0.0f, 0.0f, 0.0f, 0.0f,
					        -PHENOTYPE_INCR, PHENOTYPE_INCR, PHENOTYPE_INCR, 0.0f,
					        0.0f, 0.0f, PHENOTYPE_INCR, 0.0f,
					        -PHENOTYPE_INCR, PHENOTYPE_INCR, 0.0f, 0.0f,
				    	        -PHENOTYPE_INCR, PHENOTYPE_INCR, 0.0f, -PHENOTYPE_INCR,
					        -PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
					        PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
					        0.0f, 0.0f, -PHENOTYPE_INCR, 0.0f,
				    	        PHENOTYPE_INCR, 0.0f, -PHENOTYPE_INCR, PHENOTYPE_INCR,
					        0.0f, 0.0f, -PHENOTYPE_INCR, 0.0f,
					        PHENOTYPE_INCR, 0.0f, -PHENOTYPE_INCR, PHENOTYPE_INCR};

__managed__ double downreg_phenotype_map[11*4] = {0.0f, 0.0f, 0.0f, 0.0f,
						  PHENOTYPE_INCR, -PHENOTYPE_INCR, -PHENOTYPE_INCR, 0.0f,
						  0.0f, 0.0f, -PHENOTYPE_INCR, 0.0f,
						  PHENOTYPE_INCR, -PHENOTYPE_INCR, 0.0f, 0.0f,
				    		  PHENOTYPE_INCR, -PHENOTYPE_INCR, 0.0f, PHENOTYPE_INCR,
						  PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
						  -PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
						  0.0f, 0.0f, PHENOTYPE_INCR, 0.0f,
				    		  -PHENOTYPE_INCR, 0.0f, PHENOTYPE_INCR, -PHENOTYPE_INCR,
						  0.0f, 0.0f, PHENOTYPE_INCR, 0.0f,
						  -PHENOTYPE_INCR, 0.0f, PHENOTYPE_INCR, -PHENOTYPE_INCR};

__managed__ double phenotype_init[7*4] = {0.05f, 0.9f, 0.2f, 0.0f,
					  0.1f, 0.9f, 0.1f, 0.0f,
					  0.1f, 0.9f, 0.05f, 0.3f,
					  0.2f, 0.9f, 0.025f, 0.4f,
					  0.2f, 0.9f, 0.0125f, 0.5f,
					  0.4f, 0.9f, 0.00625f, 0.0f,
					  0.0f, 0.0f, 0.0f, 0.0f};

__managed__ int state_mut_map[6*11] = {0, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
				       1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
				       2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				       3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
				       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

__managed__ int prolif_mut_map[6*11] = {0, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
					1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
					2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
					3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				  	4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4,
					5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

__managed__ int diff_mut_map[6*11] = {-1, 1, -1, -1, 4, -1, -1, -1, 4, -1, 4,
				      -1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
				      0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 4,
				      1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4,
				      -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1,
				      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
#pragma omp threadprivate(carcinogen_mutation_map, index_map, upreg_phenotype_map, downreg_phenotype_map, phenotype_init, state_mut_map, prolif_mut_map, diff_mut_map)

void prefetch_params(int loc) {
	int location = loc;
	if (loc == -1) location = cudaCpuDeviceId;
	CudaSafeCall(cudaMemPrefetchAsync(carcinogen_mutation_map, 10*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(index_map, 11*12*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(upreg_phenotype_map, 11*4*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(downreg_phenotype_map, 11*4*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(phenotype_init, 7*4*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(state_mut_map, 6*11*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(prolif_mut_map, 6*11*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(diff_mut_map, 6*11*sizeof(int), location, NULL));
}

/* Start Array Functions */
__device__ int max_idx(double *L, int *location, int N) {
        double max = L[0];
        int count = 0; int i;

        for (i = 1; i < N; i++) if (L[i] > max) max = L[i];

        for (i = 0; i < N; i++) {
                if (L[i] == max) {
                        location[count] = i;
                        count++;
                }
        }

        return count;
}

__device__ int get_indexes(double val, double *L, int *idx, int N) {
        int count = 0; int i;

        for (i = 0; i < N; i++) {
                if (L[i] == val) {
                        idx[count] = i;
                        count++;
                }
        }

        return count;
}


// Bitonic Sort Functions
__device__ unsigned int greatestPowerOfTwoLessThan(int n) {
	int k = 1;
	while (k > 0 && k < n) k <<= 1;
	return (unsigned int) k >> 1;
}

__device__ void exchange(double *L, int i, int j) {
        double t = L[i];
        L[i] = L[j];
        L[j] = t;
}

__device__ void compare(double *L, int i, int j, bool dir) {
        if (dir==(L[i] > L[j])) exchange(L, i, j);
}

__device__ void bitonic_merge(double *L, int lo, int N, bool dir) {
	int i;
        if (N > 1) {
                int m =greatestPowerOfTwoLessThan(N);
                for (i = lo; i < lo+N-m; i++) compare(L, i, i+m, dir);
                bitonic_merge(L, lo, m, dir);
                bitonic_merge(L, lo+m, N-m, dir);
        }
}

__device__ void bitonic_sort(double *L, int lo, int N, bool dir ) {
        if (N > 1) {
                int m = N/2;
                bitonic_sort(L, lo, m, !dir);
                bitonic_sort(L, lo+m, N-m, dir);
                bitonic_merge(L, lo, N, dir);
        }
}
/* end array functions */

/* Start functions around random number generation */
// Set random seed for host
void set_seed( void ) {
        timespec seed;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &seed);
        srand(seed.tv_nsec);
}

// Set random seed for gpu
__global__ void init_curand(unsigned int seed, curandState_t* states) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &states[id]);
}
/* end functions around random number generation */

int numDigits(double x) {
        x = fabsf(x);
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

#include "../gene_expression_nn.h"
#include "../cell.h"
#include "../carcinogen_pde.h"
#include "gpu_anim.h"
#include "lodepng.h"
#include "../ca.h"

#endif // __GENERAL_H__
