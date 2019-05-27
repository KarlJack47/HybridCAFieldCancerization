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
#define MAX_EXCISE 100
#define NUM_GENES 10
#define NUM_PHENO 4
#define NUM_CARCIN 1
#define NUM_STATES 7
#define NC 0
#define MNC 1
#define SC 2
#define MSC 3
#define CSC 4
#define TC 5
#define EMPTY 6
#define NUM_NEIGH 8
#define PROLIF 0
#define QUIES 1
#define APOP 2
#define DIFF 3
#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3
#define NORTH_EAST 4
#define SOUTH_EAST 5
#define SOUTH_WEST 6
#define NORTH_WEST 7

#define PHENOTYPE_INCR 0.001f
#define MUT_THRESHOLD 0.1f
#define BIAS 0.001f
#define ALPHA 100000
#define CHANCE_MOVE 0.25f
#define CHANCE_KILL 0.35f
#define CHANCE_UPREG 0.5f
#define CHANCE_PHENO_MUT 0.5f
#define CSC_GENE_IDX 2

__managed__ unsigned int state_colors[NUM_STATES*3] = {0, 0, 0, // black (NC)
					      	       87, 207, 0, // green (MNC)
					      	       244, 131, 0, // orange (SC)
					      	       0, 0, 255, // blue (MSC)
					     	       89, 35, 112, // purple (CSC)
					      	       255, 0, 0, // red (TC)
					      	       255, 255, 255}; // white (empty)

__managed__ double carcinogen_mutation_map[NUM_GENES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
__managed__ double upreg_phenotype_map[NUM_GENES*NUM_PHENO] = {-PHENOTYPE_INCR, PHENOTYPE_INCR, PHENOTYPE_INCR, 0.0f,
					        		0.0f, 0.0f, PHENOTYPE_INCR, 0.0f,
					        		-PHENOTYPE_INCR, PHENOTYPE_INCR, 0.0f, 0.0f,
				    	        		-PHENOTYPE_INCR, PHENOTYPE_INCR, 0.0f, -PHENOTYPE_INCR,
					       			-PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
					        		PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
					        		0.0f, 0.0f, -PHENOTYPE_INCR, 0.0f,
				    	        		PHENOTYPE_INCR, 0.0f, -PHENOTYPE_INCR, PHENOTYPE_INCR,
					        		0.0f, 0.0f, -PHENOTYPE_INCR, 0.0f,
					        		PHENOTYPE_INCR, 0.0f, -PHENOTYPE_INCR, PHENOTYPE_INCR};
__managed__ double downreg_phenotype_map[NUM_GENES*NUM_PHENO] = {PHENOTYPE_INCR, -PHENOTYPE_INCR, -PHENOTYPE_INCR, 0.0f,
						  	 	 0.0f, 0.0f, -PHENOTYPE_INCR, 0.0f,
						  	 	 PHENOTYPE_INCR, -PHENOTYPE_INCR, 0.0f, 0.0f,
				    		  	 	 PHENOTYPE_INCR, -PHENOTYPE_INCR, 0.0f, PHENOTYPE_INCR,
						  	 	 PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
						  	 	 -PHENOTYPE_INCR, 0.0f, 0.0f, 0.0f,
						  	 	 0.0f, 0.0f, PHENOTYPE_INCR, 0.0f,
				    		  	 	 -PHENOTYPE_INCR, 0.0f, PHENOTYPE_INCR, -PHENOTYPE_INCR,
						  	 	 0.0f, 0.0f, PHENOTYPE_INCR, 0.0f,
						  	 	 -PHENOTYPE_INCR, 0.0f, PHENOTYPE_INCR, -PHENOTYPE_INCR};
__managed__ double phenotype_init[NUM_STATES*NUM_PHENO] = {0.05f, 0.9f, 0.01f, 0.0f,
					  		   0.1f, 0.9f, 0.005f, 0.0f,
					  		   0.05f, 0.9f, 0.01f, 0.2f,
					  		   0.1f, 0.9f, 0.005f, 0.25f,
					  		   0.05f, 0.9f, 0.0025f, 0.2f,
					  		   0.25f, 0.9f, 0.005f, 0.0f,
					  		   0.0f, 0.0f, 0.0f, 0.0f};
__managed__ int state_mut_map[(NUM_STATES-1)*NUM_GENES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				      	      		   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				       	      		   3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				       	      		   3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				       	      		   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
					      		   5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
__managed__ int prolif_mut_map[(NUM_STATES-1)*NUM_GENES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					       		    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					       		    3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
					       		    3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				  	       		    4, 4, 5, 4, 4, 4, 4, 4, 4, 4,
					       		    5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
__managed__ int diff_mut_map[(NUM_STATES-1)*(NUM_GENES+1)] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				      		 	      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				      		 	      0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 4,
				      		 	      1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4,
				      		 	      -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1,
				      		 	      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

// 0: tumour supressor, 1: oncogene
__managed__ unsigned int gene_type[NUM_GENES] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

#pragma omp threadprivate(carcinogen_mutation_map, upreg_phenotype_map, downreg_phenotype_map, phenotype_init, state_mut_map, prolif_mut_map, diff_mut_map)

void prefetch_params(int loc) {
	int location = loc;
	if (loc == -1) location = cudaCpuDeviceId;
	CudaSafeCall(cudaMemPrefetchAsync(state_colors, NUM_STATES*3*sizeof(unsigned int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(carcinogen_mutation_map, NUM_GENES*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(upreg_phenotype_map, NUM_GENES*NUM_PHENO*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(downreg_phenotype_map, NUM_GENES*NUM_PHENO*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(phenotype_init, NUM_STATES*NUM_PHENO*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(state_mut_map, (NUM_STATES-1)*NUM_GENES*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(prolif_mut_map, (NUM_STATES-1)*NUM_GENES*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(diff_mut_map, (NUM_STATES-1)*NUM_GENES*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(gene_type, NUM_GENES*sizeof(unsigned int), location, NULL));
}

/* Start Array Functions */
__device__ int max_idx(double *L, unsigned int *location, unsigned int N) {
        double max = L[0];
        unsigned int count = 0; unsigned int i;

        for (i = 1; i < N; i++) if (L[i] > max) max = L[i];

        for (i = 0; i < N; i++) {
                if (L[i] == max) {
                        location[count] = i;
                        count++;
                }
        }

        return count;
}

__device__ unsigned int get_indexes(double val, double *L, unsigned int *idx, unsigned int N) {
        unsigned int count = 0; unsigned int i;

        for (i = 0; i < N; i++) {
                if (L[i] == val) {
                        idx[count] = i;
                        count++;
                }
        }

        return count;
}


// Bitonic Sort Functions
__device__ unsigned int greatestPowerOfTwoLessThan(unsigned int n) {
	int k = 1;
	while (k > 0 && k < n) k <<= 1;
	return (unsigned int) k >> 1;
}

__device__ void exchange(double *L, unsigned int i, unsigned int j) {
        double t = L[i];
        L[i] = L[j];
        L[j] = t;
}

__device__ void compare(double *L, unsigned int i, unsigned int j, bool dir) {
        if (dir==(L[i] > L[j])) exchange(L, i, j);
}

__device__ void bitonic_merge(double *L, int lo, unsigned int N, bool dir) {
	int i;
        if (N > 1) {
                int m =greatestPowerOfTwoLessThan(N);
                for (i = lo; i < lo+N-m; i++) compare(L, i, i+m, dir);
                bitonic_merge(L, lo, m, dir);
                bitonic_merge(L, lo+m, N-m, dir);
        }
}

__device__ void bitonic_sort(double *L, int lo, unsigned int N, bool dir) {
        if (N > 1) {
                unsigned int m = N/2;
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
        unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void copy_frame(uchar4 *optr, unsigned char *frame) {
	// map from threadIdx/BlockIdx to pixel position
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int dim = gridDim.x*blockDim.x;
	unsigned int idx = x + ((dim-1)-y)*dim;

	frame[4*dim*y+4*x] = optr[idx].x;
	frame[4*dim*y+4*x+1] = optr[idx].y;
	frame[4*dim*y+4*x+2] = optr[idx].z;
	frame[4*dim*y+4*x+3] = 255;
}

#include "../gene_expression_nn.h"
#include "../cell.h"
#include "../carcinogen_pde.h"
#include "lodepng.h"
#include "gpu_anim.h"
#include "../ca.h"

#endif // __GENERAL_H__
