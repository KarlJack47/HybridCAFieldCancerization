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
#include "lodepng.h"

// Biological numbers
#define CELL_VOLUME 1.596e-9 // relative to cm, epithelial cell
#define CELL_CYCLE_LEN 10.0f // in hours, for tastebuds

// Number of iterations for infinite sums
#define MAX_ITER 100

// block size for CUDA kernels
#define BLOCK_SIZE 32

// Definitions related to quantities with fixed sizes
#define MAX_EXCISE 100
#define NUM_GENES 10
#define NUM_PHENO 4
#define NUM_CARCIN 1
#define NUM_STATES 7
#define NUM_NEIGH 8

// CA states
#define NC 0
#define MNC 1
#define SC 2
#define MSC 3
#define CSC 4
#define TC 5
#define EMPTY 6

// Phenotype states
#define PROLIF 0
#define QUIES 1
#define APOP 2
#define DIFF 3

// Neighborhood directions
#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3
#define NORTH_EAST 4
#define SOUTH_EAST 5
#define SOUTH_WEST 6
#define NORTH_WEST 7

// Parameters for the gene expression NN
#define ALPHA 100000
#define BIAS 0.001f

// Thresholds and parameters for the CA
#define PHENOTYPE_INCR 0.001f
#define EXPR_ADJ_MAX_INCR 1.0f/sqrtf(ALPHA)
#define MUT_THRESHOLD 0.1f
#define CSC_GENE_IDX 2
#define CHANCE_MOVE 0.25f
#define CHANCE_KILL 0.35f
#define CHANCE_UPREG 0.5f
#define CHANCE_PHENO_MUT 0.5f
#define CHANCE_EXPR_ADJ 0.3f

// Related to CSC and TC formation tracking.
__managed__ bool csc_formed; // used to check when first CSC forms
__managed__ bool tc_formed[MAX_EXCISE+1]; // Used to check when first TC forms
// These are related to tracking tumor excisions
__managed__ unsigned int excise_count; // Current excision number
__managed__ unsigned int time_tc_alive; // Records the time step the TC was formed
__managed__ unsigned int time_tc_dead; // Records the time step the TC died

__managed__ unsigned int state_colors[NUM_STATES*3] = {0, 0, 0, // black (NC)
					      	       87, 207, 0, // green (MNC)
					      	       244, 131, 0, // orange (SC)
					      	       0, 0, 255, // blue (MSC)
					     	       89, 35, 112, // purple (CSC)
					      	       255, 0, 0, // red (TC)
					      	       255, 255, 255}; // white (empty)
__managed__ unsigned int gene_colors[NUM_GENES*3] = {84, 48, 5, // Dark brown (TP53)
						     140, 81, 10, // Light brown
						     191, 129, 45, // Brown orange
						     223, 194, 125, // Pale brown
						     246, 232, 195, // Pale
						     199, 234, 229, // Baby blue
						     128, 205, 193, // Blueish green
						     53, 151, 143, // Tourquois
						     1, 102, 94, // Dark green
						     0, 60, 48}; // Forest green


__managed__ double carcinogen_mutation_map[NUM_GENES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // Used to initialize W_x

// Upregulation and downregulation phenotype matrices
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

// Initial values for the phenotype vectors for each CA state
__managed__ double phenotype_init[NUM_STATES*NUM_PHENO] = {0.05f, 0.9f, 0.01f, 0.0f,
					  		   0.1f, 0.9f, 0.005f, 0.0f,
					  		   0.05f, 0.9f, 0.01f, 0.2f,
					  		   0.1f, 0.9f, 0.005f, 0.25f,
					  		   0.05f, 0.9f, 0.0025f, 0.2f,
					  		   0.25f, 0.9f, 0.005f, 0.0f,
					  		   0.0f, 0.0f, 0.0f, 0.0f};

// Says what the new state of a cell is relative a mutation in each gene.
__managed__ int state_mut_map[(NUM_STATES-1)*NUM_GENES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				      	      		   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				       	      		   3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				       	      		   3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				       	      		   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
					      		   5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
// Used to determine the state of the child cell produced by proliferation related to a mutation in each gene
__managed__ int prolif_mut_map[(NUM_STATES-1)*NUM_GENES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					       		    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					       		    3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
					       		    3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				  	       		    4, 4, 5, 4, 4, 4, 4, 4, 4, 4,
					       		    5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
// Used to determine the state of the child cell produced by differentiation related to a mutation in each gene
__managed__ int diff_mut_map[(NUM_STATES-1)*(NUM_GENES+1)] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				      		 	      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
				      		 	      0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 4,
				      		 	      1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4,
				      		 	      -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1,
				      		 	      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

// Labels each gene as a tumor supressor (0) or oncogene (1). Used to see if a gene is positively mutated towards cancer.
__managed__ unsigned int gene_type[NUM_GENES] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
// Gene relationships, 0: unrelated, 1: related
__managed__ int gene_relations[NUM_GENES*NUM_GENES] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
						       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						       1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
						       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						       0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
						       0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
						       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						       0, 0, 0, 0, 0, 0, 1, 1, 0, 0};

void prefetch_params(int loc) {
	int location = loc;
	if (loc == -1) location = cudaCpuDeviceId;
	CudaSafeCall(cudaMemPrefetchAsync(state_colors, NUM_STATES*3*sizeof(unsigned int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(gene_colors, NUM_GENES*3*sizeof(unsigned int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(carcinogen_mutation_map, NUM_GENES*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(upreg_phenotype_map, NUM_GENES*NUM_PHENO*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(downreg_phenotype_map, NUM_GENES*NUM_PHENO*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(phenotype_init, NUM_STATES*NUM_PHENO*sizeof(double), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(state_mut_map, (NUM_STATES-1)*NUM_GENES*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(prolif_mut_map, (NUM_STATES-1)*NUM_GENES*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(diff_mut_map, (NUM_STATES-1)*NUM_GENES*sizeof(int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(gene_type, NUM_GENES*sizeof(unsigned int), location, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(gene_relations, NUM_GENES*NUM_GENES*sizeof(int), location, NULL));
}

/* Start Array Functions */
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

__device__ unsigned int get_rand_idx(double *L, const unsigned int N, unsigned int curand_idx, curandState_t *states, unsigned int *idx=NULL) {
	double *sorted = (double*)malloc(N*sizeof(double));
	unsigned int *idx_t; unsigned int i;
	if (idx==NULL) idx_t = (unsigned int*)malloc(N*sizeof(unsigned int));
	for (i = 0; i < N; i++) sorted[i] = 1000000000.0f * L[i];
	bitonic_sort(sorted, 0, N, false);
	double sum = 0.0f;
	for (i = 0; i < N; i++) sum += sorted[i];
	double rnd = curand_uniform_double(&states[curand_idx]) * sum;
	for (i = 0; i < N; i++) {
		rnd -= sorted[i];
		if (rnd < 0) {
			if (idx==NULL) {
				unsigned int count = get_indexes(sorted[i] / 1000000000.0f, L, idx_t, N);
				unsigned int chosen = idx_t[(unsigned int) ceilf(curand_uniform_double(&states[curand_idx])*(double) count) % count];
				free(sorted); free(idx_t);
				return chosen;
			} else {
				int count = get_indexes(sorted[i] / 1000000000.0f, L, idx, N);
				free(sorted);
				return count;
			}
		}
	}

	return (unsigned int) ceilf(curand_uniform_double(&states[curand_idx])*(double) N) % N;
}
/* end array functions */

/* Start functions around random number generation */
// Set random seed for host
void set_seed( void ) {
        timespec seed;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &seed);
        srand(seed.tv_nsec);
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

void print_progress(int curr_amount, int num_vals) {
	char format[40] = { '\0' };
	double progress = (curr_amount / (double) num_vals) * 100.0f;
	int num_dig = numDigits(progress);
	int limit = num_dig+10;
	if (trunc(progress*1000.0f) >= 9995 && num_dig == 1) limit += 1;
	else if (trunc(progress*1000.0f) >= 99995 && num_dig == 2) limit += 1;
	for (int k = 0; k < limit; k++) strcat(format, "\b");
	strcat(format, "%.2f/%.2f");
	printf(format, progress, 100.0f);
}

#include "../gene_expression_nn.h"
#include "../cell.h"
__global__ void init_pde(double*, double, double, unsigned int);
__global__ void pde_space_step(double*, unsigned int, unsigned int,
			       double, double, double, double, double);
#include "../carcinogen_pde.h"

struct DataBlock {
	unsigned int dev_id_1, dev_id_2;
	Cell *prevGrid, *newGrid;
	CarcinogenPDE *pdes;

	unsigned int grid_size, cell_size;
	const unsigned int dim = 1024;
	unsigned int maxT;
	unsigned int maxt_tc_alive;

	clock_t start, end;
	clock_t start_step, end_step;
	unsigned int frame_rate, save_frames;
} d;

__global__ void copy_frame(uchar4*, unsigned char*);

void save_image(uchar4 *outputBitmap, size_t size, char *prefix, unsigned int time, unsigned int maxT) {
	char fname[150] = { '\0' };
	if (prefix != NULL) strcat(fname, prefix);
	int dig_max = numDigits(maxT); int dig = numDigits(time);
	for (int i = 0; i < dig_max-dig; i++) strcat(fname, "0");
	sprintf(&fname[strlen(fname)], "%d.png", time);
	unsigned char *frame;
	CudaSafeCall(cudaMallocManaged((void**)&frame, size*size*4*sizeof(unsigned char)));
	CudaSafeCall(cudaMemPrefetchAsync(frame, size*size*4*sizeof(unsigned char), 1, NULL));
	dim3 blocks(size/BLOCK_SIZE, size/BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	copy_frame<<< blocks, threads >>>(outputBitmap, frame);
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());

	unsigned error = lodepng_encode32_file(fname, frame, size, size);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	CudaSafeCall(cudaFree(frame));
}

void save_video(char *input_prefix, char *output_name, unsigned int frame_rate, unsigned int maxT) {
	char command[250] = { '\0' };
	sprintf(command, "ffmpeg -y -v quiet -framerate %d -start_number 0 -i ", frame_rate);
	int numDigMaxT = numDigits(maxT);
	if (input_prefix != NULL) strcat(command, input_prefix);
	if (numDigMaxT == 1) strcat(command, "%%d.png");
	else sprintf(&command[strlen(command)], "%%%d%dd.png", 0, numDigMaxT);
	strcat(command, " -c:v libx264 -pix_fmt yuv420p ");
	strcat(command, output_name);
	strcat(command, ".mp4");
	system(command);
}

#include "../gpu_anim.h"
GPUAnimBitmap bitmap(d.dim, d.dim, &d);

#include "../cuda_kernels.h"
#include "../anim_functions.h"
#include "../ca.h"

#endif // __GENERAL_H__
