#ifndef __CELL_H__
#define __CELL_H__

#include "mutation_nn.h"

#define MUT_THRESHOLD 0.1f
#define PHENOTYPE_INCR 0.01f
#define CHANCE_MOVE 0.05f

// "NC": 0
// "MNC": 1
// "SC": 2
// "MSC": 3
// "CSC": 4
// "TC": 5
// "Empty": 6

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

__device__ int get_rand_idx(double *L, const int N, int cell, curandState_t *states, int *idx=NULL) {
	double *sorted = (double*)malloc(N*sizeof(double));
	int *idx_t;
	if (idx==NULL) idx_t = (int*)malloc(N*sizeof(int));
	for (int i = 0; i < N; i++) sorted[i] = 1000000000.0f * L[i];
	bitonic_sort(sorted, 0, N, false);
	double sum = 0;
	for (int i = 0; i < N; i++) sum += sorted[i];
	double rnd = curand_uniform_double(&states[cell]) * sum;
	for (int i = 0; i < N; i++) {
		rnd -= sorted[i];
		if (rnd < 0) {
			if (idx==NULL) {
				int count = get_indexes(sorted[i] / 1000000000.0f, L, idx_t, N);
				int chosen = idx_t[(int) ceilf(curand_uniform_double(&states[cell])*(double) count) % count];
				free(sorted); free(idx_t);
				return chosen;
			} else {
				int count = get_indexes(sorted[i] / 1000000000.0f, L, idx, N);
				free(sorted);
				return count;
			}
		}
	}

	return (int) ceilf(curand_uniform_double(&states[cell])*(double) N) % N;
}

struct Cell {
	int state;
	int age;
	int device;
	int *neighbourhood;
	double *phenotype;
	double *mutations;
	double *consumption;

	double *W_y_init;
	MutationNN *NN;

	void initialize(int x, int y, int grid_size, int n_in, int n_out, double *carcinogen_mutation_map) {
		CudaSafeCall(cudaMemPrefetchAsync(index_map, 11*12*sizeof(int), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(upreg_phenotype_map, 11*4*sizeof(double), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(downreg_phenotype_map, 11*4*sizeof(double), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(phenotype_init, 7*4*sizeof(double), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(state_mut_map, 6*11*sizeof(int), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(prolif_mut_map, 6*11*sizeof(int), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(diff_mut_map, 6*11*sizeof(int), device, NULL));

		CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(MutationNN)));
		NN[0] = MutationNN(n_in, n_out);
		int n_input = NN->n_input; int n_hidden = NN->n_hidden; int n_output = NN->n_output;

		double *W_x = (double*)malloc(n_hidden*n_input*sizeof(double));
                double *b_x = (double*)malloc(n_hidden*sizeof(double));
                double *W_y = (double*)malloc(n_hidden*n_output*sizeof(double));
                double *b_y = (double*)malloc(n_output*sizeof(double));

		memset(W_x, 0.0f, n_hidden*n_input*sizeof(double));

		for (int i = 0; i < n_hidden; i++) {
			for (int j = 0; j < n_input; j++) {
				if (i == 0 && j == n_input-1) W_x[i*n_input+j] = -0.1f;
				else if (i == 0) W_x[i*n_input+j] = -1.0f;
				else {
					if (j != n_input-1) {
						W_x[i*n_input+j] = carcinogen_mutation_map[j*(n_input-1)+(i-1)];
					}
					else W_x[i*n_input+j] = 0.01f;
				}
			}
		}

		memset(b_x, 0.0f, n_hidden*sizeof(double));

		memset(W_y, 0.0f, n_hidden*n_output*sizeof(double));

		W_y[0] = 0.5f;
		W_y[n_output+1] = 0.25f;
		W_y[2*n_output+2] = 0.25f;
		W_y[3*n_output+3] = 0.167f;
		W_y[4*n_output+4] = 0.125f;
		W_y[5*n_output+5] = 0.25f;
		W_y[6*n_output+6] = 0.25f;
		W_y[7*n_output+7] = 0.125f;
		W_y[8*n_output+8] = 0.167f;
		W_y[9*n_output+9] = 0.25f;
		W_y[10*n_output+10] = 0.167f;
		for (int i = 1; i < n_hidden; i++) {
			W_y[i*n_output] = -1.45f;
		}
		for (int i = 2; i < n_hidden; i++) {
			W_y[i*n_output+1] = 0.01f;
		}
		W_y[3*n_output+1] = 0.02f;
		W_y[n_output+3] = 0.02f;
		W_y[7*n_output+3] = 0.01f;
		W_y[4*n_output+7] = 0.01f;
		W_y[4*n_output+8] = 0.01f;
		W_y[10*n_output+8] = 0.02f;
		W_y[7*n_output+10] = 0.01f;
		W_y[8*n_output+10] = 0.02f;

		memset(b_y, 0.0f, n_output*sizeof(double));

		NN->memory_allocate(W_x, b_x, W_y, b_y, device);
		CudaSafeCall(cudaMemPrefetchAsync(NN, sizeof(NN), device, NULL));

		free(W_x);
		free(b_x);
		free(b_y);
		CudaSafeCall(cudaMallocManaged((void**)&W_y_init, n_hidden*n_output*sizeof(double)));
		memcpy(W_y_init, W_y, n_hidden*n_output*sizeof(double));
		free(W_y);
		CudaSafeCall(cudaMemPrefetchAsync(W_y_init, n_hidden*n_output*sizeof(double), device, NULL));

		CudaSafeCall(cudaMallocManaged((void**)&phenotype, 4*sizeof(double)));

		CudaSafeCall(cudaMallocManaged((void**)&neighbourhood, 8*sizeof(int)));
		neighbourhood[0] = x + ((y+1) % grid_size) * grid_size; // n
		neighbourhood[1] = ((x+1) % grid_size) + y * grid_size; // e
		neighbourhood[2] = x + abs((y-1) % grid_size) * grid_size; // s
		neighbourhood[3] = abs((x-1) % grid_size) + y * grid_size; // w
		neighbourhood[4] = ((x+1) % grid_size) + ((y+1) % grid_size) * grid_size; // ne
		neighbourhood[5] = ((x+1) % grid_size) + abs((y-1) % grid_size) * grid_size; // se
		neighbourhood[6] = abs((x-1) % grid_size) + abs((y-1) % grid_size) * grid_size; // sw
		neighbourhood[7] = abs((x-1) % grid_size) + ((y+1) % grid_size) * grid_size; // nw
		CudaSafeCall(cudaMemPrefetchAsync(neighbourhood, 8*sizeof(int), device, NULL));

		CudaSafeCall(cudaMallocManaged((void**)&mutations, n_output*sizeof(double)));
		memset(mutations, 0.0f, n_output*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(mutations, n_output*sizeof(double), device, NULL));

		CudaSafeCall(cudaMallocManaged((void**)&consumption, (n_input-1)*sizeof(double)));
		memset(consumption, 0.0f, (n_input-1)*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(consumption, (n_input-1)*sizeof(double), device, NULL));

		age = 0;

	}

	Cell(int x, int y, int grid_size, int n_in, int n_out, double *carcin_map, int dev) {
		device = dev;
		set_seed();
		int init_states[3] = {0, 2, 6};
		double weight_states[3] = {0.70f, 0.01f, 0.29f};

		double check = rand() / (double) RAND_MAX;

		if (fabsf(check) > weight_states[2])
			state = init_states[0];
		else if (fabsf(check) > weight_states[1] && fabsf(check) < weight_states[0])
			state = init_states[2];
		else
			state = init_states[1];

		initialize(x, y, grid_size, n_in, n_out, carcin_map);

		phenotype[0] = phenotype_init[state*4];
		phenotype[1] = phenotype_init[state*4+1];
		phenotype[2] = phenotype_init[state*4+2];
		phenotype[3] = phenotype_init[state*4+3];
		CudaSafeCall(cudaMemPrefetchAsync(phenotype, 4*sizeof(double), device, NULL));
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(neighbourhood));
		CudaSafeCall(cudaFree(mutations));
		CudaSafeCall(cudaFree(consumption));
		CudaSafeCall(cudaFree(W_y_init));
		NN->free_resources();
		CudaSafeCall(cudaFree(NN));
	}

	__device__ void change_state(int new_state) {
		for (int i = 0; i < 4; i++) {
			double check = phenotype_init[new_state*4+i] + phenotype[i] - phenotype_init[state*4+i];
			if (check < 0.0f)
				phenotype[i] = 0.0f;
			else if (check > 1.0f)
				phenotype[i] = 1.0f;
			else
				phenotype[i] = phenotype_init[new_state*4+i] + phenotype[i] - phenotype_init[state*4+i];
		}
		state = new_state;
	}

	__device__ int get_phenotype(int cell, curandState_t *states) {
		if (state != 6) {
			return get_rand_idx(phenotype, 4, cell, states);
		} else return -1;
	}

	__device__ void copy_mutations(Cell *c) {
		for (int i = 0; i < 4; i++) {
			double check = c->phenotype[i] + phenotype[i] - phenotype_init[state*4+i];
			if (check < 0.0f)
				c->phenotype[i] = 0.0f;
			else if (check > 1.0f)
				c->phenotype[i] = 1.0f;
			else
				c->phenotype[i] += phenotype[i] - phenotype_init[state*4+i];
		}
		for (int i = 0; i < NN->n_hidden*NN->n_output; i++)
			c->NN->W_out[i] = NN->W_out[i];
		for (int i = 0; i < 11; i++)
			c->mutations[i] = mutations[i];
	}

	__device__ int proliferate(Cell *c, int cell, curandState_t *states) {
		int idx = get_rand_idx(mutations, 11, cell, states);
		int new_state = -1;
		if ((state == 4 && mutations[3] >= MUT_THRESHOLD) || state != 4) {
			if (mutations[idx] < MUT_THRESHOLD) new_state = prolif_mut_map[state*11];
			else {
				if ((int) ceilf(curand_uniform_double(&states[cell])*2.0f) % 2 == 0) {
					change_state(state_mut_map[state*11+idx]);
					new_state = prolif_mut_map[state*11+idx];
				} else {
					new_state = prolif_mut_map[state*11+idx];
					change_state(state_mut_map[state*11+idx]);
				}
			}
			c->change_state(new_state);
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int differentiate(Cell *c, int cell, curandState_t *states) {
		int idx = get_rand_idx(mutations, 11, cell, states);
		int new_state = -1;
		if (mutations[idx] < MUT_THRESHOLD) new_state = diff_mut_map[state*11];
		else {
			if (curand_uniform_double(&states[cell]) <= 0.5f) {
				change_state(state_mut_map[state*11+idx]);
				new_state = diff_mut_map[state*11+idx];
			} else {
				new_state = diff_mut_map[state*11+idx];
				change_state(state_mut_map[state*11+idx]);
			}
		}
		if (new_state == -1) return new_state;
		c->change_state(new_state);
		copy_mutations(c);

		return new_state;
	}

	__device__ void move(Cell *c, int cell, curandState_t *states) {
		if (curand_uniform_double(&states[cell]) <= CHANCE_MOVE) {
			c->change_state(state);
			copy_mutations(c);
			c->age = age;
			apoptosis();
		}
	}

	__device__ void apoptosis(void) {
		state = 6;
		for (int i = 0; i < 4; i++) {
			phenotype[i] = phenotype_init[6*4+i];
		}
		for (int i = 0; i < NN->n_hidden*NN->n_output; i++) {
			NN->W_out[i] = W_y_init[i];
		}
		for (int i = 0; i < 11; i++) {
			mutations[i] = 0.0f;
		}
		age = 0;
	}

	__device__ void phenotype_mutate(int M, double *prevMut, double *newMut) {
		if (newMut[M] >= MUT_THRESHOLD) {
			// down-regulation
			if (prevMut[M] > newMut[M]) {
				for (int j = 0; j < 4; j++) {
					if (downreg_phenotype_map[M*4+j] < 0.0f)
						phenotype[j] = fmaxf(0.0f, phenotype[j] + downreg_phenotype_map[M*4+j]);
					else
						phenotype[j] = fminf(phenotype[j] + downreg_phenotype_map[M*4+j], 1.0f);
				}
			// up-regulation
			} else {
				for (int j = 0; j < 4; j++) {
					if (upreg_phenotype_map[M*4+j] < 0.0f)
						phenotype[j] = fmaxf(0.0f, phenotype[j] + upreg_phenotype_map[M*4+j]);
					else
						phenotype[j] = fminf(phenotype[j] + upreg_phenotype_map[M*4+j], 1.0f);
				}
			}
		}
	}

	__device__ void update_consumption(int idx, double *in, double *prevMut, double *newMut, int cell, curandState_t *states) {
		int M_idx[11];
		for (int i = 0; i < NN->n_input; i++) NN->input[i] = 0.0f;
		consumption[idx] = 0.0f;
		NN->input[idx] = in[idx];
		NN->evaluate();
		int count = get_rand_idx(NN->output, 11, cell, states, M_idx);
		for (int i = 0; i < count; i++) consumption[idx] += fabsf(newMut[M_idx[i]] - prevMut[M_idx[i]]);
	}

	__device__ void mutate(double *in, double *result, int cell, curandState_t *states) {
		int M_idx[11];
		int count = get_rand_idx(result, 11, cell, states, M_idx);

		for (int i = 0; i < count; i++) {
			int M = M_idx[i];
			if (M != 0 && state != 6) {
				double prevMut[11];
				for (int j = 0; j < NN->n_output; j++) prevMut[j] = mutations[j];
				NN->mutate(M, index_map, mutations, cell, states);
				phenotype_mutate(M, prevMut, mutations);
				for (int j = 0; j < NN->n_input-1; j++) update_consumption(j, in, prevMut, mutations, cell, states);
			}
		}

		if (state != 6) {
			// Random mutations
			if ((int) ceilf(curand_uniform_double(&states[cell])*1000.0f) <= 10) {
				int num = (int) ceilf(curand_uniform_double(&states[cell])*4.0f) % 4;
				for (int i = 0; i < num; i++) {
					int idx = (int) ceilf(curand_uniform_double(&states[cell])*11.0f) % 11;
					double prevMut[11];
					for (int j = 0; j < NN->n_output; j++) prevMut[j] = mutations[j];
					NN->mutate(idx, index_map, mutations, cell, states);
					for (int j = 0; j < NN->n_input; j++) NN->input[j] = in[j];
					NN->evaluate();
					count = get_rand_idx(NN->output, 11, cell, states, M_idx);
					for (int k = 0; k < count; k++) {
						int M = M_idx[k];
						if (M != 0 && state != 6) {
							phenotype_mutate(M, prevMut, mutations);
							for (int l = 0; l < NN->n_input-1; l++) update_consumption(l, in, prevMut, mutations, cell, states);
						}
					}
				}
			}
		}

		if (state != 6) age++;
	}
};

#endif // __CELL_H__
