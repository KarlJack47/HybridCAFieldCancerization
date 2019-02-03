#ifndef __CELL_H__
#define __CELL_H__

#include "common/general.h"

#define MUT_THRESHOLD 0.1f // possibly decrease to 0.05
#define CHANCE_MOVE 0.05f // possibly increase to 0.25 or higher

// "NC": 0
// "MNC": 1
// "SC": 2
// "MSC": 3
// "CSC": 4
// "TC": 5
// "Empty": 6


__device__ int get_rand_idx(double *L, const int N, int cell, curandState_t *states, int *idx=NULL) {
	double *sorted = (double*)malloc(N*sizeof(double));
	int *idx_t; int i;
	if (idx==NULL) idx_t = (int*)malloc(N*sizeof(int));
	for (i = 0; i < N; i++) sorted[i] = 1000000000.0f * L[i];
	bitonic_sort(sorted, 0, N, false);
	double sum = 0.0f;
	for (i = 0; i < N; i++) sum += sorted[i];
	double rnd = curand_uniform_double(&states[cell]) * sum;
	for (i = 0; i < N; i++) {
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
	int device;
	int state;
	int age;
	int *neighbourhood;
	double *phenotype;
	double *gene_expressions;

	double *W_y_init;
	GeneExpressionNN *NN;

	int chosen_phenotype;

	void initialize(int x, int y, int grid_size, int n_in, int n_out) {
		CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(GeneExpressionNN)));
		*NN = GeneExpressionNN(n_in, n_out);
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
					if (j != n_input-1)
						W_x[i*n_input+j] = carcinogen_mutation_map[j*(n_input-1)+(i-1)];
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
		for (int i = 1; i < n_hidden; i++)
			W_y[i*n_output] = -1.45f;
		for (int i = 2; i < n_hidden; i++)
			W_y[i*n_output+1] = 0.01f;
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

		free(W_x);
		free(b_x);
		free(b_y);
		CudaSafeCall(cudaMallocManaged((void**)&W_y_init, n_hidden*n_output*sizeof(double)));
		memcpy(W_y_init, W_y, n_hidden*n_output*sizeof(double));
		free(W_y);

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

		CudaSafeCall(cudaMallocManaged((void**)&gene_expressions, n_output*sizeof(double)));
		memset(gene_expressions, 0.0f, n_output*sizeof(double));

		age = 0;

		chosen_phenotype = -1;

	}

	Cell(int x, int y, int grid_size, int n_in, int n_out, int dev) {
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

		initialize(x, y, grid_size, n_in, n_out);

		phenotype[0] = phenotype_init[state*4];
		phenotype[1] = phenotype_init[state*4+1];
		phenotype[2] = phenotype_init[state*4+2];
		phenotype[3] = phenotype_init[state*4+3];
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(neighbourhood));
		CudaSafeCall(cudaFree(gene_expressions));
		CudaSafeCall(cudaFree(W_y_init));
		NN->free_resources();
		CudaSafeCall(cudaFree(NN));
	}

	void prefetch_cell_params(int loc, int g_size) {
		int loc1 = loc;
		if (loc == -1) loc1 = cudaCpuDeviceId;
		int n_hidden = NN->n_hidden; int n_output = NN->n_output;
		NN->prefetch_nn_params(loc1);
		CudaSafeCall(cudaMemPrefetchAsync(NN, sizeof(NN), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(W_y_init, n_hidden*n_output*sizeof(double), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(neighbourhood, 8*sizeof(int), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(gene_expressions, n_output*sizeof(double), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(phenotype, 4*sizeof(double), loc1, NULL));
	}

	__device__ void change_state(int new_state) {
		int i;
		for (i = 0; i < 4; i++) {
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
		if (state != 6)
			return get_rand_idx(phenotype, 4, cell, states);
		else return -1;
	}

	__device__ void copy_mutations(Cell *c) {
		int i;
		for (i = 0; i < 4; i++) {
			double check = c->phenotype[i] + phenotype[i] - phenotype_init[state*4+i];
			if (check < 0.0f)
				c->phenotype[i] = 0.0f;
			else if (check > 1.0f)
				c->phenotype[i] = 1.0f;
			else
				c->phenotype[i] += phenotype[i] - phenotype_init[state*4+i];
		}
		for (i = 0; i < NN->n_hidden*NN->n_output; i++)
			c->NN->W_out[i] = NN->W_out[i];
		for (i = 0; i < 11; i++)
			c->gene_expressions[i] = gene_expressions[i];
	}

	__device__ int proliferate(Cell *c, int cell, curandState_t *states) {
		int idx = get_rand_idx(gene_expressions, 11, cell, states);
		int new_state = -1;

		if (state != 5 && c->state != 6) return -2;

		if ((state == 4 && gene_expressions[3] >= MUT_THRESHOLD) || state != 4) {
			if (gene_expressions[idx] < MUT_THRESHOLD) new_state = prolif_mut_map[state*11];
			else {
				if ((int) ceilf(curand_uniform_double(&states[cell])*2.0f) % 2 == 0) {
					change_state(state_mut_map[state*11+idx]);
					new_state = prolif_mut_map[state*11+idx];
				} else {
					new_state = prolif_mut_map[state*11+idx];
					change_state(state_mut_map[state*11+idx]);
				}
			}
			if (c->state != 6 && state == 5) c->apoptosis();
			c->change_state(new_state);
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int differentiate(Cell *c, int cell, curandState_t *states) {
		int idx = get_rand_idx(gene_expressions, 11, cell, states);
		int new_state = -1;

		if (state != 5 && c->state != 6) return -2;

		if (gene_expressions[idx] < MUT_THRESHOLD) new_state = diff_mut_map[state*11];
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

	__device__ int move(Cell *c, int cell, curandState_t *states) {
		if (state != 5 && c->state != 6) return -2;

		if (curand_uniform_double(&states[cell]) <= CHANCE_MOVE) {
			c->change_state(state);
			copy_mutations(c);
			c->age = age;
			apoptosis();
		}

		return 0;
	}

	__device__ void apoptosis(void) {
		state = 6;
		int i;
		for (i = 0; i < 4; i++)
			phenotype[i] = phenotype_init[6*4+i];
		for (i = 0; i < NN->n_hidden*NN->n_output; i++)
			NN->W_out[i] = W_y_init[i];
		for (i = 0; i < 11; i++)
			gene_expressions[i] = 0.0f;
		age = 0;
	}

	__device__ void phenotype_mutate(int M, double *prevMut, double *newMut) {
		int i;
		if (newMut[M] >= MUT_THRESHOLD) {
			// down-regulation
			if (prevMut[M] > newMut[M]) {
				for (i = 0; i < 4; i++) {
					if (downreg_phenotype_map[M*4+i] < 0.0f)
						phenotype[i] = fmaxf(0.0f, phenotype[i] + downreg_phenotype_map[M*4+i]);
					else
						phenotype[i] = fminf(phenotype[i] + downreg_phenotype_map[M*4+i], 1.0f);
				}
			// up-regulation
			} else {
				for (i = 0; i < 4; i++) {
					if (upreg_phenotype_map[M*4+i] < 0.0f)
						phenotype[i] = fmaxf(0.0f, phenotype[i] + upreg_phenotype_map[M*4+i]);
					else
						phenotype[i] = fminf(phenotype[i] + upreg_phenotype_map[M*4+i], 1.0f);
				}
			}
		}
	}

	__device__ void mutate(double *in, double *result, int cell, curandState_t *states) {
		int i, j, k;

		if (state != 6) {
			int M_idx[11];
			M_idx[0] = -1;
			int count = get_rand_idx(result, 11, cell, states, M_idx);
			if (M_idx[0] == -1) { M_idx[0] = count; count = 1; }
			for (i = 0; i < count; i++) {
				int M = M_idx[i];
				if (M != 0) {
					double prevMut[11];
					for (j = 0; j < NN->n_output; j++) prevMut[j] = gene_expressions[j];
					NN->mutate(M, gene_expressions, cell, states);
					phenotype_mutate(M, prevMut, gene_expressions);
				}
			}

			// Random mutations
			if ((int) ceilf(curand_uniform_double(&states[cell])*1000.0f) <= 10) {
				int num = (int) ceilf(curand_uniform_double(&states[cell])*4.0f) % 4;
				for (i = 0; i < num; i++) {
					int idx = (int) ceilf(curand_uniform_double(&states[cell])*11.0f) % 11;
					double prevMut[11];
					for (j = 0; j < NN->n_output; j++) prevMut[j] = gene_expressions[j];
					NN->mutate(idx, gene_expressions, cell, states);
					for (j = 0; j < NN->n_input; j++) NN->input[j] = in[j];
					NN->evaluate();
					M_idx[0] = -1;
					count = get_rand_idx(NN->output, 11, cell, states, M_idx);
					if (M_idx[0] == -1) { M_idx[0] = count; count = 1; }
					for (k = 0; k < count; k++) {
						int M = M_idx[k];
						if (M != 0 && state != 6) {
							phenotype_mutate(M, prevMut, gene_expressions);
						}
					}
				}
			}

			age++;
		}
	}
};

#endif // __CELL_H__
