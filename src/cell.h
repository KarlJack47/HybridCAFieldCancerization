#ifndef __CELL_H__
#define __CELL_H__

#include "common/general.h"

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

	GeneExpressionNN *NN;

	int chosen_phenotype;

	void initialize(int x, int y, int grid_size, int n_in, int n_out, double *W_x, double *W_y, double *b_y) {
		CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(GeneExpressionNN)));
		*NN = GeneExpressionNN(n_in, n_out);

		NN->memory_allocate(W_x, W_y, b_y, device);

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

		CudaSafeCall(cudaMallocManaged((void**)&gene_expressions, 2*n_out*sizeof(double)));
		memset(gene_expressions, 0.0f, 2*n_out*sizeof(double));

		age = 0;

		chosen_phenotype = -1;

	}

	Cell(int x, int y, int grid_size, int n_in, int n_out, int dev, double *W_x, double *W_y, double *b_y) {
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

		initialize(x, y, grid_size, n_in, n_out, W_x, W_y, b_y);

		phenotype[0] = phenotype_init[state*4];
		phenotype[1] = phenotype_init[state*4+1];
		phenotype[2] = phenotype_init[state*4+2];
		phenotype[3] = phenotype_init[state*4+3];
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(neighbourhood));
		CudaSafeCall(cudaFree(gene_expressions));
		NN->free_resources();
		CudaSafeCall(cudaFree(NN));
	}

	void prefetch_cell_params(int loc, int g_size) {
		int loc1 = loc;
		if (loc == -1) loc1 = cudaCpuDeviceId;
		NN->prefetch_nn_params(loc1);
		CudaSafeCall(cudaMemPrefetchAsync(NN, sizeof(NN), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(neighbourhood, 8*sizeof(int), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(gene_expressions, 2*NN->n_output*sizeof(double*), loc1, NULL));
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
				phenotype[i] = check;
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
				c->phenotype[i] = check;
		}
		for (i = 0; i < NN->n_output; i++) {
			c->gene_expressions[i*2] = gene_expressions[i*2];
			c->gene_expressions[i*2+1] = gene_expressions[i*2+1];
			c->NN->b_out[i] = NN->b_out[i];
		}
	}

	__device__ int proliferate(Cell *c, int cell, curandState_t *states) {
		int idx = (int) ceilf(curand_uniform_double(&states[cell]) * (double) NN->n_output) % NN->n_output;
		int new_state = -1;

		if ((state != 4 || state != 5) && c->state != 6) return -2;
		if ((state == 4 || state == 5) && (c->state == 5 || c->state == 4)) return -2;

		if ((state == 4 && fabs(NN->b_out[2] - BIAS) <= FLT_EPSILON && c->state == 6) ||
		    (state == 4 && fabs(NN->b_out[2] - BIAS) <= FLT_EPSILON && c->state != 6 && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
		    (state == 5 && c->state != 6 && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
		     state != 4) {
			if (!(fabsf(NN->b_out[idx] - BIAS) <= FLT_EPSILON)) new_state = state;
			else {
				if (curand_uniform_double(&states[cell]) <= 0.5f) {
					change_state(state_mut_map[state*NN->n_output+idx]);
					new_state = prolif_mut_map[state*NN->n_output+idx];
				} else {
					new_state = prolif_mut_map[state*NN->n_output+idx];
					change_state(state_mut_map[state*NN->n_output+idx]);
				}
			}
			if (c->state != 6 && (state == 4 || state == 5)) c->apoptosis();
			c->change_state(new_state);
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int differentiate(Cell *c, int cell, curandState_t *states) {
		int idx = (int) ceilf(curand_uniform_double(&states[cell]) * (double) NN->n_output) % NN->n_output;
		int new_state = -1;

		if ((state != 4 || state != 5) && c->state != 6) return -2;
		if ((state == 4 || state == 5) && (c->state == 5 || c->state == 4)) return -2;

		if (((state == 4 || state == 5) && c->state != 6 && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
		    c->state == 6) {
			if (!(fabsf(NN->b_out[idx] - BIAS) <= FLT_EPSILON)) new_state = diff_mut_map[state*(NN->n_output+1)];
			else {
				if (curand_uniform_double(&states[cell]) <= 0.5f) {
					change_state(state_mut_map[state*NN->n_output+idx]);
					new_state = diff_mut_map[state*(NN->n_output+1)+(idx+1)];
				} else {
					new_state = diff_mut_map[state*(NN->n_output+1)+(idx+1)];
					change_state(state_mut_map[state*NN->n_output+idx]);
				}
			}
			if (new_state == -1) return new_state;
			if (c->state != 6 && (state == 4 || state == 5)) c->apoptosis();
			c->change_state(new_state);
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int move(Cell *c, int cell, curandState_t *states) {
		if ((state != 4 || state != 5) && c->state != 6) return -2;
		if ((state == 4 || state == 5) && (c->state == 5 || c->state == 4)) return -2;

		if (curand_uniform_double(&states[cell]) <= CHANCE_MOVE) {
			if ((state == 5 && c->state != 6 && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
			     c->state == 6) {
				if (c->state != 6 && (state == 4 || state == 5)) c->apoptosis();
				c->change_state(state);
				copy_mutations(c);
				c->age = age;
				apoptosis();
			}
		}

		return 0;
	}

	__device__ void apoptosis(void) {
		state = 6;
		int i;
		for (i = 0; i < 4; i++)
			phenotype[i] = phenotype_init[6*4+i];
		for (i = 0; i < NN->n_output; i++) {
			gene_expressions[i*2] = 0.0f;
			gene_expressions[i*2+1] = 0.0f;
			NN->b_out[i] = 0.0f;
		}
		age = 0;
	}

	__device__ void phenotype_mutate(int M, int cell, curandState_t *states) {
		int i;
		if (!(fabsf(NN->b_out[M] - BIAS) <= FLT_EPSILON)) return;
		if (!(curand_uniform_double(&states[cell]) <= CHANCE_PHENO_MUT)) return;
		if (!(fabsf(gene_expressions[M*2] - gene_expressions[M*2+1]) <= FLT_EPSILON) && fmaxf(gene_expressions[M*2], gene_expressions[M*2+1]) >= MUT_THRESHOLD) {
			// down-regulation
			if (gene_expressions[M*2] < gene_expressions[M*2+1]) {
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

	__device__ void mutate(double *result, int cell, curandState_t *states) {
		int i;

		if (state != 6) {
			for (i = 0; i < NN->n_output; i++) {
				if (curand_uniform_double(&states[cell]) <= CHANCE_UPREG)
					gene_expressions[i*2] += result[i];
				else
					gene_expressions[i*2+1] -= result[i];
			}
			NN->mutate(gene_expressions);

			for (i = 0; i < NN->n_output; i++) phenotype_mutate(i, cell, states);

			age++;
		}
	}
};

#endif // __CELL_H__
