#ifndef __CELL_H__
#define __CELL_H__

#include "common/general.h"

struct Cell {
	int device;
	unsigned int state;
	unsigned int age;
	unsigned int *neighbourhood;
	double *phenotype;
	double *gene_expressions;

	GeneExpressionNN *NN;

	int chosen_phenotype;

	void initialize(unsigned int x, unsigned int y, unsigned int grid_size, double *W_x, double *W_y, double *b_y) {
		CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(GeneExpressionNN)));
		*NN = GeneExpressionNN(NUM_CARCIN+1, NUM_GENES);

		NN->memory_allocate(W_x, W_y, b_y, device);

		CudaSafeCall(cudaMallocManaged((void**)&phenotype, NUM_PHENO*sizeof(double)));

		CudaSafeCall(cudaMallocManaged((void**)&neighbourhood, NUM_NEIGH*sizeof(int)));
		neighbourhood[NORTH] = x + ((y+1) % grid_size) * grid_size;
		neighbourhood[EAST] = ((x+1) % grid_size) + y * grid_size;
		neighbourhood[SOUTH] = x + abs((int) (((int) y-1) % grid_size)) * grid_size;
		neighbourhood[WEST] = abs((int) (((int) x-1) % grid_size)) + y * grid_size;
		neighbourhood[NORTH_EAST] = ((x+1) % grid_size) + ((y+1) % grid_size) * grid_size;
		neighbourhood[SOUTH_EAST] = ((x+1) % grid_size) + abs((int)(((int) y-1) % grid_size)) * grid_size;
		neighbourhood[SOUTH_WEST] = abs((int) (((int) x-1) % grid_size)) + abs((int) (((int) y-1) % grid_size)) * grid_size;
		neighbourhood[NORTH_WEST] = abs((int) (((int) x-1) % grid_size)) + ((y+1) % grid_size) * grid_size;

		CudaSafeCall(cudaMallocManaged((void**)&gene_expressions, 2*NUM_GENES*sizeof(double)));
		memset(gene_expressions, 0.0f, 2*NUM_GENES*sizeof(double));

		age = 0;

		chosen_phenotype = -1;

	}

	Cell(unsigned int x, unsigned int y, unsigned int grid_size, unsigned int dev, double *W_x, double *W_y, double *b_y) {
		device = dev;
		set_seed();
		unsigned int init_states[3] = {0, 2, 6};
		double weight_states[3] = {0.70f, 0.01f, 0.29f};

		double check = rand() / (double) RAND_MAX;

		if (fabsf(check) > weight_states[2])
			state = init_states[0];
		else if (fabsf(check) > weight_states[1] && fabsf(check) < weight_states[0])
			state = init_states[2];
		else
			state = init_states[1];

		initialize(x, y, grid_size, W_x, W_y, b_y);

		phenotype[0] = phenotype_init[state*NUM_PHENO];
		phenotype[1] = phenotype_init[state*NUM_PHENO+1];
		phenotype[2] = phenotype_init[state*NUM_PHENO+2];
		phenotype[3] = phenotype_init[state*NUM_PHENO+3];
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(phenotype));
		CudaSafeCall(cudaFree(neighbourhood));
		CudaSafeCall(cudaFree(gene_expressions));
		NN->free_resources();
		CudaSafeCall(cudaFree(NN));
	}

	void prefetch_cell_params(int loc, unsigned int g_size) {
		int loc1 = loc;
		if (loc == -1) loc1 = cudaCpuDeviceId;
		NN->prefetch_nn_params(loc1);
		CudaSafeCall(cudaMemPrefetchAsync(NN, sizeof(NN), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(neighbourhood, NUM_NEIGH*sizeof(unsigned int), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(gene_expressions, 2*NUM_GENES*sizeof(double), loc1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(phenotype, NUM_PHENO*sizeof(double), loc1, NULL));
	}

	__device__ void change_state(unsigned int new_state) {
		unsigned int i;
		for (i = 0; i < 4; i++) {
			double check = phenotype_init[new_state*NUM_PHENO+i] + phenotype[i] - phenotype_init[state*NUM_PHENO+i];
			if (check < 0.0f)
				phenotype[i] = 0.0f;
			else if (check > 1.0f)
				phenotype[i] = 1.0f;
			else
				phenotype[i] = check;
		}
		state = new_state;
	}

	__device__ int get_phenotype(unsigned int cell, curandState_t *states) {
		if (state != EMPTY)
			return get_rand_idx(phenotype, NUM_PHENO, cell, states);
		else return -1;
	}

	__device__ void copy_mutations(Cell *c) {
		unsigned int i;
		for (i = 0; i < NUM_PHENO; i++) {
			double check = c->phenotype[i] + phenotype[i] - phenotype_init[state*NUM_PHENO+i];
			if (check < 0.0f)
				c->phenotype[i] = 0.0f;
			else if (check > 1.0f)
				c->phenotype[i] = 1.0f;
			else
				c->phenotype[i] = check;
		}
		for (i = 0; i < NUM_GENES; i++) {
			c->gene_expressions[i*2] = gene_expressions[i*2];
			c->gene_expressions[i*2+1] = gene_expressions[i*2+1];
			c->NN->b_out[i] = NN->b_out[i];
		}
	}

	__device__ unsigned int positively_mutated(unsigned int M) {
		if ((gene_type[M] == 0 && gene_expressions[M*2+1] >= MUT_THRESHOLD) || (gene_type[M] == 1 && gene_expressions[M*2] >= MUT_THRESHOLD))
			return 0;
		else
			return 1;
	}

	__device__ int proliferate(Cell *c, unsigned int cell, curandState_t *states) {
		int new_state = -1;

		if ((state != CSC || state != TC) && c->state != EMPTY) return -2;
		if ((state == CSC || state == TC) && (c->state == TC || c->state == CSC)) return -2;

		unsigned int idx = (unsigned int) ceilf(curand_uniform_double(&states[cell]) * (double) NUM_GENES) % NUM_GENES;

		if ((state == CSC && positively_mutated(CSC_GENE_IDX) == 0 && c->state == EMPTY) ||
		    (state == CSC && positively_mutated(CSC_GENE_IDX) == 0 && c->state != EMPTY && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
		    (state == TC && c->state != EMPTY && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
		     state != CSC) {
			if (positively_mutated(idx) == 1) new_state = state;
			else {
				if (curand_uniform_double(&states[cell]) <= 0.5f) {
					change_state(state_mut_map[state*NUM_GENES+idx]);
					new_state = prolif_mut_map[state*NUM_GENES+idx];
				} else {
					new_state = prolif_mut_map[state*NUM_GENES+idx];
					change_state(state_mut_map[state*NUM_GENES+idx]);
				}
			}
			if (c->state != EMPTY && (state == CSC || state == TC)) c->apoptosis();
			c->change_state(new_state);
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int differentiate(Cell *c, unsigned int cell, curandState_t *states) {
		int new_state = -1;

		if ((state != CSC || state != TC) && c->state != EMPTY) return -2;
		if ((state == CSC || state == TC) && (c->state == TC || c->state == CSC)) return -2;

		unsigned int idx = (unsigned int) ceilf(curand_uniform_double(&states[cell]) * (double) NUM_GENES) % NUM_GENES;

		if (((state == CSC || state == TC) && c->state != EMPTY && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
		    c->state == EMPTY) {
			if (positively_mutated(idx) == 1) new_state = diff_mut_map[state*(NUM_GENES+1)];
			else {
				if (curand_uniform_double(&states[cell]) <= 0.5f) {
					change_state(state_mut_map[state*NUM_GENES+idx]);
					new_state = diff_mut_map[state*(NUM_GENES+1)+(idx+1)];
				} else {
					new_state = diff_mut_map[state*(NUM_GENES+1)+(idx+1)];
					change_state(state_mut_map[state*NUM_GENES+idx]);
				}
			}
			if (new_state == -1) return new_state;
			if (c->state != EMPTY && (state == CSC || state == TC)) c->apoptosis();
			c->change_state(new_state);
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int move(Cell *c, unsigned int cell, curandState_t *states) {
		if ((state != CSC || state != TC) && c->state != EMPTY) return -2;
		if ((state == CSC || state == TC) && (c->state == CSC || c->state == TC)) return -2;

		if (curand_uniform_double(&states[cell]) <= CHANCE_MOVE) {
			if (((state == CSC || state == TC) && c->state != EMPTY && curand_uniform_double(&states[cell]) < CHANCE_KILL) ||
			     c->state == EMPTY) {
				if (c->state != EMPTY && (state == CSC || state == TC)) c->apoptosis();
				c->change_state(state);
				copy_mutations(c);
				c->age = age;
				apoptosis();
			}
		}

		return 0;
	}

	__device__ void apoptosis(void) {
		state = EMPTY;
		unsigned int i;
		for (i = 0; i < NUM_PHENO; i++)
			phenotype[i] = phenotype_init[(NUM_STATES-1)*NUM_PHENO+i];
		for (i = 0; i < NUM_GENES; i++) {
			gene_expressions[i*2] = 0.0f;
			gene_expressions[i*2+1] = 0.0f;
			NN->b_out[i] = 0.0f;
		}
		age = 0;
	}

	__device__ void phenotype_mutate(unsigned int M, unsigned int cell, curandState_t *states) {
		unsigned int i;
		if (!(fabsf(NN->b_out[M] - BIAS) <= FLT_EPSILON)) return;
		if (!(curand_uniform_double(&states[cell]) <= CHANCE_PHENO_MUT)) return;
		if (!(fabsf(gene_expressions[M*2] - gene_expressions[M*2+1]) <= FLT_EPSILON) && fmaxf(gene_expressions[M*2], gene_expressions[M*2+1]) >= MUT_THRESHOLD) {
			// down-regulation
			if (gene_expressions[M*2] < gene_expressions[M*2+1]) {
				for (i = 0; i < NUM_PHENO; i++) {
					if (downreg_phenotype_map[M*NUM_PHENO+i] < 0.0f)
						phenotype[i] = fmaxf(0.0f, phenotype[i] + downreg_phenotype_map[M*NUM_PHENO+i]);
					else
						phenotype[i] = fminf(phenotype[i] + downreg_phenotype_map[M*NUM_PHENO+i], 1.0f);
				}
			// up-regulation
			} else {
				for (i = 0; i < NUM_PHENO; i++) {
					if (upreg_phenotype_map[M*NUM_PHENO+i] < 0.0f)
						phenotype[i] = fmaxf(0.0f, phenotype[i] + upreg_phenotype_map[M*NUM_PHENO+i]);
					else
						phenotype[i] = fminf(phenotype[i] + upreg_phenotype_map[M*NUM_PHENO+i], 1.0f);
				}
			}
		}
	}

	__device__ void mutate(double *result, unsigned int cell, curandState_t *states) {
		unsigned int i;

		if (state != EMPTY) {
			for (i = 0; i < NUM_GENES; i++) {
				if (curand_uniform_double(&states[cell]) <= CHANCE_UPREG)
					gene_expressions[i*2] += result[i];
				else
					gene_expressions[i*2+1] += result[i];
			}
			NN->mutate(gene_expressions);

			for (i = 0; i < NUM_GENES; i++) phenotype_mutate(i, cell, states);

			age++;
		}
	}
};

#endif // __CELL_H__
