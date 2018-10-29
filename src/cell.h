#ifndef __CELL_H__
#define __CELL_H__

#include "mutation_nn.h"

// "NC": 0
// "MNC": 1
// "SC": 2
// "MSC": 3
// "CSC": 4
// "TC": 5
// "Empty": 6

struct Cell {
	int state;
	int age;
	float *phenotype;
	float *mutations;
	float consumption;

	MutationNN *NN;
	float *W_y_init;

	int *index_map;
	float *upreg_phenotype_map;
	float *downreg_phenotype_map;
	float *phenotype_init;
	int *state_mut_map;
	int *child_mut_map;

	void initialize(int n_in, int n_out, float *carcinogen_mutation_map) {
		int host_index_map[11*12] = {0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
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
		float host_upreg_phenotype_map[11*4] = {0.0, 0.0, 0.0, 0.0,
						        -0.01, 0.01, 0.01, 0.0,
						        0.0, 0.0, 0.01, 0.0,
						        -0.01, 0.01, 0.0, 0.0,
				    		        -0.01, 0.01, 0.0, -0.01,
						        -0.01, 0.0, 0.0, 0.0,
						        0.01, 0.0, 0.0, 0.0,
						        0.0, 0.0, -0.01, 0.0,
				    		        0.01, 0.0, -0.01, 0.01,
						        0.0, 0.0, -0.01, 0.0,
						        0.01, 0.0, -0.01, 0.01};
		float host_downreg_phenotype_map[11*4] = {0.0, 0.0, 0.0, 0.0,
						          0.01, -0.01, -0.01, 0.0,
						          0.0, 0.0, -0.01, 0.0,
						          0.01, -0.01, 0.0, 0.0,
				    		          0.01, -0.01, 0.0, 0.01,
						          0.01, 0.0, 0.0, 0.0,
						          -0.01, 0.0, 0.0, 0.0,
						          0.0, 0.0, 0.01, 0.0,
				    		          -0.01, 0.0, 0.01, -0.01,
						          0.0, 0.0, 0.01, 0.0,
						          -0.01, 0.0, 0.01, -0.01};
		float host_phenotype_init[7*4] = {0.1, 0.7, 0.3, 0.0,
						  0.2, 0.7, 0.15, 0.0,
						  0.1, 0.7, 0.075, 0.3,
						  0.1, 0.7, 0.0375, 0.4,
						  0.1, 0.7, 0.01875, 0.5,
						  0.6, 0.7, 0.009375, 0.0,
						  0.0, 0.0, 0.0, 0.0};
		int host_state_mut_map[6*11] = {0, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
						1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
						2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
						3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				  		4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
						5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
		int host_child_mut_map[6*11] = {0, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
						1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 4,
						2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
						3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
				  		4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4,
						5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
		CudaSafeCall(cudaMalloc((void**)&index_map, 11*12*sizeof(int)));
		CudaSafeCall(cudaMalloc((void**)&upreg_phenotype_map, 11*4*sizeof(float)));
		CudaSafeCall(cudaMalloc((void**)&downreg_phenotype_map, 11*4*sizeof(float)));
		CudaSafeCall(cudaMalloc((void**)&phenotype_init, 7*4*sizeof(float)));
		CudaSafeCall(cudaMalloc((void**)&state_mut_map, 6*11*sizeof(int)));
		CudaSafeCall(cudaMalloc((void**)&child_mut_map, 6*11*sizeof(int)));
		CudaSafeCall(cudaMemcpy(index_map, host_index_map, 11*12*sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(upreg_phenotype_map, host_upreg_phenotype_map, 11*4*sizeof(float), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(downreg_phenotype_map, host_downreg_phenotype_map, 11*4*sizeof(float), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(phenotype_init, host_phenotype_init, 7*4*sizeof(float), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(state_mut_map, host_state_mut_map, 6*11*sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(child_mut_map, host_child_mut_map, 6*11*sizeof(int), cudaMemcpyHostToDevice));

		NN=(MutationNN*)malloc(sizeof(MutationNN));
		NN[0] = MutationNN(n_in, n_out);
		int n_input = NN->n_input; int n_hidden = NN->n_hidden; int n_output = NN->n_output;

		float *W_x = (float*)malloc(n_hidden*n_input*sizeof(float));
                float *b_x = (float*)malloc(n_hidden*sizeof(float));
                float *W_y = (float*)malloc(n_hidden*n_output*sizeof(float));
                float *b_y = (float*)malloc(n_output*sizeof(float));

		memset(W_x, 0.0, n_hidden*n_input*sizeof(float));

		for (int i = 0; i < n_hidden; i++) {
			for (int j = 0; j < n_input; j++) {
				if (i == 0 && j == n_input-1) W_x[i*n_input+j] = -0.1f;
				else if (i == 0) W_x[i*n_input+j] = -1.0f;
				else {
					if (j != n_input-1) {
						W_x[i*n_input+j] = carcinogen_mutation_map[j*(n_input-1)+(i-1)];
					}
					else W_x[i*n_input+j] = 0.01;
				}
			}
		}

		memset(b_x, 0.0, n_hidden*sizeof(float));

		memset(W_y, 0.0, n_hidden*n_output*sizeof(float));

		W_y[0] = 0.5;
		W_y[n_output+1] = 0.25;
		W_y[2*n_output+2] = 0.25;
		W_y[3*n_output+3] = 0.167;
		W_y[4*n_output+4] = 0.125;
		W_y[5*n_output+5] = 0.25;
		W_y[6*n_output+6] = 0.25;
		W_y[7*n_output+7] = 0.125;
		W_y[8*n_output+8] = 0.167;
		W_y[9*n_output+9] = 0.25;
		W_y[10*n_output+10] = 0.167;
		for (int i = 1; i < n_hidden; i++) {
			W_y[i*n_output] = -1.45;
		}
		for (int i = 2; i < n_hidden; i++) {
			W_y[i*n_output+1] = 0.01;
		}
		W_y[3*n_output+1] = 0.02;
		W_y[n_output+3] = 0.02;
		W_y[7*n_output+3] = 0.01;
		W_y[4*n_output+7] = 0.01;
		W_y[4*n_output+8] = 0.01;
		W_y[10*n_output+8] = 0.02;
		W_y[7*n_output+10] = 0.01;
		W_y[8*n_output+10] = 0.02;

		memset(b_y, 0.0, n_output * sizeof(float));

		NN->memory_allocate(W_x, b_x, W_y, b_y);

		free(W_x);
		free(b_x);
		free(b_y);
		CudaSafeCall(cudaMalloc((void**)&W_y_init, n_hidden*n_output*sizeof(float)));
		CudaSafeCall(cudaMemcpy(W_y_init, W_y, n_hidden*n_output*sizeof(float), cudaMemcpyHostToDevice));
		free(W_y);

		CudaSafeCall(cudaMalloc((void**)&phenotype, 4*sizeof(float)));
		float phenotype_temp[4] = {host_phenotype_init[state*4], host_phenotype_init[state*4+1], host_phenotype_init[state*4+2], host_phenotype_init[state*4+3]};
		CudaSafeCall(cudaMemcpy(phenotype, phenotype_temp, 4*sizeof(float), cudaMemcpyHostToDevice));

		float mutations_temp[n_output];
		memset(mutations_temp, 0, n_output*sizeof(float));
		CudaSafeCall(cudaMalloc((void**)&mutations, n_output*sizeof(float)));
		CudaSafeCall(cudaMemcpy(mutations, mutations_temp, n_output*sizeof(float), cudaMemcpyHostToDevice));
		consumption = 0.0f;
		age = 0;

	}

	Cell(int n_in, int n_out, float *carcin_map) {
		set_seed();
		int init_states[3] = {0, 2, 6};
		float weight_states[3] = {0.70, 0.01, 0.29};

		float check = (float) rand() / (float) RAND_MAX;

		if (abs(check) > weight_states[2])
			state = init_states[0];
		else if (abs(check) > weight_states[1] && abs(check) < weight_states[0])
			state = init_states[2];
		else
			state = init_states[1];

		initialize(n_in, n_out, carcin_map);
	}

	void free_resources(void) {
		CudaSafeCall(cudaFree(mutations));
		CudaSafeCall(cudaFree(W_y_init));
		CudaSafeCall(cudaFree(index_map));
		CudaSafeCall(cudaFree(upreg_phenotype_map));
		CudaSafeCall(cudaFree(downreg_phenotype_map));
		CudaSafeCall(cudaFree(phenotype_init));
		CudaSafeCall(cudaFree(child_mut_map));
		NN->free_resources();
		free(NN);
	}

	void host_to_gpu_copy(int idx, Cell *dev_c) {
		Cell *l_c = (Cell*)malloc(sizeof(Cell));
		l_c->state = state;
		l_c->age = age;
                float *dev_phenotype;
                CudaSafeCall(cudaMalloc((void**)&dev_phenotype, 4*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_phenotype, phenotype, 4*sizeof(float), cudaMemcpyHostToDevice));
                l_c->phenotype = dev_phenotype;
                l_c->consumption = consumption;
                float *dev_mutations;
                CudaSafeCall(cudaMalloc((void**)&dev_mutations, NN->n_output*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_mutations, mutations, NN->n_output*sizeof(float), cudaMemcpyHostToDevice));
                l_c->mutations = dev_mutations;
                MutationNN *dev_NN;
                CudaSafeCall(cudaMalloc((void**)&dev_NN, sizeof(MutationNN)));
                CudaSafeCall(cudaMemcpy(dev_NN, NN, sizeof(MutationNN), cudaMemcpyHostToDevice));
                l_c->NN = dev_NN;
                float *dev_W_y_init;
                CudaSafeCall(cudaMalloc((void**)&dev_W_y_init, NN->n_output*NN->n_output*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_W_y_init, W_y_init, NN->n_output*NN->n_output*sizeof(float), cudaMemcpyHostToDevice));
                l_c->W_y_init = dev_W_y_init;
                int *dev_index_map;
                CudaSafeCall(cudaMalloc((void**)&dev_index_map, NN->n_output*(NN->n_output+1)*sizeof(int)));
                CudaSafeCall(cudaMemcpy(dev_index_map, index_map, NN->n_output*(NN->n_output+1)*sizeof(int), cudaMemcpyHostToDevice));
                l_c->index_map = dev_index_map;
                float *dev_upreg_phenotype_map;
                CudaSafeCall(cudaMalloc((void**)&dev_upreg_phenotype_map, NN->n_output*4*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_upreg_phenotype_map, upreg_phenotype_map, NN->n_output*4*sizeof(float), cudaMemcpyHostToDevice));
                l_c->upreg_phenotype_map = dev_upreg_phenotype_map;
		float *dev_downreg_phenotype_map;
                CudaSafeCall(cudaMalloc((void**)&dev_downreg_phenotype_map, NN->n_output*4*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_downreg_phenotype_map, upreg_phenotype_map, NN->n_output*4*sizeof(float), cudaMemcpyHostToDevice));
                l_c->downreg_phenotype_map = dev_downreg_phenotype_map;
                float *dev_phenotype_init;
                CudaSafeCall(cudaMalloc((void**)&dev_phenotype_init, 7*4*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_phenotype_init, phenotype_init, 7*4*sizeof(float), cudaMemcpyHostToDevice));
                l_c->phenotype_init = dev_phenotype_init;
		int *dev_state_mut_map;
		CudaSafeCall(cudaMalloc((void**)&dev_state_mut_map, 6*NN->n_output*sizeof(int)));
		CudaSafeCall(cudaMemcpy(dev_state_mut_map, state_mut_map, 6*NN->n_output*sizeof(int), cudaMemcpyHostToDevice));
		l_c->state_mut_map = dev_state_mut_map;
                int *dev_child_mut_map;
                CudaSafeCall(cudaMalloc((void**)&dev_child_mut_map, 6*NN->n_output*sizeof(int)));
                CudaSafeCall(cudaMemcpy(dev_child_mut_map, child_mut_map, 6*NN->n_output*sizeof(int), cudaMemcpyHostToDevice));
                l_c->child_mut_map = dev_child_mut_map;

		CudaSafeCall(cudaMemcpy(&dev_c[idx], &l_c[0], sizeof(Cell), cudaMemcpyHostToDevice));

		free(l_c);
	}

	__device__ void change_state(int new_state) {
		for (int i = 0; i < 4; i++) {
			if (phenotype_init[new_state*4+i] + abs(phenotype[i] - phenotype_init[state*4+i]) < 0)
				phenotype[i] = 0.0f;
			else if (phenotype_init[new_state*4+i] + abs(phenotype[i] - phenotype_init[state*4+i]) > 1)
				phenotype[i] = 1.0f;
			else
				phenotype[i] = phenotype_init[new_state*4+i] + abs(phenotype[i] - phenotype_init[state*4+i]);
		}
		state = new_state;
	}

	__device__ int get_phenotype(int cell, curandState_t *states) {
		if (state != 6) {
			float s = curand_uniform(&states[cell]);
			float sorted_phenotype[4];
			for (int i = 0; i < 4; i++) sorted_phenotype[i] = phenotype[i];
			bitonic_sort(sorted_phenotype, 0, 4, true);
			float R[4];
			R[0] = sorted_phenotype[0]*0.25 + (0.25 - 0.25*sorted_phenotype[3]);
			R[1] = R[0] + 0.25*sorted_phenotype[1] + (0.25 - 0.25*sorted_phenotype[2]);
			R[2] = R[1] + 0.25*sorted_phenotype[2] + (0.25 - 0.25*sorted_phenotype[1]);
			R[3] = R[2] + 0.25*sorted_phenotype[3] + (0.25 - 0.25*sorted_phenotype[0]);
			int idx[4];
			int count;
			if (s <= R[0])
				count = get_indexes(sorted_phenotype[0], phenotype, idx, 4);
			else if (s > R[0] && s <= R[1])
				count = get_indexes(sorted_phenotype[1], phenotype, idx, 4);
			else if (s > R[1] && s <= R[2])
				count = get_indexes(sorted_phenotype[2], phenotype, idx, 4);
			else
				count = get_indexes(sorted_phenotype[3], phenotype, idx, 4);
			int i = (int) ceilf(curand_uniform(&states[cell])*count) % count;
			return idx[i];
		} else return -1;
	}

	__device__ void copy_mutations(Cell *c) {
		for (int i = 0; i < 4; i++) {
			if (c->phenotype[i] + abs(phenotype[i] - phenotype_init[state*4+i]) < 0)
				c->phenotype[i] = 0.0f;
			else if (c->phenotype[i] + abs(phenotype[i] - phenotype_init[state*4+i]) > 1)
				c->phenotype[i] = 1.0f;
			else
				c->phenotype[i] += abs(phenotype[i] - phenotype_init[state*4+i]);
		}
		for (int i = 0; i < NN->n_hidden*NN->n_output; i++) {
			if (W_y_init[i] < 0)
				c->NN->W_out[i] -= abs(W_y_init[i] - NN->W_out[i]);
			else
				c->NN->W_out[i] += abs(W_y_init[i] - NN->W_out[i]);
		}
		for (int i = 0; i < 11; i++) {
			c->mutations[i] += mutations[i];
		}
	}

	__device__ int proliferate(int cell, Cell *c, curandState_t *states) {
		int idx = (int) ceilf(curand_uniform(&states[cell])*NN->n_output) % NN->n_output;
		int new_state = -1;
		if ((state == 4 && mutations[3] >= 0.1) || state != 4) {
			if (mutations[idx] >= 0.1) {
				if ((int) ceilf(curand_uniform(&states[cell])*2) % 2 == 0) {
					change_state(state_mut_map[state*11+idx]);
					c->change_state(child_mut_map[state*11+idx]);
				} else {
					c->change_state(child_mut_map[state*11+idx]);
					change_state(state_mut_map[state*11+idx]);
				}
				new_state = child_mut_map[state*11+idx];
			} else {
				c->change_state(child_mut_map[state*11]);
				new_state = state;
			}
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ int differentiate( int cell, Cell *c, curandState_t *states ) {
		int new_state = -1;
		if (state == 2) {
			if (mutations[1] >= 0.1 || mutations[8] >= 0.1 || mutations[10] >= 0.1)
				new_state = 1;
			else
				new_state = 0;
		} else if (state == 3)
			new_state = 1;
		else if (state == 4 && mutations[3] >= 0.1)
			new_state = 5;
		if (new_state != -1) {
			int idx = (int) ceilf(curand_uniform(&states[cell]) * NN->n_output) % NN->n_output;
			if ((int) ceilf(curand_uniform(&states[cell]) * 2) % 2 == 0) {
				change_state(state_mut_map[state * 11 + idx]);
				c->change_state(new_state);
			} else {
				c->change_state(new_state);
				change_state(state_mut_map[state * 11 + idx]);
			}
			copy_mutations(c);
		}

		return new_state;
	}

	__device__ void move(Cell *c) {
		c->change_state(state);
		copy_mutations(c);
		c->age = age;
		apoptosis();
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
			mutations[i] = 0;
		}
		age = 0;
	}

	__device__ void phenotype_mutate(int M, float prevMut, float newMut) {
		// down-regulation
		if (prevMut > newMut && count != 4) {
			for (int j = 0; j < 4; j++) {
				if (downreg_phenotype_map[M*4+j] < 0)
					phenotype[j] = fmaxf(0.0f, phenotype[j] + downreg_phenotype_map[M * 4 + j]);
				else
					phenotype[j] = fminf(phenotype[j] + downreg_phenotype_map[M*4+j], 1.0f);
			}
		// up-regulation
		} else {
			for (int j = 0; j < 4; j++) {
				if (upreg_phenotype_map[M*4+j] < 0)
					phenotype[j] = fmaxf(0.0f, phenotype[j] + upreg_phenotype_map[M * 4 + j]);
				else
					phenotype[j] = fminf(phenotype[j] + upreg_phenotype_map[M*4+j], 1.0f);
			}
		}
	}

	__device__ void mutate(int cell, float *in, float* result, curandState_t *states) {
		int M_idx[11];
		int count = max_idx(result, M_idx, NN->n_output);
		consumption = 0.0f;

		for (int i = 0; i < count; i++) {
			int M = M_idx[i];
			if (M != 0 && state != 6) {
				float prevMut = mutations[M];
				NN->mutate(cell, M, index_map, mutations, consumption, states);
				float newMut = mutations[M];
				phenotype_mutate(M, prevMut, newMut);
			}
		}

		if (state != 6) {
			// Random mutations
			if ((int) ceilf(curand_uniform(&states[cell])*1000) <= 10) {
				unsigned int num = (int) ceilf(curand_uniform(&states[cell])*4) % 4;
				for (int j = 0; j < num; j++) {
					unsigned int idx = (int) ceilf(curand_uniform(&states[cell])*11) % 11;
					float prevMut = mutations[idx];
					NN->mutate(cell, idx, index_map, mutations, consumption, states);
					float newMut = mutations[idx];
					float out[11];
					NN->evaluate(in, out);
					count = max_idx(out, M_idx, NN->n_output);
					for (int i = 0; i < count; i++) {
						int M = M_idx[i];
						if (M != 0 && state != 6) phenotype_mutate(M, prevMut, newMut);
					}
				}
			}
		}

		if (state != 6) age++;
	}
};

#endif // __CELL_H__
