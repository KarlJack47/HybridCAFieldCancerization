#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

// Set random seed for gpu
__global__ void init_curand(unsigned int seed, curandState_t* states) {
        unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &states[id]);
}

// Copy gpu bitmap to unsigned char array so that pngs can be created.
__global__ void copy_frame(uchar4 *optr, unsigned char *frame) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int dim = gridDim.x*blockDim.x;
	unsigned int idx = x + ((dim-1)-y)*dim;

	frame[4*dim*y+4*x] = optr[idx].x;
	frame[4*dim*y+4*x+1] = optr[idx].y;
	frame[4*dim*y+4*x+2] = optr[idx].z;
	frame[4*dim*y+4*x+3] = optr[idx].w;
}

// Initalizes the carcinogen pde grid.
__global__ void init_pde(double *results, double ic, double bc, unsigned int N) {
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = row + col * gridDim.x * blockDim.x;

	if (row < N && col < N) {
		if (row == 0 || row == N-1 || col == 0 || col == N-1)
			results[idx] = CELL_VOLUME*bc;
		else
			results[idx] = CELL_VOLUME*ic;
	}
}

// Spacial step for the carcinogen pde.
__global__ void pde_space_step(double *results, unsigned int t, unsigned int N, double bc, double ic,
			       double D, double influx_per_cell, double outflux_per_cell) {
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = row + col * gridDim.x * blockDim.x;

	if (row < N && col < N) {
		if (!(row == 0 || row == N-1 || col == 0 || col == N-1)) {
			double sum = 0.0f;
			double pi_div_N = M_PI / (double) N;
			double pi_squared = M_PI*M_PI;
			double source_term = influx_per_cell - outflux_per_cell;
			double ic_minus_bc = ic - bc;
			double N_squared = N*N;
			double Dt = D*t;
			for (int n = 0; n < MAX_ITER; n++) {
				double n_odd = 2.0f*n + 1.0f;
				for (int m = 0; m < MAX_ITER; m++) {
					double m_odd = 2.0f*m + 1.0f;
					double lambda = ((n_odd*n_odd + m_odd*m_odd)*pi_squared) / N_squared;
					double exp_result = exp(-lambda*Dt);
					sum += ((source_term * (1.0f - exp_result) / (lambda*D) + exp_result * ic_minus_bc) *
					       sin(col*n_odd*pi_div_N) * sin(row*m_odd*pi_div_N)) / (n_odd*m_odd);
				}
			}
			results[idx] = CELL_VOLUME*((16.0f / pi_squared)*sum + bc);
		} else results[idx] = CELL_VOLUME*bc;
	}
}

// CA related kernels
__global__ void cells_gpu_to_gpu_copy(Cell *src, Cell *dst, int g_size) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = x + y * blockDim.x * gridDim.x;
	unsigned int i;

	if (x < g_size && y < g_size) {

		dst[idx].state = src[idx].state;
        	dst[idx].age = src[idx].age;

        	for (i = 0; i < NUM_PHENO; i++) dst[idx].phenotype[i] = src[idx].phenotype[i];
        	for (i = 0; i < src[idx].NN->n_output; i++) {
			dst[idx].gene_expressions[i*2] = src[idx].gene_expressions[i*2];
			dst[idx].gene_expressions[i*2+1] = src[idx].gene_expressions[i*2+1];
		}

        	dst[idx].NN->n_input = src[idx].NN->n_input;
        	dst[idx].NN->n_hidden = src[idx].NN->n_hidden;
        	dst[idx].NN->n_output = src[idx].NN->n_output;
       		for (i = 0; i < NUM_CARCIN+1; i++) dst[idx].NN->input[i] = src[idx].NN->input[i];
        	for (i = 0; i < NUM_GENES; i++) {
                	dst[idx].NN->hidden[i] = src[idx].NN->hidden[i];
                	dst[idx].NN->output[i] = src[idx].NN->output[i];
                	dst[idx].NN->b_out[i] = src[idx].NN->b_out[i];
        	}
        	for (i = 0; i < (NUM_CARCIN+1)*NUM_GENES; i++) dst[idx].NN->W_in[i] = src[idx].NN->W_in[i];
        	for (i = 0; i < NUM_GENES*NUM_GENES; i++)
                	dst[idx].NN->W_out[i] = src[idx].NN->W_out[i];
	}
}

__global__ void mutate_grid(Cell *prevG, unsigned int g_size, unsigned int t, CarcinogenPDE *pdes, curandState_t *states) {
	// map from threadIdx/blockIdx to pixel position
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = x + y * blockDim.x * gridDim.x;
	unsigned int i;

	if (x < g_size && y < g_size) {
		for (i = 0; i < NUM_CARCIN; i++) prevG[offset].NN->input[i] = pdes[i].results[offset];
		prevG[offset].NN->input[NUM_CARCIN] = prevG[offset].age;
		prevG[offset].NN->evaluate();

		prevG[offset].mutate(prevG[offset].NN->output, offset, states);
	}
}

__global__ void check_CSC_or_TC_formed(Cell *newG, Cell *prevG, unsigned int g_size, unsigned int t) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		unsigned int offset = x + y * blockDim.x * gridDim.x;
		if (csc_formed == false && prevG[offset].state != CSC && newG[offset].state == CSC) {
			printf("A CSC was formed at time step %d.\n", t);
			csc_formed = true;
		}
		if (tc_formed[excise_count] == false && prevG[offset].state != TC && newG[offset].state == TC) {
			if (excise_count == 0) printf("A TC was formed at time step %d.\n", t);
			else printf("A TC was reformed after excision %d and %d time steps at time step %d.\n", excise_count, time_tc_dead, t);
			tc_formed[excise_count] = true;
		}
	}
}

__global__ void reset_rule_params(Cell *prevG, unsigned int g_size) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) prevG[x + y * blockDim.x * gridDim.x].chosen_phenotype = -1;
}

__global__ void rule(Cell *newG, Cell *prevG, unsigned int g_size, unsigned int phenotype, curandState_t *states) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		unsigned int offset = x + y * blockDim.x * gridDim.x;

		if (prevG[offset].state == EMPTY) return;

		if (prevG[offset].chosen_phenotype == -1)
			prevG[offset].chosen_phenotype = prevG[offset].get_phenotype(offset, states);

		if (phenotype == APOP && prevG[offset].chosen_phenotype == APOP) {
			newG[offset].apoptosis();
			return;
		}

		int state = -2; unsigned int i;

		bool neigh[NUM_NEIGH] = { false, false, false, false, false, false, false, false };
		while (neigh[NORTH] == false && neigh[EAST] == false && neigh[SOUTH] == false && neigh[WEST] == false &&
		       neigh[NORTH_EAST] == false && neigh[SOUTH_EAST] == false && neigh[SOUTH_WEST] == false && neigh[NORTH_WEST] == false) {
			unsigned int idx = (unsigned int) ceilf(curand_uniform(&states[offset])*NUM_NEIGH) % NUM_NEIGH;
			unsigned int neigh_idx = prevG[offset].neighbourhood[idx];
			if (neigh[idx] == false) {
				if (phenotype == PROLIF && prevG[offset].chosen_phenotype == PROLIF)
					state = newG[offset].proliferate(&newG[neigh_idx], offset, states);
				else if (phenotype == DIFF && prevG[offset].chosen_phenotype == DIFF)
					state = newG[offset].differentiate(&newG[neigh_idx], offset, states);
				else if (phenotype == QUIES && prevG[offset].chosen_phenotype == QUIES)
					state = newG[offset].move(&newG[neigh_idx], offset, states);
				if (state != -2) break;
				neigh[i] = true;
			}
		}
	}
}

__global__ void update_states(Cell *G, unsigned int g_size) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		unsigned int offset = x + y * blockDim.x * gridDim.x;

		if (G[offset].state != CSC && G[offset].state != TC && G[offset].state != EMPTY) {
			int check = 1;
			for (int i = 0; i < NUM_GENES; i++) {
				if (G[offset].positively_mutated(i) == 0) {
					check = 0;
					break;
				}
			}
			if (check == 1 && G[offset].state == MNC) G[offset].change_state(NC);
			else if (check == 1 && G[offset].state == MSC) G[offset].change_state(SC);
		}
	}
}

__global__ void tumour_excision(Cell *G, unsigned int g_size) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		unsigned int offset = x + y * blockDim.x * gridDim.x;

		if (G[offset].state != TC) return;

		for (int i = 0; i < NUM_NEIGH; i++) G[G[offset].neighbourhood[i]].apoptosis();
		G[offset].apoptosis();
	}
}

__global__ void display_ca(uchar4 *optr, Cell *grid, unsigned int g_size, unsigned int cell_size, unsigned int dim) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = x + y * blockDim.x * gridDim.x;
	unsigned int offsetOptr = x * cell_size + y * cell_size * dim;
	unsigned int i, j;

	if (x < g_size && y < g_size) {
		for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = state_colors[grid[offset].state*3];
				optr[j].y = state_colors[grid[offset].state*3 + 1];
				optr[j].z = state_colors[grid[offset].state*3 + 2];
				optr[j].w = 255;
			}
		}
	}
}

__global__ void display_genes(uchar4 *optr, Cell *grid, unsigned int g_size, unsigned int cell_size, unsigned int dim) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = x + y * blockDim.x * gridDim.x;
	unsigned int offsetOptr = x * cell_size + y * cell_size * dim;
	unsigned int i, j;

	int max_gene = 0;
	for (int i = 0; i < NUM_GENES; i++) {
		if (grid[offset].positively_mutated(i) == 0) {
			if (grid[offset].positively_mutated(max_gene) == 1) {
				max_gene = i;
				continue;
			}
			if (fabsf(grid[offset].gene_expressions[i*2] - grid[offset].gene_expressions[i*2+1]) >
			    fabsf(grid[offset].gene_expressions[max_gene*2] - grid[offset].gene_expressions[max_gene*2+1]))
				max_gene = i;
		}
	}

	if (x < g_size && y < g_size) {
		for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (j = i; j < i + cell_size * dim; j += dim) {
				if (grid[offset].state != 0 && grid[offset].state != 2 && grid[offset].state != 6 &&
				    grid[offset].positively_mutated(max_gene) == 0) {
					optr[j].x = gene_colors[max_gene*3];
					optr[j].y = gene_colors[max_gene*3 + 1];
					optr[j].z = gene_colors[max_gene*3 + 2];
					optr[j].w = 255;
				} else {
					optr[j].x = 255;
					optr[j].y = 255;
					optr[j].z = 255;
					optr[j].w = 255;
				}
			}
		}
	}
}

__global__ void display_carcin(uchar4 *optr, CarcinogenPDE *pde, unsigned int g_size, unsigned int cell_size, unsigned int dim) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = x + y * blockDim.x * gridDim.x;
	unsigned int offsetOptr = x * cell_size + y * cell_size * dim;
	unsigned int i, j;

	double carcin_con = pde->results[offset];

	if (x < g_size && y < g_size) {
		for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = ceil(fmaxf(0.0f, 255.0f - 255.0f*carcin_con));
				optr[j].y = ceil(fmaxf(0.0f, 255.0f - 255.0f*carcin_con));
				optr[j].z = ceil(fmaxf(0.0f, 255.0f - 255.0f*carcin_con));
				optr[j].w = 255;
			}
		}
	}
}

__global__ void display_cell_data(uchar4 *optr, Cell *grid, unsigned int cell_idx, unsigned int dim) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < dim && y < dim) {
		unsigned int offset = x + y * blockDim.x * gridDim.x;
		unsigned int gene = x / (dim / (double) (2*NUM_GENES));
		unsigned int scale = 100;
		unsigned int height = y / (dim / (double) (2*scale+1));

		optr[offset].x = 255;
		optr[offset].y = 255;
		optr[offset].z = 255;
		optr[offset].w = 255;

		if (abs((int) scale - (int) height) == trunc(MUT_THRESHOLD*(double) scale)) {
			optr[offset].x = 248;
			optr[offset].y = 222;
			optr[offset].z = 126;
			optr[offset].w = 255;
		}

		if (abs((int) scale - (int) height) == 0) {
			optr[offset].x = 0;
			optr[offset].y = 0;
			optr[offset].z = 0;
			optr[offset].w = 255;
		}

		if (gene % 2 == 1) gene = floor(gene / 2.0f);
		else return;

		int gene_expr_up = grid[cell_idx].gene_expressions[gene*2] * scale;
		int gene_expr_down = grid[cell_idx].gene_expressions[gene*2+1] * scale;

		optr[offset].x = 248;
		optr[offset].y = 222;
		optr[offset].z = 126;
		optr[offset].w = 255;

		if ((gene_expr_up < gene_expr_down && height < scale && (scale - height) <= gene_expr_down) ||
		     (gene_expr_up > gene_expr_down && height > scale && (height - scale) <= gene_expr_up))  {
			optr[offset].x = state_colors[grid[cell_idx].state*3];
			optr[offset].y = state_colors[grid[cell_idx].state*3 + 1];
			optr[offset].z = state_colors[grid[cell_idx].state*3 + 2];
			optr[offset].w = 255;
		}
	}
}

#endif // __CUDA_KERNELS_H__
