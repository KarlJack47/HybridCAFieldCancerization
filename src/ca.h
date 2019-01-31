#ifndef __CA_H__
#define __CA_H__

#include "common/general.h"

// globals needed by the update routines
struct DataBlock {
	int dev_id_1, dev_id_2;
	unsigned char *output_bitmap;
	Cell *prevGrid, *newGrid;
	CarcinogenPDE *pdes;

	int grid_size, cell_size;
	const int dim = 1024;
	int frame_rate;
	int maxT;
	int n_carcinogens;
	int n_output;

	clock_t start, end;
	int frames, save_frames;
} d;
#pragma omp threadprivate(d)

__managed__ bool csc_formed;
__managed__ bool tc_formed;

GPUAnimBitmap bitmap(d.dim, d.dim, &d);

__global__ void cells_gpu_to_gpu_copy(Cell *src, Cell *dst, int g_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = x + y * blockDim.x * gridDim.x;
	int i;

	if (x < g_size && y < g_size) {

		dst[idx].state = src[idx].state;
        	dst[idx].age = src[idx].age;

        	for (i = 0; i < 4; i++) dst[idx].phenotype[i] = src[idx].phenotype[i];
        	for (i = 0; i < src[idx].NN->n_output; i++) dst[idx].mutations[i] = src[idx].mutations[i];

        	dst[idx].NN->n_input = src[idx].NN->n_input;
        	dst[idx].NN->n_hidden = src[idx].NN->n_hidden;
        	dst[idx].NN->n_output = src[idx].NN->n_output;
       		for (i = 0; i < src[idx].NN->n_input; i++) dst[idx].NN->input[i] = src[idx].NN->input[i];
        	for (i = 0; i < src[idx].NN->n_hidden; i++) {
                	dst[idx].NN->hidden[i] = src[idx].NN->hidden[i];
                	dst[idx].NN->output[i] = src[idx].NN->output[i];
                	dst[idx].NN->b_in[i] = src[idx].NN->b_in[i];
                	dst[idx].NN->b_out[i] = src[idx].NN->b_out[i];
        	}
        	for (i = 0; i < src[idx].NN->n_input*src[idx].NN->n_hidden; i++) dst[idx].NN->W_in[i] = src[idx].NN->W_in[i];
        	for (i = 0; i < src[idx].NN->n_hidden*src[idx].NN->n_output; i++) {
                	dst[idx].NN->W_out[i] = src[idx].NN->W_out[i];
                	dst[idx].W_y_init[i] = src[idx].W_y_init[i];
        	}
	}
}

__global__ void mutate_grid(Cell *prevG, int g_size, int t, CarcinogenPDE *pdes, curandState_t *states) {
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int i;

	if (x < g_size && y < g_size) {
		for (i = 0; i < prevG[offset].NN->n_input-1; i++)
			prevG[offset].NN->input[i] = pdes[i].get(offset, t-1);
		prevG[offset].NN->input[prevG[offset].NN->n_input-1] = prevG[offset].age;
		prevG[offset].NN->evaluate();

		prevG[offset].mutate(prevG[offset].NN->input, prevG[offset].NN->output, offset, states);
	}
}

__device__ void check_CSC_or_TC_formed(Cell *newG, Cell *prevG, int idx, int t) {
	if (csc_formed == false && prevG[idx].state != 4 && newG[idx].state == 4) {
		printf("A CSC was formed at time step %d.\n", t);
		csc_formed = true;
	}
	if (tc_formed == false && prevG[idx].state != 5 && newG[idx].state == 5) {
		printf("A TC was formed at time step %d.\n", t);
		tc_formed = true;
	}
}

__global__ void reset_rule_params(Cell *prevG, int g_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) prevG[x + y * blockDim.x * gridDim.x].chosen_phenotype = -1;
}

__device__ void choose_idx_and_phenotype(Cell *prevG, int g_size, int offset, int *idx, int *phenotype, curandState_t *states) {
	if (*phenotype == -1)
		*phenotype = prevG[offset].get_phenotype(offset, states);

	if (*idx == -1) {
		int i;
		int empty[8]; int num_empty = 0;

		for (i = 0; i < 8; i++) {
			int neigh_idx = prevG[offset].neighbourhood[i];
			if (prevG[neigh_idx].state == 6)
				empty[num_empty++] = neigh_idx;
		}

		if (num_empty == 0) { *idx = -1; return; }

		*idx = empty[(int) ceilf(curand_uniform(&states[offset])*num_empty) % num_empty];
	}
}

__global__ void rule(Cell *newG, Cell *prevG, int g_size, int phenotype, int t, curandState_t *states) {
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		int offset = x + y * blockDim.x * gridDim.x;

		if (prevG[offset].state == 6) return;

		int idx = -1;
		if (phenotype == 2) idx = -2;
		choose_idx_and_phenotype(prevG, g_size, offset, &idx, &prevG[offset].chosen_phenotype, states);

		if (phenotype == 2 && prevG[offset].chosen_phenotype == 2) {
			newG[offset].apoptosis();
			return;
		}

		if (idx == -1) return;

		int state = -2;

		if (phenotype == 0 && prevG[offset].chosen_phenotype == 0)
			state = newG[offset].proliferate(&newG[idx], offset, states);
		else if (phenotype == 3 && prevG[offset].chosen_phenotype == 3)
			state = newG[offset].differentiate(&newG[idx], offset, states);
		else if (phenotype == 1 && prevG[offset].chosen_phenotype == 1)
			newG[offset].move(&newG[idx], offset, states);

		if (state == -1)
			newG[offset].move(&newG[idx], offset, states);

		check_CSC_or_TC_formed(prevG, newG, offset, t);
	}
}

__global__ void display(uchar4 *optr, Cell *grid, int g_size, int cell_size, int dim) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int offsetOptr = x * cell_size + y * cell_size * dim;
	int i, j;

	if (x < g_size && y < g_size) {
		if (grid[offset].state == 6.0f) { // white (empty)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 255;
					optr[j].y = 255;
					optr[j].z = 255;
					optr[j].w = 255;
				}
			}
		} else if (grid[offset].state == 0.0f) { // black (NC)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 0;
					optr[j].y = 0;
					optr[j].z = 0;
					optr[j].w = 255;
				}
			}
		} else if (grid[offset].state == 1.0f) { // green (MNC)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 87;
					optr[j].y = 207;
					optr[j].z = 0;
					optr[j].w = 255;
				}
			}
		} else if (grid[offset].state == 2.0f) { // orange (SC)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 244;
					optr[j].y = 131;
					optr[j].z = 0;
					optr[j].w = 255;
				}
			}
		} else if (grid[offset].state == 3.0f) { // blue (MSC)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 0;
					optr[j].y = 0;
					optr[j].z = 255;
					optr[j].w = 255;
				}
			}
		} else if (grid[offset].state == 4.0f) { // purple (CSC)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 89;
					optr[j].y = 35;
					optr[j].z = 112;
					optr[j].w = 255;
				}
			}
		} else if (grid[offset].state == 5.0f) { // red (TC)
			for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
				for (j = i; j < i + cell_size * dim; j += dim) {
					optr[j].x = 255;
					optr[j].y = 0;
					optr[j].z = 0;
					optr[j].w = 255;
				}
			}
		}

	}
}

__global__ void copy_frame(uchar4 *optr, unsigned char *frame) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int dim = gridDim.x*blockDim.x;
	int idx = x + ((dim-1)-y)*dim;

	frame[4*dim*y+4*x] = optr[idx].x;
	frame[4*dim*y+4*x+1] = optr[idx].y;
	frame[4*dim*y+4*x+2] = optr[idx].z;
	frame[4*dim*y+4*x+3] = 255;
}

void prefetch_grids(int loc1, int loc2) {
	int location1 = loc1; int location2 = loc2;
	if (loc1 == -1) location1 = cudaCpuDeviceId;
	if (loc2 == -1) location2 = cudaCpuDeviceId;

	CudaSafeCall(cudaMemPrefetchAsync(d.prevGrid, d.grid_size*d.grid_size*sizeof(Cell), location1, NULL));
	CudaSafeCall(cudaMemPrefetchAsync(d.newGrid, d.grid_size*d.grid_size*sizeof(Cell), location2, NULL));
}

void prefetch_pdes(int loc) {
	int location = loc;
	if (loc == -1) location = cudaCpuDeviceId;
	CudaSafeCall(cudaMemPrefetchAsync(d.pdes, d.n_carcinogens*sizeof(CarcinogenPDE), location, NULL));
}

void anim_gpu(uchar4* outputBitmap, DataBlock *d, int ticks) {
	clock_t start_step, end_step;
	start_step = clock();
	if (ticks == 0) d->start = clock();
	set_seed();

	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks <= d->maxT) {
		printf("Starting %d\n", ticks);

		if (ticks == 0) {
			display<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
		} else {
			curandState_t *states;
			CudaSafeCall(cudaMalloc((void**)&states, d->grid_size*d->grid_size*sizeof(curandState_t)));
			timespec seed;
        		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &seed);
			init_curand<<< d->grid_size*d->grid_size, 1 >>>(seed.tv_nsec, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			for (int i = 0; i < d->n_carcinogens; i++)
				CudaSafeCall(cudaMemPrefetchAsync(d->pdes[i].results, d->pdes[i].T*d->pdes[i].Nx*sizeof(double), d->dev_id_2, NULL));
			prefetch_pdes(d->dev_id_2);

			mutate_grid<<< blocks, threads >>>(d->prevGrid, d->grid_size, ticks, d->pdes, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			for (int i = 0; i < d->n_carcinogens; i++)
				CudaSafeCall(cudaMemPrefetchAsync(d->pdes[i].results, d->pdes[i].T*d->pdes[i].Nx*sizeof(double), cudaCpuDeviceId, NULL));
			prefetch_pdes(-1);

			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->prevGrid, d->newGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			bool used_pheno[4] = { false, false, false, false };
			int pheno = rand() % 4;
			while (used_pheno[0] == false && used_pheno[1] == false && used_pheno[2] == false && used_pheno[3] == false) {
				int pheno = rand() % 4;
				if (used_pheno[pheno] == false) {
					rule<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size, pheno, ticks, states);
					CudaCheckError();
					CudaSafeCall(cudaDeviceSynchronize());
					cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
					CudaCheckError();
					CudaSafeCall(cudaDeviceSynchronize());

					used_pheno[pheno] = true;
				}
			}

			reset_rule_params<<< blocks, threads >>>(d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			if (ticks % d->frame_rate == 0) {
				display<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
			}

			for (int i = 0; i < d->n_carcinogens; i++) {
				d->pdes[i].time_step(ticks, d->newGrid);
			}

			CudaSafeCall(cudaFree(states));
		}

		++d->frames;
		printf("Done %d\n", ticks);
	}

	if (d->save_frames == 1 && ticks <= d->maxT) {
		char fname[14] = {' '};
		int dig_max = numDigits(d->maxT);
		int dig = numDigits(ticks);
		for (int i = 0; i < dig_max-dig; i++) fname[i] = '0';
		sprintf(&fname[dig_max-dig], "%d", ticks);
		fname[dig_max] = '.';
		fname[dig_max+1] = 'p';
		fname[dig_max+2] = 'n';
		fname[dig_max+3] = 'g';
		unsigned char *frame;
		CudaSafeCall(cudaMallocManaged((void**)&frame, d->dim*d->dim*4*sizeof(unsigned char)));
		CudaSafeCall(cudaMemPrefetchAsync(frame, d->dim*d->dim*4*sizeof(unsigned char), 1, NULL));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>( outputBitmap, frame );
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

		CudaSafeCall(cudaFree(frame));
	}

	end_step = clock();
	printf("The time step took %f seconds to complete.\n", (double) (end_step - start_step) / CLOCKS_PER_SEC);

	if (ticks == d->maxT) {
		d->end = clock();
		printf("It took %f seconds to run the %d time steps.\n", (double) (d->end - d->start) / CLOCKS_PER_SEC, d->maxT);
	}
}

void anim_exit( DataBlock *d ) {
	CudaSafeCall(cudaDeviceSynchronize());
	prefetch_params(-1);
	prefetch_grids(-1, -1);
	#pragma omp parallel for schedule(guided)
		for (int i = 0; i < d->grid_size*d->grid_size; i++) {
			d->prevGrid[i].prefetch_cell_params(-1, d->grid_size);
			d->prevGrid[i].free_resources();
			d->prevGrid[i].prefetch_cell_params(-1, d->grid_size);
			d->newGrid[i].free_resources();
		}
	CudaSafeCall(cudaFree(d->prevGrid));
	CudaSafeCall(cudaFree(d->newGrid));
	for (int i = 0; i < d->n_carcinogens; i++)
		d->pdes[i].free_resources();
	CudaSafeCall(cudaFree(d->pdes));
}

struct CA {
	CA(int g_size, int T, int n_carcin, int n_out, int save_frames, int display) {
		d.frames = 0;
		d.grid_size = g_size;
		d.cell_size = d.dim/d.grid_size;
		d.maxT = T;
		bitmap.maxT = T;
		d.n_carcinogens = n_carcin;
		d.n_output = n_out;
		d.save_frames = save_frames;
		bitmap.save_frames = save_frames;
		bitmap.display = display;
		if (bitmap.display == 0)
			bitmap.hide_window();
		csc_formed = false;
		tc_formed = false;
	}

	~CA(void) {
		anim_exit(&d);
		bitmap.free_resources();
	}

	void initialize_memory() {
		int num_devices;
		CudaSafeCall(cudaGetDeviceCount(&num_devices));
		for (int i = num_devices-1; i > -1; i--) {
			CudaSafeCall(cudaSetDevice(i));
			for (int j = i-1; j > -1; j--)
				CudaSafeCall(cudaDeviceEnablePeerAccess(j, 0));
			CudaSafeCall(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 33554432));
		}

		d.dev_id_1 = 0; d.dev_id_2 = 0;
		if (num_devices == 2) d.dev_id_2 = 1;

		CudaSafeCall(cudaMallocManaged((void**)&d.pdes, d.n_carcinogens*sizeof(CarcinogenPDE)));
		CudaSafeCall(cudaMalloc((void**)&d.output_bitmap, bitmap.image_size()));
		CudaSafeCall(cudaMallocManaged((void**)&d.prevGrid, d.grid_size*d.grid_size*sizeof(Cell)));
		CudaSafeCall(cudaMallocManaged((void**)&d.newGrid, d.grid_size*d.grid_size*sizeof(Cell)));
	}

	void init(double *diffusion, double *consum, double *in, bool *liquid) {
		int i, j;

		#pragma omp parallel for collapse(2) schedule(guided)
			for (i = 0; i < d.grid_size; i++) {
				for (j = 0; j < d.grid_size; j++) {
					d.prevGrid[i*d.grid_size + j] = Cell(j, i, d.grid_size, d.n_carcinogens+1, d.n_output, d.dev_id_1);
					d.prevGrid[i*d.grid_size + j].prefetch_cell_params(d.dev_id_1, d.grid_size);
					d.newGrid[i*d.grid_size + j] = Cell(j, i, d.grid_size, d.n_carcinogens+1, d.n_output, d.dev_id_2);
					d.newGrid[i*d.grid_size + j].prefetch_cell_params(d.dev_id_2, d.grid_size);
				}
			}

		prefetch_grids(d.dev_id_1, d.dev_id_2);

		for (int k = 0; k < d.n_carcinogens; k++) {
			d.pdes[k] = CarcinogenPDE(d.grid_size, d.maxT, diffusion[k], consum[k], in[k], liquid[k], k, d.dev_id_2);
			d.pdes[k].init();
		}

		prefetch_params(d.dev_id_1);
	}

	void animate(int frame_rate) {
		d.frame_rate = frame_rate;
		bitmap.anim_and_exit((void (*)(uchar4*, void*, int))anim_gpu, (void (*)(void*))anim_exit);
	}
};

#endif // __CA_H__
