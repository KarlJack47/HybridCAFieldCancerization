#ifndef __CA_H__
#define __CA_H__

#include "common/gpu_anim.h"
#include "common/lodepng.h"
#include "cell.h"
#include "carcinogen_pde.h"

// globals needed by the update routines
struct DataBlock {
	unsigned char *output_bitmap;
	Cell *prevGrid;
	Cell *newGrid;
	CarcinogenPDE *pdes;

	int grid_size;
	int cell_size;
	const int dim = 1024;
	int frame_rate;
	int maxT;
	int n_carcinogens;
	int n_output;

	int frames;
	int save_frames;
} d;

__managed__ bool csc_formed;
__managed__ bool tc_formed;

GPUAnimBitmap bitmap(d.dim, d.dim, &d);

__global__ void cells_gpu_to_gpu_copy(Cell *src, Cell *dst) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = x + y * blockDim.x * gridDim.x;

        dst[idx].state = src[idx].state;
        dst[idx].age = src[idx].age;

        for (int i = 0; i < 4; i++) dst[idx].phenotype[i] = src[idx].phenotype[i];
        for (int i = 0; i < src[idx].NN->n_input-1; i++) dst[idx].consumption[i] = src[idx].consumption[i];
        for (int i = 0; i < src[idx].NN->n_output; i++) dst[idx].mutations[i] = src[idx].mutations[i];

        dst[idx].NN->n_input = src[idx].NN->n_input;
        dst[idx].NN->n_hidden = src[idx].NN->n_hidden;
        dst[idx].NN->n_output = src[idx].NN->n_output;
        for (int i = 0; i < src[idx].NN->n_input; i++) dst[idx].NN->input[i] = src[idx].NN->input[i];
        for (int i = 0; i < src[idx].NN->n_hidden; i++) {
                dst[idx].NN->hidden[i] = src[idx].NN->hidden[i];
                dst[idx].NN->output[i] = src[idx].NN->output[i];
                dst[idx].NN->b_in[i] = src[idx].NN->b_in[i];
                dst[idx].NN->b_out[i] = src[idx].NN->b_out[i];
        }
        for (int i = 0; i < src[idx].NN->n_input*src[idx].NN->n_hidden; i++) dst[idx].NN->W_in[i] = src[idx].NN->W_in[i];
        for (int i = 0; i < src[idx].NN->n_hidden*src[idx].NN->n_output; i++) {
                dst[idx].NN->W_out[i] = src[idx].NN->W_out[i];
                dst[idx].W_y_init[i] = src[idx].W_y_init[i];
        }
}

__global__ void rule(Cell *newG, Cell *prevG, int t, CarcinogenPDE *pdes, curandState_t *states) {
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int phenotype = prevG[offset].get_phenotype(offset, states);

	int num_empty = 0;
	int empty[8];
	for (int i = 0; i < 8; i++) {
		if (prevG[prevG[offset].neighbourhood[i]].state == 6) {
			empty[num_empty] = prevG[offset].neighbourhood[i];
			num_empty++;
		}
	}

	int cell_idx = -1;
	if (num_empty != 0)
		cell_idx = empty[(int) ceilf(curand_uniform(&states[offset])*num_empty) % num_empty];

	if (phenotype == 0) {
		if (cell_idx != -1) {
			int state = newG[offset].proliferate(&newG[cell_idx], offset, states);
			if (state == -1)
				newG[offset].move(&newG[cell_idx]);
		}
	} else if (phenotype == 1) {
		if ((int) ceilf(curand_uniform(&states[offset])*1000) % 1000 == 50)
			newG[offset].move(&newG[cell_idx]);
	} else if (phenotype == 2) {
		newG[offset].apoptosis();
	} else if (phenotype == 3) {
		if (cell_idx != -1) {
			int state = newG[offset].differentiate(&newG[cell_idx], offset, states);
			if (state == -1)
				newG[offset].move(&newG[cell_idx]);
		}
	}

	for (int j = 0; j < prevG[offset].NN->n_input-1; j++) {
		prevG[offset].NN->input[j] = pdes[j].get(offset, t-1);
	}
	prevG[offset].NN->input[prevG[offset].NN->n_input-1] = prevG[offset].age;

	prevG[offset].NN->evaluate();

	newG[offset].mutate(prevG[offset].NN->input, prevG[offset].NN->output, offset, states);

	if (csc_formed == false && prevG[offset].state != 4 && newG[offset].state == 4) {
		printf("A CSC was formed at time step %d.\n", t);
		csc_formed = true;
	}
	if (tc_formed == false && prevG[offset].state != 5 && newG[offset].state == 5) {
		printf("A TC was formed at time step %d.\n", t);
		tc_formed = true;
	}
}

__global__ void display(uchar4 *optr, Cell *grid, int cell_size, int dim) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int offsetOptr = x * cell_size + y * cell_size * dim;

	if (grid[offset].state == 6.0f) { // white (empty)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 255;
				optr[j].y = 255;
				optr[j].z = 255;
				optr[j].w = 255;
			}
		}
	} else if (grid[offset].state == 0.0f) { // black (NC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j = j + dim) {
				optr[j].x = 0;
				optr[j].y = 0;
				optr[j].z = 0;
				optr[j].w = 255;
			}
		}
	} else if (grid[offset].state == 1.0f) { // green (MNC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 87;
				optr[j].y = 207;
				optr[j].z = 0;
				optr[j].w = 255;
			}
		}
	} else if (grid[offset].state == 2.0f) { // orange (SC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 244;
				optr[j].y = 131;
				optr[j].z = 0;
				optr[j].w = 255;
			}
		}
	} else if (grid[offset].state == 3.0f) { // blue (MSC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 0;
				optr[j].y = 0;
				optr[j].z = 255;
				optr[j].w = 255;
			}
		}
	} else if (grid[offset].state == 4.0f) { // purple (CSC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 89;
				optr[j].y = 35;
				optr[j].z = 112;
				optr[j].w = 255;
			}
		}
	} else if (grid[offset].state == 5.0f) { // red (TC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 255;
				optr[j].y = 0;
				optr[j].z = 0;
				optr[j].w = 255;
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

void anim_gpu(uchar4* outputBitmap, DataBlock *d, int ticks) {
	clock_t start, end;
	start = clock();
	set_seed();

	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks <= d->maxT) {
		printf("Starting %d\n", ticks);
		if (ticks == 0) {
			display<<< blocks, threads >>>(outputBitmap, d->newGrid, d->cell_size, d->dim);
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

			rule<<< blocks, threads >>>(d->newGrid, d->prevGrid, ticks, d->pdes, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			if (ticks % d->frame_rate == 0) {
				display<<< blocks, threads >>>(outputBitmap, d->newGrid, d->cell_size, d->dim);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
			}

			CudaSafeCall(cudaSetDevice(1));
			for (int i = 0; i < d->n_carcinogens; i++) {
				d->pdes[i].time_step(ticks, d->newGrid);
			}
			CudaSafeCall(cudaSetDevice(0));
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
		printf("%s\n", fname);
		unsigned char *frame;
		CudaSafeCall(cudaMallocManaged((void**)&frame, d->dim*d->dim*4*sizeof(unsigned char)));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>( outputBitmap, frame );
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

		CudaSafeCall(cudaFree(frame));
	}

	end = clock();
	printf("The time step took %f seconds to complete.\n", (double) (end - start) / CLOCKS_PER_SEC);
}

void anim_exit( DataBlock *d ) {
	CudaSafeCall(cudaDeviceSynchronize());
	for (int i = 0; i < d->grid_size*d->grid_size; i++) {
		d->prevGrid[i].free_resources();
		d->newGrid[i].free_resources();
	}
	CudaSafeCall(cudaFree(d->prevGrid));
	CudaSafeCall(cudaFree(d->newGrid));
	CudaSafeCall(cudaSetDevice(1));
	for (int i = 0; i < d->n_carcinogens; i++)
		d->pdes[i].free_resources();
	CudaSafeCall(cudaFree(d->pdes));
	CudaSafeCall(cudaSetDevice(0));
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
		CudaSafeCall(cudaSetDevice(1));
		CudaSafeCall(cudaMallocManaged((void**)&d.pdes, d.n_carcinogens*sizeof(CarcinogenPDE)));
		CudaSafeCall(cudaDeviceEnablePeerAccess(0, 0));
		CudaSafeCall(cudaSetDevice(0));
		CudaSafeCall(cudaDeviceEnablePeerAccess(1, 0));

		CudaSafeCall(cudaMalloc((void**)&d.output_bitmap, bitmap.image_size()));
		CudaSafeCall(cudaMallocManaged((void**)&d.prevGrid, d.grid_size*d.grid_size*sizeof(Cell)));
		CudaSafeCall(cudaMallocManaged((void**)&d.newGrid, d.grid_size*d.grid_size*sizeof(Cell)));
	}

	void init(float *carcin_map, double *diffusion, bool *liquid) {
		for (int i = 0; i < d.grid_size; i++) {
			for (int j = 0; j < d.grid_size; j++) {
				d.prevGrid[i*d.grid_size + j] = Cell(i, j, d.grid_size, d.n_carcinogens+1, d.n_output, carcin_map);
				d.newGrid[i*d.grid_size + j] = Cell(i, j, d.grid_size, d.n_carcinogens+1, d.n_output, carcin_map);
			}
		}

		for (int i = 0; i < d.n_carcinogens; i++) {
			d.pdes[i] = CarcinogenPDE(d.grid_size, d.maxT, diffusion[i], liquid[i], i);
			d.pdes[i].init();
		}
	}

	void animate(int frame_rate) {
		d.frame_rate = frame_rate;
		bitmap.anim_and_exit((void (*)(uchar4*, void*, int))anim_gpu, (void (*)(void*))anim_exit);
	}
};

#endif // __CA_H__
