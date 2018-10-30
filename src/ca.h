#ifndef __CA_H__
#define __CA_H__

#include "common/gpu_anim.h"
#include "common/lodepng.h"
#include "cell.h"
#include "carcinogen_pde.h"

// globals needed by the update routines
struct DataBlock {
	unsigned char *output_bitmap;
	Cell *dev_prevGrid;
	Cell *dev_newGrid;
	CarcinogenPDE *dev_pdes;
	CarcinogenPDE *pdes;

	unsigned int grid_size;
	unsigned int cell_size;
	const unsigned int dim = 1024;
	unsigned int frame_rate;
	unsigned int maxT;
	unsigned int n_carcinogens;
	unsigned int n_output;

	unsigned int frames;
	int save_frames;

	bool *csc_formed;
	bool *tc_formed;
} d;

GPUAnimBitmap bitmap(d.dim, d.dim, &d);

__global__ void rule(Cell *newG, Cell *prevG, int t, int grid_size, bool *csc_formed, bool *tc_formed,
		     CarcinogenPDE *pdes, curandState_t *states) {
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float result[2];
	for (int j = 0; j < prevG[offset].NN->n_input-1; j++) {
		result[j] = pdes[j].get(offset, t-1);
	}
	result[prevG[offset].NN->n_input-1] = prevG[offset].age;

	float output[11];
	prevG[offset].NN->evaluate(result, output);

	int phenotype;
	phenotype = prevG[offset].get_phenotype(offset, states);

	int neighbourhood[8];
	neighbourhood[0] = x + abs((y+1) % grid_size) * blockDim.x * gridDim.x; // n
	neighbourhood[1] = abs((x+1) % grid_size) + abs((y+1) % grid_size) * blockDim.x * gridDim.x; // ne
	neighbourhood[2] = abs((x+1) % grid_size) + y * blockDim.x * gridDim.x; // e
	neighbourhood[3] = abs((x+1) % grid_size) + abs((y-1) % grid_size) * blockDim.x * gridDim.x; // se
	neighbourhood[4] = x + abs((y-1) % grid_size) * blockDim.x * gridDim.x; // s
	neighbourhood[5] = abs((x-1) % grid_size) + abs((y-1) % grid_size) * blockDim.x * gridDim.x; // sw
	neighbourhood[6] = abs((x-1) % grid_size) + y; // w
	neighbourhood[7] = abs((x-1) % grid_size) + abs((y+1) % grid_size) * blockDim.x * gridDim.x; // nw

	int num_empty = 0;
	int empty[8];
	for (int i = 0; i < 8; i++) {
		if (prevG[neighbourhood[i]].state == 6) {
			empty[num_empty] = neighbourhood[i];
			num_empty++;
		}
	}

	int cell_idx = -1;
	if (num_empty != 0)
		cell_idx = empty[(int) ceilf(curand_uniform(&states[offset])*num_empty) % num_empty];

	if (phenotype == 0) {
		if (cell_idx != -1) {
			int state = newG[offset].proliferate(offset, &newG[cell_idx], states);
			if (state == -1) {
				newG[offset].move(&newG[cell_idx]);
			}
		}
	} else if (phenotype == 1) {
		if ((int) ceilf(curand_uniform(&states[offset])*1000) % 1000 == 50 && cell_idx != -1)
			newG[offset].move(&newG[cell_idx]);
	} else if (phenotype == 2)
		newG[offset].apoptosis();
	else if (phenotype == 3) {
		if (cell_idx != -1) {
			int state = newG[offset].differentiate(offset, &newG[cell_idx], states);
			if (state == -1) {
				newG[offset].move(&newG[cell_idx]);
			}
		}
	}

	newG[offset].mutate(offset, result, output, states);

	if (csc_formed[0] == false && prevG[offset].state != 4 && newG[offset].state == 4) {
		printf("A CSC was formed at time step %d.\n", t);
		csc_formed[0] = true;
	}
	if (tc_formed[0] == false && prevG[offset].state != 5 && newG[offset].state == 5) {
		printf("A TC was formed at time step %d.\n", t);
		tc_formed[0] = true;
	}
}

__global__ void display(uchar4 *optr, Cell *dev_new, int cell_size, int dim) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int offsetOptr = x * cell_size + y * cell_size * dim;

	if (dev_new[offset].state == 6.0f) { // white (empty)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 255;
				optr[j].y = 255;
				optr[j].z = 255;
				optr[j].w = 255;
			}
		}
	} else if (dev_new[offset].state == 0.0f) { // black (NC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j = j + dim) {
				optr[j].x = 0;
				optr[j].y = 0;
				optr[j].z = 0;
				optr[j].w = 255;
			}
		}
	} else if (dev_new[offset].state == 1.0f) { // green (MNC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 87;
				optr[j].y = 207;
				optr[j].z = 0;
				optr[j].w = 255;
			}
		}
	} else if (dev_new[offset].state == 2.0f) { // orange (SC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 244;
				optr[j].y = 131;
				optr[j].z = 0;
				optr[j].w = 255;
			}
		}
	} else if (dev_new[offset].state == 3.0f) { // blue (MSC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 0;
				optr[j].y = 0;
				optr[j].z = 255;
				optr[j].w = 255;
			}
		}
	} else if (dev_new[offset].state == 4.0f) { // purple (CSC)
		for (int i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (int j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = 89;
				optr[j].y = 35;
				optr[j].z = 112;
				optr[j].w = 255;
			}
		}
	} else if (dev_new[offset].state == 5.0f) { // red (TC)
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

__global__ void copy_result(int idx, int step, CarcinogenPDE *pdes, float *result) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	pdes[idx].results[step*pdes[idx].Nx+i] = result[step*pdes[idx].Nx+i];
}

void anim_gpu(uchar4* outputBitmap, DataBlock *d, int ticks) {
	clock_t start, end;
	start = clock();

	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks <= d->maxT) {
		printf("Starting %d\n", ticks);
		if (ticks == 0) {
			display<<< blocks, threads >>>(outputBitmap, d->dev_newGrid, d->cell_size, d->dim);
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
			rule<<< blocks, threads >>>(d->dev_newGrid, d->dev_prevGrid, ticks, d->grid_size, d->csc_formed, d->tc_formed, d->dev_pdes, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
			CudaSafeCall(cudaMemcpy(d->dev_prevGrid, d->dev_newGrid, d->grid_size*d->grid_size*sizeof(Cell), cudaMemcpyDeviceToDevice));
			if (ticks % d->frame_rate == 0) {
				display<<< blocks, threads >>>(outputBitmap, d->dev_newGrid, d->cell_size, d->dim);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
			}
			CudaSafeCall(cudaSetDevice(1));
			for (int i = 0; i < d->n_carcinogens; i++) {
				d->pdes[i].time_step(ticks, d->dev_newGrid, states);
				copy_result<<< d->pdes[i].Nx / BLOCK_SIZE, BLOCK_SIZE >>>(i, ticks, d->dev_pdes, d->pdes[i].results);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
			}
			CudaSafeCall(cudaSetDevice(0));
			CudaSafeCall(cudaFree(states));
		}
		++d->frames;
		printf("Done %d\n", ticks);
	}
	if (d->save_frames == 1 && ticks <= d->maxT) {
		char fname[14];
		int dig_max = numDigits(d->maxT);
		int dig = numDigits(ticks);
		for (int i = 0; i < dig_max-dig; i++) fname[i] = '0';
		sprintf(&fname[dig_max-dig], "%d", ticks);
		fname[dig_max] = '.';
		fname[dig_max+1] = 'p';
		fname[dig_max+2] = 'n';
		fname[dig_max+3] = 'g';
		unsigned char frame[d->dim*d->dim*4];
		unsigned char *dev_frame;
		CudaSafeCall(cudaMalloc((void**)&dev_frame, d->dim*d->dim*4*sizeof(unsigned char)));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>( outputBitmap, dev_frame );
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaMemcpy(frame, dev_frame, d->dim*d->dim*4*sizeof(unsigned char), cudaMemcpyDeviceToHost));
		CudaSafeCall(cudaFree(dev_frame));

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	end = clock();
	printf("The time step took %f seconds to complete.\n", (double) (end - start) / CLOCKS_PER_SEC);
}

void anim_exit( DataBlock *d ) {
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaFree(d->dev_prevGrid));
	CudaSafeCall(cudaFree(d->dev_newGrid));
	CudaSafeCall(cudaFree(d->csc_formed));
	CudaSafeCall(cudaFree(d->tc_formed));
	for (int i = 0; i < d->n_carcinogens; i++)
		d->pdes[i].free_resources();
	free(d->pdes);
}

struct CA {
	CA(unsigned int g_size, unsigned int T, unsigned int n_carcin, unsigned int n_out, int save_frames, int display) {
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
		bool csc_formed[1] = {false};
		bool tc_formed[1] = {false};
		CudaSafeCall(cudaMalloc(&d.csc_formed, sizeof(bool)));
		CudaSafeCall(cudaMalloc(&d.tc_formed, sizeof(bool)));
		CudaSafeCall(cudaMemcpy(d.csc_formed, csc_formed, sizeof(bool), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(d.tc_formed, tc_formed, sizeof(bool), cudaMemcpyHostToDevice));
	}

	~CA(void) {
		anim_exit(&d);
		bitmap.free_resources();
	}

	void initialize_memory() {
		d.pdes = (CarcinogenPDE*)malloc(d.n_carcinogens*sizeof(CarcinogenPDE));

		CudaSafeCall(cudaMalloc((void**)&d.output_bitmap, bitmap.image_size()));
		CudaSafeCall(cudaMalloc((void**)&d.dev_prevGrid, d.grid_size*d.grid_size*sizeof(Cell)));
		CudaSafeCall(cudaMalloc((void**)&d.dev_newGrid, d.grid_size*d.grid_size*sizeof(Cell)));

		CudaSafeCall(cudaSetDevice(1));
		CudaSafeCall(cudaMalloc((void**)&d.dev_pdes, d.n_carcinogens*sizeof(CarcinogenPDE)));
		CudaSafeCall(cudaDeviceEnablePeerAccess(0, 0));
		CudaSafeCall(cudaSetDevice(0));
		CudaSafeCall(cudaDeviceEnablePeerAccess(1, 0));
	}

	void init(float *carcin_map, double *diffusion, bool *liquid) {
		Cell *c = (Cell*)malloc(d.grid_size*d.grid_size*sizeof(Cell));
		for (int i = 0; i < d.grid_size*d.grid_size; i++) {
			c[i] = Cell(d.n_carcinogens+1, d.n_output, carcin_map);
			c[i].host_to_gpu_copy(i, d.dev_prevGrid);
		}
		d.n_output = c[0].NN->n_output;
		for (int i = 0; i < d.grid_size*d.grid_size; i++) {
			c[i].free_resources();
		}
		free(c);
		CudaSafeCall(cudaMemcpy(d.dev_newGrid, d.dev_prevGrid, d.grid_size*d.grid_size*sizeof(Cell), cudaMemcpyDeviceToDevice));

		CudaSafeCall(cudaSetDevice(1));
		for (int i = 0; i < d.n_carcinogens; i++) {
			d.pdes[i] = CarcinogenPDE(d.grid_size, d.maxT, diffusion[i], liquid[i], i);
			d.pdes[i].init();
			d.pdes[i].host_to_gpu_copy(i, d.dev_pdes);
		}
		CudaSafeCall(cudaSetDevice(0));
	}

	void animate(unsigned int frame_rate) {
		d.frame_rate = frame_rate;
		bitmap.anim_and_exit((void (*)(uchar4*, void*, int))anim_gpu, (void (*)(void*))anim_exit);
	}
};

#endif // __CA_H__
