#ifndef __CA_H__
#define __CA_H__

#include "common/general.h"

// globals needed by the update routines
struct DataBlock {
	int dev_id_1, dev_id_2;
	Cell *prevGrid, *newGrid;
	CarcinogenPDE *pdes;

	int grid_size, cell_size;
	const int dim = 1024;
	int frame_rate;
	int maxT;
	int n_carcinogens;
	int n_output;

	clock_t start, end;
	clock_t start_step, end_step;
	int frames, save_frames;
} d;
#pragma omp threadprivate(d)

__managed__ bool csc_formed;
#define MAX_EXCISE 20
__managed__ bool tc_formed[MAX_EXCISE+1];
__managed__ int excise_count;

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
        	for (i = 0; i < src[idx].NN->n_output; i++) {
			dst[idx].gene_expressions[i*2] = src[idx].gene_expressions[i*2];
			dst[idx].gene_expressions[i*2+1] = src[idx].gene_expressions[i*2+1];
		}

        	dst[idx].NN->n_input = src[idx].NN->n_input;
        	dst[idx].NN->n_hidden = src[idx].NN->n_hidden;
        	dst[idx].NN->n_output = src[idx].NN->n_output;
       		for (i = 0; i < src[idx].NN->n_input; i++) dst[idx].NN->input[i] = src[idx].NN->input[i];
        	for (i = 0; i < src[idx].NN->n_hidden; i++) {
                	dst[idx].NN->hidden[i] = src[idx].NN->hidden[i];
                	dst[idx].NN->output[i] = src[idx].NN->output[i];
                	dst[idx].NN->b_out[i] = src[idx].NN->b_out[i];
        	}
        	for (i = 0; i < src[idx].NN->n_input*src[idx].NN->n_hidden; i++) dst[idx].NN->W_in[i] = src[idx].NN->W_in[i];
        	for (i = 0; i < src[idx].NN->n_hidden*src[idx].NN->n_output; i++)
                	dst[idx].NN->W_out[i] = src[idx].NN->W_out[i];
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

		prevG[offset].mutate(prevG[offset].NN->output, offset, states);
	}
}

__global__ void check_CSC_or_TC_formed(Cell *newG, Cell *prevG, int g_size, int t) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		int offset = x + y * blockDim.x * gridDim.x;
		if (csc_formed == false && prevG[offset].state != 4 && newG[offset].state == 4) {
			printf("A CSC was formed at time step %d.\n", t);
			csc_formed = true;
		}
		if (tc_formed == false && prevG[offset].state != 5 && newG[offset].state == 5) {
			if (excise_count == 0) printf("A TC was formed at time step %d.\n", t);
			else printf("A TC was reformed after excision %d at time step %d.\n", excise_count, t);
			tc_formed[excise_count] = true;
		}
	}
}

__global__ void reset_rule_params(Cell *prevG, int g_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) prevG[x + y * blockDim.x * gridDim.x].chosen_phenotype = -1;
}

__global__ void rule(Cell *newG, Cell *prevG, int g_size, int phenotype, curandState_t *states) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		int offset = x + y * blockDim.x * gridDim.x;

		if (prevG[offset].state == 6) return;

		if (prevG[offset].chosen_phenotype == -1)
			prevG[offset].chosen_phenotype = prevG[offset].get_phenotype(offset, states);

		if (phenotype == 2 && prevG[offset].chosen_phenotype == 2) {
			newG[offset].apoptosis();
			return;
		}

		int state = -2; int i;

		bool neigh[8] = { false, false, false, false, false, false, false, false };
		while (neigh[0] == false && neigh[1] == false && neigh[2] == false && neigh[3] == false &&
		       neigh[4] == false && neigh[5] == false && neigh[6] == false && neigh[7] == false) {
			int idx = (int) ceilf(curand_uniform(&states[offset])*8) % 8;
			int neigh_idx = prevG[offset].neighbourhood[idx];
			if (neigh[idx] == false) {
				if (phenotype == 0 && prevG[offset].chosen_phenotype == 0)
					state = newG[offset].proliferate(&newG[neigh_idx], offset, states);
				else if (phenotype == 3 && prevG[offset].chosen_phenotype == 3)
					state = newG[offset].differentiate(&newG[neigh_idx], offset, states);
				else if (phenotype == 1 && prevG[offset].chosen_phenotype == 1)
					state = newG[offset].move(&newG[neigh_idx], offset, states);
				if (state != -2) break;
				neigh[i] = true;
			}
		}
	}
}

__global__ void tumour_excision(Cell *G, int g_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < g_size && y < g_size) {
		int offset = x + y * blockDim.x * gridDim.x;

		if (G[offset].state != 5) return;

		for (int i = 0; i < 8; i++) G[G[offset].neighbourhood[i]].apoptosis();
		G[offset].apoptosis();
	}
}

__global__ void display_ca(uchar4 *optr, Cell *grid, int g_size, int cell_size, int dim) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int offsetOptr = x * cell_size + y * cell_size * dim;
	int i, j;

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

__global__ void display_carcin(uchar4 *optr, CarcinogenPDE *pde, int g_size, int cell_size, int dim, int t) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int offsetOptr = x * cell_size + y * cell_size * dim;
	int i, j;

	if (x < g_size && y < g_size) {
		for (i = offsetOptr; i < offsetOptr + cell_size; i++) {
			for (j = i; j < i + cell_size * dim; j += dim) {
				optr[j].x = ceil(fmaxf(0.0f, 255.0f - 255.0f*pde->get(offset, t)));
				optr[j].y = ceil(fmaxf(0.0f, 255.0f - 255.0f*pde->get(offset, t)));
				optr[j].z = ceil(fmaxf(0.0f, 255.0f - 255.0f*pde->get(offset, t)));
				optr[j].w = 255;
			}
		}
	}
}

__global__ void display_cell_data(uchar4 *optr, Cell *grid, int cell_idx, int dim) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int gene = x / (float) (dim / 20);
	if (gene % 2 == 1) gene = floor((float) gene / 2.0f);
	else return;
	int gene_expr_up = grid[cell_idx].gene_expressions[gene*2] * 100;
	int gene_expr_down = grid[cell_idx].gene_expressions[gene*2+1] * 100;
	int height = y / (float) (dim / 201);

	if (x < dim && y < dim) {
		if ((gene_expr_up < gene_expr_down && height < 100 && (100 - height) <= gene_expr_down) ||
		     (gene_expr_up > gene_expr_down && height > 100 && height - 100 <= gene_expr_up))  {
			optr[offset].x = state_colors[grid[cell_idx].state*3];
			optr[offset].y = state_colors[grid[cell_idx].state*3 + 1];
			optr[offset].z = state_colors[grid[cell_idx].state*3 + 2];
			optr[offset].w = 255;
		} else {
			optr[offset].x = 255;
			optr[offset].y = 255;
			optr[offset].z = 255;
			optr[offset].w = 255;
		}
	}
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

void anim_gpu_ca(uchar4* outputBitmap, DataBlock *d, int ticks) {
	set_seed();

	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks <= d->maxT) {
		if (ticks == 0) {
			display_ca<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
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
			while (used_pheno[0] == false && used_pheno[1] == false && used_pheno[2] == false && used_pheno[3] == false) {
				int pheno = rand() % 4;
				if (used_pheno[pheno] == false) {
					rule<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size, pheno, states);
					CudaCheckError();
					CudaSafeCall(cudaDeviceSynchronize());

					if (pheno == 1) {
						cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
						CudaCheckError();
						CudaSafeCall(cudaDeviceSynchronize());
					}

					used_pheno[pheno] = true;
				}
			}

			check_CSC_or_TC_formed<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size, ticks);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			reset_rule_params<<< blocks, threads >>>(d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			if (ticks % d->frame_rate == 0) {
				display_ca<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
			}

			for (int i = 0; i < d->n_carcinogens; i++)
				d->pdes[i].time_step(ticks, d->newGrid);

			CudaSafeCall(cudaFree(states));

			if (bitmap.excise == true && excise_count <= MAX_EXCISE && tc_formed[excise_count] == true) {
				tumour_excision<<< blocks, threads >>>(d->newGrid, d->grid_size);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
				printf("Tumour excision was performed at time step %d.\n", ticks);
				excise_count++;
			}
		}

		++d->frames;
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

	if (ticks == d->maxT && d->save_frames == 1) {
		if (numDigits(d->maxT) == 1)
                    system("ffmpeg -y -v quiet -framerate 5 -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p out_ca.mp4");
                else if (numDigits(d->maxT) == 2)
                    system("ffmpeg -y -v quiet -framerate 5 -start_number 0 -i %02d.png -c:v libx264 -pix_fmt yuv420p out_ca.mp4");
                else if (numDigits(d->maxT) == 3)
                    system("ffmpeg -y -v quiet -framerate 5 -start_number 0 -i %03d.png -c:v libx264 -pix_fmt yuv420p out_ca.mp4");
	}
}

void anim_gpu_carcin(uchar4* outputBitmap, DataBlock *d, int carcin_idx, int ticks) {
	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks % d->frame_rate == 0) {
		CudaSafeCall(cudaMemPrefetchAsync(d->pdes[carcin_idx].results, d->pdes[carcin_idx].T*d->pdes[carcin_idx].Nx*sizeof(double), d->dev_id_1, NULL));
		prefetch_pdes(d->dev_id_2);

		display_carcin<<< blocks, threads >>>(outputBitmap, &d->pdes[carcin_idx], d->grid_size, d->cell_size, d->dim, ticks);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		CudaSafeCall(cudaMemPrefetchAsync(d->pdes[carcin_idx].results, d->pdes[carcin_idx].T*d->pdes[carcin_idx].Nx*sizeof(double), cudaCpuDeviceId, NULL));
		prefetch_pdes(-1);
	}

	if (!bitmap.paused && d->save_frames == 1 && ticks <= d->maxT) {
		char fname[25] = {'c', 'a', 'r', 'c', 'i', 'n'};
		sprintf(&fname[6], "%d", carcin_idx);
		int dig_car = numDigits(carcin_idx);
		fname[6+dig_car] = '_';
		int dig_max = numDigits(d->maxT);
		int dig = numDigits(ticks);
		for (int i = 6+dig_car+1; i < 6+dig_car+1+dig_max-dig; i++) fname[i] = '0';
		sprintf(&fname[6+dig_car+1+dig_max-dig], "%d", ticks);
		fname[6+dig_car+1+dig_max] = '.';
		fname[6+dig_car+1+dig_max+1] = 'p';
		fname[6+dig_car+1+dig_max+2] = 'n';
		fname[6+dig_car+1+dig_max+3] = 'g';
		unsigned char *frame;
		CudaSafeCall(cudaMallocManaged((void**)&frame, d->dim*d->dim*4*sizeof(unsigned char)));
		CudaSafeCall(cudaMemPrefetchAsync(frame, d->dim*d->dim*4*sizeof(unsigned char), 1, NULL));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>(outputBitmap, frame);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

		CudaSafeCall(cudaFree(frame));
	}

	if (!bitmap.paused && ticks == d->maxT && d->save_frames == 1) {
		char command[250] = { '\0' };
		char car_idx[5] = { '\0' };
		strcat(command, "ffmpeg -y -v quiet -framerate 5 -start_number 0 -i carcin");
		sprintf(car_idx, "%d", carcin_idx);
		strcat(command, car_idx);
		strcat(command, "_");
		if (numDigits(d->maxT) == 1)
			strcat(command, "%d.png -c:v libx264 -pix_fmt yuv420p out_carcin");
		else if (numDigits(d->maxT) == 2)
			strcat(command, "%02d.png -c:v libx264 -pix_fmt yuv420p out_carcin");
		else if (numDigits(d->maxT) == 3)
			strcat(command, "%03d.png -c:v libx264 -pix_fmt yuv420p out_carcin");
		strcat(command, car_idx);
		strcat(command, ".mp4");
		system(command);
	}
}

void anim_gpu_cell(uchar4* outputBitmap, DataBlock *d, int cell_idx, int ticks) {
	dim3 blocks(d->dim / BLOCK_SIZE, d->dim / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks % d->frame_rate == 0) {
		display_cell_data<<< blocks, threads >>>(outputBitmap, d->newGrid, cell_idx, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}
}

void anim_gpu_timer(DataBlock *d, bool start, int ticks) {
	if (start) {
		d->start_step = clock();
		printf("starting %d\n", ticks);
	} else {
		printf("done %d\n", ticks);
		d->end_step = clock();
		printf("The time step took %f seconds to complete.\n", (double) (d->end_step - d->start_step) / CLOCKS_PER_SEC);
	}
	if (ticks == 0) d->start = clock();

	if (ticks == d->maxT && !start) {
		d->end = clock();
		printf("It took %f seconds to run the %d time steps.\n", (double) (d->end - d->start) / CLOCKS_PER_SEC, d->maxT);
		exit(EXIT_SUCCESS);
	}
}

void anim_exit( DataBlock *d ) {
	CudaSafeCall(cudaDeviceSynchronize());
	prefetch_params(-1);
	#pragma omp parallel for schedule(guided)
		for (int i = 0; i < d->grid_size*d->grid_size; i++) {
			d->prevGrid[i].prefetch_cell_params(-1, d->grid_size);
			d->prevGrid[i].free_resources();
			d->prevGrid[i].prefetch_cell_params(-1, d->grid_size);
			d->newGrid[i].free_resources();
		}
	prefetch_grids(-1, -1);
	CudaSafeCall(cudaFree(d->prevGrid));
	CudaSafeCall(cudaFree(d->newGrid));
	for (int i = 0; i < d->n_carcinogens; i++)
		d->pdes[i].free_resources();
	CudaSafeCall(cudaFree(d->pdes));
}

struct CA {
	CA(int g_size, int T, int n_carcin, int n_out, int save_frames, int display, char **carcin_names) {
		d.frames = 0;
		d.grid_size = g_size;
		d.cell_size = d.dim/d.grid_size;
		d.maxT = T;
		d.n_carcinogens = n_carcin;
		d.n_output = n_out;
		d.save_frames = save_frames;
		bitmap.display = display;
		bitmap.n_carcin = n_carcin;
		bitmap.grid_size = g_size;
		bitmap.maxT = T;
		if (display == 0) bitmap.paused = false;

		bitmap.carcin_names = (char**)malloc(n_carcin*sizeof(char*));
		for (int i = 0; i < n_carcin; i++) {
			bitmap.carcin_names[i] = (char*)malloc((strlen(carcin_names[i])+1)*sizeof(char));
			strcpy(bitmap.carcin_names[i], carcin_names[i]);
		}
		bitmap.create_window(2, bitmap.width, bitmap.height, carcin_names[0], &bitmap.key_carcin);
		if (bitmap.display == 0)
			for (int i = 0; i < 3; i++)
				bitmap.hide_window(bitmap.windows[i]);
		csc_formed = false;
		for (int i = 0; i < MAX_EXCISE+1; i++) tc_formed[i] = false;
		excise_count = 0;
	}

	~CA(void) {
		anim_exit(&d);
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
		CudaSafeCall(cudaMallocManaged((void**)&d.prevGrid, d.grid_size*d.grid_size*sizeof(Cell)));
		CudaSafeCall(cudaMallocManaged((void**)&d.newGrid, d.grid_size*d.grid_size*sizeof(Cell)));
	}

	void init(double *diffusion, double *out, double *in, double *W_x, double *W_y, double *b_y) {
		int i, j;

		printf("Grid initialization progress:   0.00/100.00");
		#pragma omp parallel for collapse(2) schedule(guided)
			for (i = 0; i < d.grid_size; i++) {
				for (j = 0; j < d.grid_size; j++) {
					d.prevGrid[i*d.grid_size + j] = Cell(j, i, d.grid_size, d.n_carcinogens+1, d.n_output, d.dev_id_1, W_x, W_y, b_y);
					d.prevGrid[i*d.grid_size + j].prefetch_cell_params(d.dev_id_1, d.grid_size);
					d.newGrid[i*d.grid_size + j] = Cell(j, i, d.grid_size, d.n_carcinogens+1, d.n_output, d.dev_id_2, W_x, W_y, b_y);
					d.newGrid[i*d.grid_size + j].prefetch_cell_params(d.dev_id_2, d.grid_size);

					char format[40] = { '\0' };
					double progress = ((i*d.grid_size + j) / (double) (d.grid_size * d.grid_size)) * 100.0f;
					int num_dig = numDigits(progress);
					int limit = num_dig+10;
					if (trunc(progress*1000.0f) >= 9995 && num_dig == 1) limit += 1;
					else if (trunc(progress*1000.0f) >= 99995 && num_dig == 2) limit += 1;
					for (int k = 0; k < limit; k++) strcat(format, "\b");
					strcat(format, "%.2f/%.2f");
					printf(format, progress, 100.0f);
				}
			}
		printf("\n");

		prefetch_grids(d.dev_id_1, d.dev_id_2);

		for (int k = 0; k < d.n_carcinogens; k++) {
			d.pdes[k] = CarcinogenPDE(d.grid_size, d.maxT, diffusion[k], out[k], in[k], k, d.dev_id_2);
			d.pdes[k].init();
		}

		prefetch_params(d.dev_id_1);
	}

	void animate(int frame_rate) {
		d.frame_rate = frame_rate;
		bitmap.anim((void (*)(uchar4*, void*, int))anim_gpu_ca, (void (*)(uchar4*, void*, int, int))anim_gpu_carcin,
			    (void (*)(uchar4*, void*, int, int))anim_gpu_cell, (void (*)(void*, bool, int))anim_gpu_timer);
	}
};

#endif // __CA_H__
