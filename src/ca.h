#ifndef __CA_H__
#define __CA_H__

#include "common/general.h"

struct CA {
	CA(unsigned int g_size, unsigned int T, int save_frames, int display, int maxt_tc, char **carcin_names) {
		d.grid_size = g_size;
		d.cell_size = d.dim/d.grid_size;
		d.maxT = T;
		d.save_frames = save_frames;
		d.maxt_tc_alive = maxt_tc;
		bitmap.display = display;
		bitmap.grid_size = g_size;
		bitmap.maxT = T;
		if (display == 0) bitmap.paused = false;

		bitmap.carcin_names = (char**)malloc(NUM_CARCIN*sizeof(char*));
		for (int i = 0; i < NUM_CARCIN; i++) {
			bitmap.carcin_names[i] = (char*)malloc((strlen(carcin_names[i])+1)*sizeof(char));
			strcpy(bitmap.carcin_names[i], carcin_names[i]);
		}
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

		CudaSafeCall(cudaMallocManaged((void**)&d.pdes, NUM_CARCIN*sizeof(CarcinogenPDE)));
		CudaSafeCall(cudaMallocManaged((void**)&d.prevGrid, d.grid_size*d.grid_size*sizeof(Cell)));
		CudaSafeCall(cudaMallocManaged((void**)&d.newGrid, d.grid_size*d.grid_size*sizeof(Cell)));
	}

	void init(double *diffusion, double *out, double *in, double *ic,
		  double *bc, double *W_x, double *W_y, double *b_y) {
		int nt = omp_get_num_procs();
		int counts[nt] = { 0 };
		printf("Grid initialization progress:   0.00/100.00");
		for (int i = 0; i < d.grid_size; i++)
			for (int j = 0; j < d.grid_size; j+=2) {
				#pragma omp parallel sections num_threads(2)
				{
					#pragma omp section
					{
						d.prevGrid[i*d.grid_size + j] = Cell(j, i, d.grid_size, d.dev_id_1, W_x, W_y, b_y);
						d.prevGrid[i*d.grid_size + j].prefetch_cell_params(d.dev_id_1, d.grid_size);

						d.newGrid[i*d.grid_size + j] = Cell(j, i, d.grid_size, d.dev_id_2, W_x, W_y, b_y);
						d.newGrid[i*d.grid_size + j].prefetch_cell_params(d.dev_id_2, d.grid_size);
						counts[omp_get_thread_num()]++;
					}

					#pragma omp section
					{
						d.prevGrid[i*d.grid_size + (j+1)] = Cell(j+1, i, d.grid_size, d.dev_id_1, W_x, W_y, b_y);
						d.prevGrid[i*d.grid_size + (j+1)].prefetch_cell_params(d.dev_id_1, d.grid_size);

						d.newGrid[i*d.grid_size + (j+1)] = Cell(j+1, i, d.grid_size, d.dev_id_2, W_x, W_y, b_y);
						d.newGrid[i*d.grid_size + (j+1)].prefetch_cell_params(d.dev_id_2, d.grid_size);
						counts[omp_get_thread_num()]++;
					}
				}

				int num_finished = 0;
				for (int k = 0; k < nt; k++) num_finished += counts[k];
				print_progress(num_finished, d.grid_size*d.grid_size);
			}
		printf("\n");

		for (int k = 0; k < NUM_CARCIN; k++) {
			d.pdes[k] = CarcinogenPDE(d.grid_size, diffusion[k], out[k], in[k],
						  ic[k], bc[k], k, d.dev_id_2);
			d.pdes[k].init();
		}

		CudaSafeCall(cudaMemPrefetchAsync(d.prevGrid, d.grid_size*d.grid_size*sizeof(Cell), d.dev_id_1, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(d.newGrid, d.grid_size*d.grid_size*sizeof(Cell), d.dev_id_2, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(d.pdes, NUM_CARCIN*sizeof(CarcinogenPDE), d.dev_id_2, NULL));
		prefetch_params(d.dev_id_1);

		CudaSafeCall(cudaDeviceSynchronize());
	}

	void animate(int frame_rate) {
		d.frame_rate = frame_rate;
		bitmap.anim((void (*)(uchar4*, void*, int))anim_gpu_ca, (void (*)(uchar4*, void*, int))anim_gpu_genes,
			    (void (*)(uchar4*, void*, int, int))anim_gpu_carcin,
			    (void (*)(uchar4*, void*, int, int))anim_gpu_cell, (void (*)(void*, bool, int))anim_gpu_timer);
	}
};

#endif // __CA_H__
