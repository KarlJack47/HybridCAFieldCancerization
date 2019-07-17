#ifndef __CARCINOGEN_PDE_H__
#define __CARCINOGEN_PDE_H__

#include "common/general.h"

struct CarcinogenPDE {
	int device;
	unsigned int N;
	double T_scale;
	double ic;
	double bc;
	double diffusion;
	double outflux_per_cell;
	double influx_per_cell;
	unsigned int carcin_idx;

	double *results;

	CarcinogenPDE(unsigned int space_size, double diff, double out, double in, double ic_in, double bc_in, unsigned int idx, int dev) {
		device = dev;
		N = space_size;
		T_scale = 16.0f*120.0f;
		diffusion = diff;
		outflux_per_cell = out;
		influx_per_cell = in;
		ic = ic_in;
		bc = bc_in;
		carcin_idx = idx;

		CudaSafeCall(cudaMallocManaged((void**)&results, N*N*sizeof(double)));

		CudaSafeCall(cudaMemPrefetchAsync(results, N*N*sizeof(double), device, NULL));
	}

	void init(void) {
		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		init_pde<<< blocks, threads >>>(results, ic, bc, N);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(results));
	}

	void time_step(unsigned int step) {
		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

		pde_space_step<<< blocks, threads >>>(results, step*T_scale, N, bc, ic, diffusion,
						      influx_per_cell, outflux_per_cell);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}
};

#endif // __CARCINOGEN_PDE_H__
