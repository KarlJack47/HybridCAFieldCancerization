#ifndef __CARCINOGEN_PDE_H__
#define __CARCINOGEN_PDE_H__

#include "common/general.h"

struct CarcinogenPDE {
	int device;
	unsigned int N;
	unsigned int T;
	double T_scale;
	double dx;
	double dt;
	double ic;
	double bc;
	double diffusion;
	double outflux_per_cell;
	double influx_per_cell;
	unsigned int Nx;
	unsigned int maxT;
	double s;
	bool liquid;
	unsigned int carcin_idx;

	double *results;

	CarcinogenPDE(unsigned int space_size, unsigned int num_timesteps, double diff, double out, double in, double ic_in, double bc_in, unsigned int idx, int dev) {
		device = dev;
		N = space_size;
		T = num_timesteps + 1;
		T_scale = 16.0f*120.0f;
		diffusion = diff;
		outflux_per_cell = out;
		influx_per_cell = in;
		dx = 1.0f / (double) N;
		dt = 5e-4;
		ic = ic_in;
		bc = bc_in;
		Nx = N / dx;
		maxT = 1 / dt;
		s = (diffusion * (T_scale / (double) maxT) * dt) / (3 * dx * dx);
		carcin_idx = idx;

		CudaSafeCall(cudaMallocManaged((void**)&results, Nx*sizeof(double)));
	}

	void init(void) {
		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		CudaSafeCall(cudaMemPrefetchAsync(results, Nx*sizeof(double), device, NULL));
		initialize<<< blocks, threads >>>(results, ic, bc, Nx, N);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaMemPrefetchAsync(results, Nx*sizeof(double), cudaCpuDeviceId, NULL));
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(results));
	}

	void time_step(unsigned int step, Cell *cells) {
		double *prev;
		CudaSafeCall(cudaMallocManaged((void**)&prev, Nx*sizeof(double)));

		memcpy(prev, results, Nx*sizeof(double));

		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

		CudaSafeCall(cudaMemPrefetchAsync(prev, Nx*sizeof(double), device, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(results, Nx*sizeof(double), device, NULL));

		double T_scl = T_scale / (double) maxT;
		for (unsigned int n = 0; n < maxT-1; n++) {
			space_step<<< blocks, threads >>>(prev, results, N, bc, T_scl, dt, s,
							  influx_per_cell, outflux_per_cell, cells);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			CudaSafeCall(cudaMemcpy(prev, results, Nx*sizeof(double), cudaMemcpyDeviceToDevice));
		}

		CudaSafeCall(cudaMemPrefetchAsync(prev, Nx*sizeof(double), cudaCpuDeviceId, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(results, Nx*sizeof(double), cudaCpuDeviceId, NULL));

		CudaSafeCall(cudaFree(prev));
	}
};

#endif // __CARCINOGEN_PDE_H__
