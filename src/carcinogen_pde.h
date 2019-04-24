#ifndef __CARCINOGEN_PDE_H__
#define __CARCINOGEN_PDE_H__

#include "common/general.h"

__global__ void initialize(double *results, double ic, double bc, int Nx, int N, int T) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * gridDim.x * blockDim.x;
	int n;

	if (row < N && col < N) {
		for (n = 0; n < T; n++) {
			if (row == 0 || row == N-1 || col == 0 || col == N-1)
				results[n*Nx+idx] = bc;
			else
				results[n*Nx+idx] = ic;
		}
	}
}

__global__ void space_step(double *sol, int cur_t, int next_t, int N, double bc, double T_scale, double dt, double s,
			   double influx_per_cell, double outflux_per_cell, Cell *cells, int carcin) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * gridDim.x * blockDim.x;

	int next = next_t + idx;
	int cur = cur_t + idx;

	if (row < N && col < N) {
		if (!(row == 0 || row == N-1 || col == 0 || col == N-1)) {
			double in = 0.0f; double out = 0.0f;
			in = influx_per_cell;
			out = outflux_per_cell;
			sol[next] = sol[cur] +
				    s*(sol[cur_t+cells[idx].neighbourhood[0]] + sol[cur_t+cells[idx].neighbourhood[1]] +
		    	   	    sol[cur_t+cells[idx].neighbourhood[2]] + sol[cur_t+cells[idx].neighbourhood[3]] +
		    	    	    sol[cur_t+cells[idx].neighbourhood[4]] + sol[cur_t+cells[idx].neighbourhood[5]] +
		    	    	    sol[cur_t+cells[idx].neighbourhood[6]] + sol[cur_t+cells[idx].neighbourhood[7]] - 8.0f*sol[cur]) +
		    	    	    T_scale * dt * (in - out);
		} else sol[next] = bc;
	}
}

struct CarcinogenPDE {
	int device;
	int N;
	int T;
	double T_scale;
	double dx;
	double dt;
	double ic;
	double bc;
	double diffusion;
	double outflux_per_cell;
	double influx_per_cell;
	int Nx;
	int maxT;
	double s;
	bool liquid;
	int carcin_idx;

	double *sol;
	double *results;

	CarcinogenPDE(int space_size, int num_timesteps, double diff, double out, double in, int idx, int dev) {
		device = dev;
		N = space_size;
		T = num_timesteps + 1;
		T_scale = 16.0f*120.0f;
		diffusion = diff;
		outflux_per_cell = out;
		influx_per_cell = in;
		dx = 1.0f / (double) N;
		dt = 5e-4;
		ic = 0.0f;
		bc = 0.0f;
		Nx = N / dx;
		maxT = 1 / dt;
		s = (diffusion * (T_scale / (double) maxT) * dt) / (3 * dx * dx);
		carcin_idx = idx;

		CudaSafeCall(cudaMallocManaged((void**)&sol, maxT*Nx*sizeof(double)));
		CudaSafeCall(cudaMallocManaged((void**)&results, T*Nx*sizeof(double)));
	}

	void init(void) {
		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		CudaSafeCall(cudaMemPrefetchAsync(results, T*Nx*sizeof(double), device, NULL));
		initialize<<< blocks, threads >>>(results, ic, bc, Nx, N, T);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaMemPrefetchAsync(results, T*Nx*sizeof(double), cudaCpuDeviceId, NULL));
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(results));
		CudaSafeCall(cudaFree(sol));
	}

	void time_step(int step, Cell *cells) {
		set_seed();
		memcpy(&sol[0], &results[(step-1)*Nx], Nx*sizeof(double));

		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

		CudaSafeCall(cudaMemPrefetchAsync(sol, maxT*Nx*sizeof(double), device, NULL));

		double T_scl = T_scale / (double) maxT;
		for (int n = 0; n < maxT-1; n++) {
			int next = (n+1)*Nx;
			int cur = n*Nx;

			space_step<<< blocks, threads >>>(sol, cur, next, N, bc, T_scl, dt, s,
							  influx_per_cell, outflux_per_cell, cells, carcin_idx);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
		}

		CudaSafeCall(cudaMemPrefetchAsync(sol, maxT*Nx*sizeof(double), cudaCpuDeviceId, NULL));

		memcpy(&results[step*Nx], &sol[(maxT-1)*Nx], Nx*sizeof(double));
	}

	__host__ __device__ double get(int cell, int time) {
		return results[time*Nx+cell];
	}
};

#endif // __CARCINOGEN_PDE_H__
