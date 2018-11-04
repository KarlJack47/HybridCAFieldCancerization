#ifndef __CARCINOGEN_PDE_H__
#define __CARCINOGEN_PDE_H__

#include "cell.h"

__global__ void initialize(float *results, float ic, float bc, int Nx, int N, int T) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * gridDim.x * blockDim.x;

	for (int n = 0; n < T; n++) {
		if (row == 0 || row == N-1 || col == 0 || col == N-1) {
			results[n*Nx+idx] = bc;
		} else {
			results[n*Nx+idx] = ic;
		}
	}
}

__global__ void space_step(float *sol, int cur_t, int next_t, float T_scale, double dt, double s, bool liquid,
			   double influx, int num_cells_absorb, double amount_per_cell, Cell *cells, int carcin) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * gridDim.x * blockDim.x;

	int next = next_t + idx;
	int cur = cur_t + idx;

	__shared__ double absorbed;
	if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
		absorbed = 0.0f;
	}

	bool neigh_exist = false;
	for (int i = 0; i < 8; i++) {
		if (sol[cur_t+cells[idx].neighbourhood[i]] != 0) {
			neigh_exist = true;
			break;
		}
	}

	if ((liquid && neigh_exist) || !liquid)
		atomicAdd(&absorbed, amount_per_cell);
	__syncthreads();

	double in = 0.0f;
	if ((liquid && neigh_exist) || !liquid) in = influx - absorbed;
	sol[next] = sol[cur] +
		    s*(4*sol[cur_t+cells[idx].neighbourhood[0]] + 4*sol[cur_t+cells[idx].neighbourhood[1]] +
		    4*sol[cur_t+cells[idx].neighbourhood[2]] + 4*sol[cur_t+cells[idx].neighbourhood[3]] +
		    sol[cur_t+cells[idx].neighbourhood[4]] + sol[cur_t+cells[idx].neighbourhood[5]] +
		    sol[cur_t+cells[idx].neighbourhood[6]] + sol[cur_t+cells[idx].neighbourhood[7]] - 20*sol[cur]) +
		    T_scale * dt * (fmaxf(0, in) - cells[idx].consumption[carcin]);
}

struct CarcinogenPDE {
	int N;
	int T;
	float T_scale;
	double dx;
	double dt;
	float ic;
	float bc;
	double diffusion;
	int Nx;
	int maxT;
	double s;
	bool liquid;
	int carcin_idx;

	float *sol;
	float *results;

	CarcinogenPDE(int space_size, int num_timesteps, double diff, bool liq, int idx) {
		N = space_size;
		T = num_timesteps + 1;
		T_scale = 16.0f;
		diffusion = diff;
		dx = 1 / (double) N;
		dt = 5e-4;
		liquid = liq;
		ic = 0.0f;
		if (liquid == true) {
			bc = 1.0f;
		} else {
			bc = 0.0f;
		}
		Nx = N / dx;
		maxT = 1 / dt;
		s = (diffusion * T_scale * dt) / (6 * dx * dx);
		carcin_idx = idx;

		CudaSafeCall(cudaSetDevice(1));
		CudaSafeCall(cudaMallocManaged((void**)&sol, maxT*Nx*sizeof(float)));
		CudaSafeCall(cudaMallocManaged((void**)&results, T*Nx*sizeof(float)));
		CudaSafeCall(cudaSetDevice(0));
	}

	~CarcinogenPDE(void) {
		free_resources();
	}

	void init(void) {
		CudaSafeCall(cudaSetDevice(1));
		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		initialize<<< blocks, threads >>>(results, ic, bc, Nx, N, T);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaSetDevice(0));
	}

	void free_resources(void) {
		CudaSafeCall(cudaSetDevice(1));
		CudaSafeCall(cudaFree(results));
		CudaSafeCall(cudaFree(sol));
		CudaSafeCall(cudaSetDevice(0));
	}

	void time_step(int step, Cell *cells) {
		set_seed();
		CudaSafeCall(cudaSetDevice(1));
		CudaSafeCall(cudaMemcpy(&sol[0], &results[(step-1)*Nx], Nx*sizeof(float), cudaMemcpyDeviceToDevice));

		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

		for (int n = 0; n < maxT-1; n++) {
			int next = (n+1)*Nx;
			int cur = n*Nx;

			double influx = fabsf(rand() / (double) RAND_MAX);
			int num_cells_absorb = (int) (fabsf(ceilf(rand() % (Nx+1))) * diffusion);
			double amount_per_cell = influx / (double) num_cells_absorb;

			space_step<<< blocks, threads >>>(sol, cur, next, T_scale, dt, s, liquid, influx,
							  num_cells_absorb, amount_per_cell, cells, carcin_idx);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
		}

		CudaSafeCall(cudaMemcpy(&results[step*Nx], &sol[(maxT-1)*Nx], Nx*sizeof(float), cudaMemcpyDeviceToDevice));

		CudaSafeCall(cudaSetDevice(0));
	}

	__device__ float get(int cell, int time) {
		return results[time*Nx+cell];
	}
};

#endif // __CARCINOGEN_PDE_H__
