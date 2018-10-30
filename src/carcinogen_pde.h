#ifndef __CARCINOGEN_PDE_H__
#define __CARCINOGEN_PDE_H__

#include "cell.h"

__global__ void initialize(float *results, float ic, float bc, int Nx, int N, int T) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * blockDim.x * gridDim.x;

	for (int n = 0; n < T; n++) {
		if (row == 0 || row == N-1 || col == 0 || col == N-1) {
			results[n*Nx+idx] = bc;
		} else {
			results[n*Nx+idx] = ic;
		}
	}
}

__global__ void space_step(float *sol, int cur_t, int next_t, int Nx, int maxT, float T_scale, double dt, double s, bool liquid,
			   float influx, int num_cells_absorb, float amount_per_cell, bool *boundary_edge, Cell *cells, int carcin,
			   curandState_t* states) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * blockDim.x * gridDim.x;

	int next = next_t + idx;
	int cur = cur_t + idx;

	__shared__ float absorbed;
	if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
		absorbed = 0;
	}
	__syncthreads();

	if ((liquid == true && boundary_edge[idx] == true) || liquid == false)
		atomicAdd(&absorbed, amount_per_cell);
	__syncthreads();

	if (idx == 1) {
		if (liquid == true) {
			int in = 0;
			if (boundary_edge[idx] == true) in = influx - absorbed;
			sol[next] = sol[cur] +
				    s*(10*sol[cur-1] - 4*sol[cur+1] +
				    14*sol[cur+2] - 6*sol[cur+3] + sol[cur+4] - 15*sol[cur]) +
				    T_scale * dt * (fmaxf(0, in) - cells[idx].consumption[carcin]);
		} else
			sol[next] = sol[cur] +
				    s*(10*sol[cur-1] -
				    4*sol[cur+1] + 14*sol[cur+2] - 6*sol[cur+3] + sol[cur+4] - 15*sol[cur]) +
				    T_scale * dt * (fmaxf(0, influx - absorbed) - cells[idx].consumption[carcin]);
	} else if (idx == Nx-2) {
		if (liquid == true) {
			int in = 0;
			if (boundary_edge[idx] == true) in = influx - absorbed;
			sol[next] = sol[cur] +
				    s*(10*sol[cur] -
				    15*sol[cur-1] - 4*sol[cur-2] + 14*sol[cur-3] - 6*sol[cur-4] + sol[cur-5]) +
				    T_scale * dt * (fmaxf(0, in) - cells[idx].consumption[carcin]);
		} else
			sol[next] = sol[cur] +
				    s*(10*sol[cur] -
				    15*sol[cur-1] - 4*sol[cur-2] + 14*sol[cur-3] - 6*sol[cur-4] + sol[cur-5]) +
				    T_scale * dt * (fmaxf(0, influx - absorbed) - cells[idx].consumption[carcin]);
	} else if (idx != 0 && idx != Nx-1) {
		if (liquid == true) {
			int in = 0;
			if (boundary_edge[idx] == true) in = influx - absorbed;
			sol[next] = sol[cur] +
				    s*(16*sol[cur+1] + 16*sol[cur-1] -
				    sol[cur+2] - sol[cur-2] - 30*sol[cur]) +
				    T_scale * dt * (fmaxf(0, in) - cells[idx].consumption[carcin]);
		} else
			sol[next] = sol[cur] +
				    s*(16*sol[cur+1] + 16*sol[cur-1] -
				    sol[cur+2] - sol[cur-2] - 30*sol[cur]) +
				    T_scale * dt * (fmaxf(0, influx - absorbed) - cells[idx].consumption[carcin]);
	} else if (idx == 0 || idx == Nx-1) {
		if (liquid == true) {
			int in = 0;
			if (boundary_edge[idx] == true) in = influx - absorbed;
			sol[next] = sol[cur] +
				    s*(16*sol[(cur+1)%(Nx*maxT)] + 16*sol[abs((cur-1)%(Nx*maxT))] -
				    sol[(cur+2)%(Nx*maxT)] - sol[abs((cur-2)%(Nx*maxT))] - 30*sol[cur]) +
				    T_scale * dt * (fmaxf(0, in) - cells[idx].consumption[carcin]);
		} else
			sol[next] = sol[cur] +
				    s*(16*sol[(cur+1)%(Nx*maxT)] + 16*sol[abs((cur-1)%(Nx*maxT))] -
				    sol[(cur+2)%(Nx*maxT)] - sol[abs((cur-2)%(Nx*maxT))] - 30*sol[cur]) +
				    T_scale * dt * (fmaxf(0, influx - absorbed) - cells[idx].consumption[carcin]);
	}
}

__global__ void boundary_edge_calc(int step, int Nx, int N, float *results, bool *boundary_edge) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = row + col * blockDim.x * gridDim.x;

	int neighbourhood[8];
	neighbourhood[0] = row + abs((col+1) % N) * blockDim.x * gridDim.x; // n
	neighbourhood[1] = abs((row+1) % N) + abs((col+1) % N) * blockDim.x * gridDim.x; // ne
	neighbourhood[2] = abs((row+1) % N) + col * blockDim.x * gridDim.x; // e
	neighbourhood[3] = abs((row+1) % N) + abs((col-1) % N) * blockDim.x * gridDim.x; // se
	neighbourhood[4] = row + abs((col-1) % N) * blockDim.x * gridDim.x; // s
	neighbourhood[5] = abs((row-1) % N) + abs((col-1) % N) * blockDim.x * gridDim.x; // sw
	neighbourhood[6] = abs((row-1) % N) + col * blockDim.x * gridDim.x; // w
	neighbourhood[7] = abs((row-1) % N) + abs((col+1) % N) * blockDim.x * gridDim.y; // nw

	for (int i = 0; i < 8; i++) {
		if (results[(step-1)*Nx+neighbourhood[i]] != 0) {
			boundary_edge[idx] = true;
		} else {
			boundary_edge[idx] = false;
		}
	}
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
	unsigned int Nx;
	unsigned int maxT;
	double s;
	bool liquid;
	int carcin_idx;

	float *results;

	CarcinogenPDE(int space_size, int num_timesteps, double diff, bool liq, int idx) {
		N = space_size;
		T = num_timesteps + 1;
		T_scale = 16.0;
		diffusion = diff;
		dx = 1/(float)N;
		dt = 5e-4;
		liquid = liq;
		ic = 0.0f;
		if (liquid == true)
			bc = 1.0f;
		else
			bc = 0.0f;
		Nx = N/dx;
		maxT = 1/dt;
		s = diffusion * T_scale * dt / (12*dx*dx);
		carcin_idx = idx;
		CudaSafeCall(cudaSetDevice(1));
		CudaSafeCall(cudaMalloc((void**)&results, T*Nx*sizeof(float)));
		CudaSafeCall(cudaSetDevice(0));
	}

	void host_to_gpu_copy(int idx, CarcinogenPDE *dev_pde) {
		CarcinogenPDE *l_pde = (CarcinogenPDE*)malloc(sizeof(CarcinogenPDE));
                l_pde->N = N;
                l_pde->T = T;
                l_pde->T_scale = T_scale;
                l_pde->dx = dx;
                l_pde->dt = dt;
                l_pde->ic = ic;
                l_pde->bc = bc;
                l_pde->diffusion = diffusion;
                l_pde->Nx = Nx;
                l_pde->maxT = maxT;
                l_pde->s = s;
                l_pde->liquid = liquid;
		l_pde->carcin_idx = carcin_idx;

                float *dev_results;
                CudaSafeCall(cudaMalloc((void**)&dev_results, T*Nx*sizeof(float)));
                CudaSafeCall(cudaMemcpy(dev_results, results, T*Nx*sizeof(float), cudaMemcpyHostToDevice));
                l_pde->results = dev_results;

                CudaSafeCall(cudaMemcpy(&dev_pde[idx], &l_pde[0], sizeof(CarcinogenPDE), cudaMemcpyHostToDevice));

                free(l_pde);
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
		CudaSafeCall(cudaFree(results));
	}

	void time_step(int step, Cell *cells, curandState_t *states) {
		set_seed();
		CudaSafeCall(cudaSetDevice(1));
		float *sol;
	 	CudaSafeCall(cudaMalloc((void**)&sol, maxT*Nx*sizeof(float)));
		CudaSafeCall(cudaMemcpy(&sol[0], &results[(step-1)*Nx], Nx*sizeof(float), cudaMemcpyDeviceToDevice));
		float influx = fabsf((float) rand() / (float) RAND_MAX);
		int num_cells_absorb = (int) (fabsf(ceilf(rand() % (Nx+1))) * diffusion);
		float amount_per_cell = influx / (float) num_cells_absorb;

		dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		bool *boundary_edge;
		CudaSafeCall(cudaMalloc((void**)&boundary_edge, Nx*sizeof(bool)));
		boundary_edge_calc<<< blocks, threads >>>(step, Nx, N, results, boundary_edge);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		for (int n = 0; n < maxT-1; n++) {
			int next = (n+1)*Nx;
			int cur = n*Nx;
			space_step<<< blocks, threads >>>(sol, cur, next, Nx, maxT, T_scale, dt, s, liquid, influx,
							  num_cells_absorb, amount_per_cell, boundary_edge, cells, carcin_idx, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
		}

		CudaSafeCall(cudaMemcpy(&results[step*Nx], &sol[(maxT-1)*Nx], Nx*sizeof(float), cudaMemcpyDeviceToDevice));

		CudaSafeCall(cudaFree(sol));
		CudaSafeCall(cudaFree(boundary_edge));

		CudaSafeCall(cudaSetDevice(0));
	}

	__device__ float get(int cell, int time) {
		return results[time*Nx+cell];
	}
};

#endif // __CARCINOGEN_PDE_H__
