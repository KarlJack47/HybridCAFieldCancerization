#ifndef __CARCIN_PDE_H__
#define __CARCIN_PDE_H__

struct CarcinPDE {
    int device;
    unsigned carcinIdx;
    unsigned N;
    unsigned maxIter;

    double ic;
    double bc;
    double diffusion;
    double influx;
    double outflux;

    double *soln;

    CarcinPDE(int dev, unsigned idx, unsigned spaceSize, double diff,
              double influxIn, double outfluxIn, double icIn, double bcIn)
    {
        device = dev;
        N = spaceSize;
        diffusion = diff;
        influx = influxIn;
        outflux = outfluxIn;
        ic = icIn;
        bc = bcIn;
        carcinIdx = idx;
        maxIter = 100;

        CudaSafeCall(cudaMallocManaged((void**)&soln, N*N*sizeof(double)));
    }

    void prefetch_memory(int dev)
    {
        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(soln,
                                          N*N*sizeof(double),
                                          dev, NULL));
    }

    void init(double cellVolume, unsigned blockSize)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);

        init_pde<<< blocks, threads >>>(soln, ic, bc, N, cellVolume);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
	}

    void free_resources(void)
    {
        if (soln != NULL) {
            CudaSafeCall(cudaFree(soln)); soln = NULL;
        }
    }

    void time_step(unsigned step, double cellVolume, unsigned blockSize,
                   cudaStream_t stream)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);

        pde_space_step<<< blocks, threads, 0, stream >>>(
            soln, step, N, maxIter, bc, ic,
            diffusion, influx, outflux, cellVolume
        );
        CudaCheckError();
    }
};

#endif // __CARCIN_PDE_H__