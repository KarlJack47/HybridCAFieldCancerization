#ifndef __CARCIN_PDE_H__
#define __CARCIN_PDE_H__

__global__ void init_pde(double*,double,double,unsigned);
__global__ void pde_space_step(double*,double*,unsigned,unsigned,unsigned,
                               double,double,double,double,double,double,
                               double);

struct CarcinPDE {
    int device;
    unsigned carcinIdx;
    unsigned N;
    unsigned maxIter;

    double deltaxy, deltat;
    double diffusion;
    double ic, bc;
    double influx, outflux;

    double *soln;
    double *maxVal;

    CarcinPDE(int dev, unsigned idx, unsigned spaceSize, double diff,
              double influxIn, double outfluxIn, double icIn, double bcIn,
              double cellVolume, double cellCycleLen)
    {
        device = dev;
        N = spaceSize;
        deltaxy = pow(cellVolume, 1.0/3.0);
        deltat = cellCycleLen;
        diffusion = diff;
        influx = influxIn;
        outflux = outfluxIn;
        ic = icIn;
        bc = bcIn;
        carcinIdx = idx;
        maxIter = 100;

        CudaSafeCall(cudaMallocManaged((void**)&soln, N*N*sizeof(double)));
        CudaSafeCall(cudaMallocManaged((void**)&maxVal, sizeof(double)));
        *maxVal = ic > bc ? ic : bc;
    }

    void prefetch_memory(int dev)
    {
        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(soln,
                                          N*N*sizeof(double),
                                          dev, NULL));
    }

    void init(unsigned blockSize)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);

        init_pde<<< blocks, threads >>>(soln, ic, bc, N);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
	}

    void free_resources(void)
    {
        if (soln != NULL) {
            CudaSafeCall(cudaFree(soln)); soln = NULL;
        }
    }

    void time_step(unsigned step, unsigned blockSize,
                   cudaStream_t *stream)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);

        pde_space_step<<< blocks, threads, 0, *stream >>>(
            soln, maxVal, step, N, maxIter, bc, ic,
            diffusion, influx, outflux, deltaxy, deltat
        );
        CudaCheckError();
    }
};

#endif // __CARCIN_PDE_H__