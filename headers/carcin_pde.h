#ifndef __CARCIN_PDE_H__
#define __CARCIN_PDE_H__

__global__ void init_pde(double*,double,double,unsigned);
__global__ void pde_space_step(double*,double*,unsigned,unsigned,unsigned,
                               double,double,double,double,double,double,
                               double);

struct CarcinPDE {
    int device;
    unsigned carcinIdx, N, maxIter, t, nCycles;
    int maxTInflux, maxTNoInflux;

    double deltaxy, deltat, exposureTime;
    double diffusion, ic, bc, influx, outflux;

    double *soln, *maxVal;

    CarcinPDE(int dev, unsigned idx, unsigned spaceSize, double diff,
              double influxIn, double outfluxIn, double icIn, double bcIn,
              int maxtinflux, int maxtnoinflux,double cellVolume,
              double cellCycleLen, double exposuretime)
    {
        device = dev;
        N = spaceSize;
        nCycles = 1;
        deltaxy = pow(cellVolume, 1.0/3.0);
        deltat = cellCycleLen;
        exposureTime = exposuretime;
        maxTInflux = maxtinflux;
        maxTNoInflux = maxtnoinflux;
        diffusion = diff;
        influx = influxIn;
        outflux = outfluxIn;
        ic = icIn;
        bc = bcIn;
        carcinIdx = idx;
        maxIter = 100;
        t = 0;

        CudaSafeCall(cudaMallocManaged((void**)&soln, N*N*sizeof(double)));
        CudaSafeCall(cudaMallocManaged((void**)&maxVal, sizeof(double)));
        *maxVal = ic > bc ? ic : bc;
    }

    void prefetch_memory(int dev, cudaStream_t *stream)
    {
        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(soln,
                                          N*N*sizeof(double),
                                          dev, *stream));
    }

    void init(unsigned blockSize, cudaStream_t *stream)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);

        init_pde<<< blocks, threads, 0, *stream >>>(soln, ic, bc, N);
        CudaCheckError();
	}

    void free_resources(void)
    {
        if (soln != NULL) {
            CudaSafeCall(cudaFree(soln)); soln = NULL;
        }
        if (maxVal != NULL) {
            CudaSafeCall(cudaFree(maxVal)); maxVal = NULL;
        }
    }

    double set_influx(double tCurr, double maxTCurr)
    {
        double influxOut = influx;

        if (tCurr < maxTCurr)
            if (nCycles % 2 == 0)
                influxOut = 0;
        else {
            if (tCurr > maxTCurr) {
                influxOut *= (1.0 - maxTCurr / tCurr);
                if (nCycles % 2 == 0)
                    influxOut += influx;
            }
            nCycles++;
        }

        return influxOut;
    }

    void time_step(unsigned blockSize, cudaStream_t *stream)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);
        double influxIn = influx;
        double tCurr = t * deltat, tPrev = tCurr - deltat,
               maxTCurr, maxTPrev, maxTNoCurr, maxTNoPrev;

        if (maxTInflux != -1)
            maxTInflux *= exposureTime;
        if (maxTNoInflux != -1)
            maxTNoInflux *= exposureTime;

        maxTCurr = nCycles * maxTInflux;
        maxTPrev = maxTCurr - maxTInflux;
        maxTNoCurr = nCycles * maxTNoInflux;
        maxTNoPrev = maxTNoCurr - maxTNoInflux;

        if (maxTInflux != -1 && maxTNoInflux != -1) {
            if (tCurr >= maxTCurr + maxTNoCurr) {
                nCycles++;
                maxTPrev = maxTCurr;
                maxTCurr += maxTInflux;
                maxTNoPrev = maxTNoCurr;
                maxTNoCurr += maxTNoInflux;
            }
            if (tCurr - maxTNoPrev >= maxTCurr) {
                if (tPrev - maxTNoPrev < maxTCurr
                 && tCurr - maxTNoPrev > maxTCurr)
                    influxIn *= (1.0 - maxTCurr / (tCurr - maxTNoPrev));
                else
                    influxIn = 0.0;
            } else
                if (tPrev - maxTPrev < maxTNoPrev
                 && tCurr - maxTPrev > maxTNoPrev) {
                    influxIn *= (1.0 - maxTNoPrev / (tCurr - maxTPrev));
                    influxIn += influx;
                }
        }
        if (maxTInflux != -1 && maxTNoInflux == -1)
            influxIn = set_influx(tCurr, maxTCurr);
        if (maxTInflux == -1 && maxTNoInflux != -1)
            influxIn = set_influx(tCurr, maxTNoCurr);

        pde_space_step<<< blocks, threads, 0, *stream >>>(
            soln, maxVal, t, N, maxIter, bc, ic,
            diffusion, influxIn, outflux, deltaxy, deltat
        );
        CudaCheckError();
    }
};

#endif // __CARCIN_PDE_H__