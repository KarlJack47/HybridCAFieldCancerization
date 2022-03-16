#ifndef __CARCIN_H__
#define __CARCIN_H__

__global__ void init_carcin(double*,double*,double,double,double,double,unsigned,
                            SensitivityFunc*,unsigned,unsigned,bool);
__global__ void carcin_space_step(double*,double*,unsigned,unsigned,unsigned,
                                  double,double,double,double,double,
                                  double,SensitivityFunc*,unsigned,unsigned);

__device__ double func1(double x, double y, double factor, unsigned N)
{
    return 1.0;
}

__device__ double func2(double x, double y, double factor, unsigned N)
{
    return 0.5 * (sin(2.0 * M_PI * x / (double) N) * cos(M_PI * y / (double) N)
                  + 1.0) * factor;
}

__device__ double func3(double x, double y, double factor, unsigned N)
{
    double mux = (N / 2.0) - 1.0, muy = mux,
           sigmax = (N / 15.0), sigmay = sigmax;

    return exp(-0.5 * (pow(x - mux, 2) / (sigmax * sigmax)
                     + pow(y - muy, 2) / (sigmay * sigmay))) * factor;
}

__device__ double func4(double x, double y, double factor, unsigned N)
{
    unsigned i;
    double sigmax = N / 15.0, sigmay = sigmax, Ndiv4 = N / 4.0, result = 0.0;
    double mux[5] = { N / 2.0 - 1.0, Ndiv4 - sigmax - 1.0,
                      N - Ndiv4 + sigmax - 1.0 }, muy[5];
    mux[3] = mux[1]; mux[4] = mux[2];
    muy[0] = mux[0]; muy[1] = mux[1]; muy[2] = mux[2];
    muy[3] = mux[2]; muy[4] = mux[1];

    for (i = 0; i < 5; i++)
        result += exp(-0.5 * (pow(x - mux[i], 2) / (sigmax * sigmax)
                     + pow(y - muy[i], 2) / (sigmay * sigmay)));

    return result * factor;
}

__device__ SensitivityFunc pFunc[4] = { func1, func2, func3, func4 };

struct Carcin {
    int device;
    unsigned carcinIdx, N, maxIter, t, nCycles;
    int maxTInflux, maxTNoInflux;

    double deltaxy, deltat, exposureTime;
    double diffusion, ic, bc, influx, outflux;
    SensitivityFunc *func; unsigned funcIdx, nFunc;
    unsigned type; // 0: only func[funcIdx], 1: only pde, 2: func[funcIdx] * pde

    double *soln, *maxVal;

    Carcin(int dev, unsigned idx, unsigned typeIn, unsigned spaceSize,
           double diff, double influxIn, double outfluxIn, double icIn,
           double bcIn, int maxtinflux, int maxtnoinflux, double cellDiameter,
           double cellCycleLen, double exposuretime,
           SensitivityFunc *funcIn, unsigned nfunc=4, unsigned funcidx=0)
    {
        carcinIdx = idx;
        type = typeIn;
        device = dev;
        N = spaceSize;
        nCycles = 1;
        deltaxy = cellDiameter;
        deltat = cellCycleLen;
        exposureTime = exposuretime;
        maxTInflux = maxtinflux;
        if (maxTInflux != -1) maxTInflux *= exposureTime;
        maxTNoInflux = maxtnoinflux;
        if (maxTNoInflux != -1) maxTNoInflux *= exposureTime;
        diffusion = diff;
        influx = influxIn;
        outflux = outfluxIn;
        ic = icIn;
        bc = bcIn;
        maxIter = 50;
        t = 0;
        nFunc = nfunc;
        funcIdx = 0;
        if (funcidx < nFunc) funcIdx = funcidx;

        CudaSafeCall(cudaMallocManaged((void**)&soln, N*N*sizeof(double)));
        CudaSafeCall(cudaMallocManaged((void**)&func,
                                       nFunc*sizeof(SensitivityFunc)));
        memcpy(func, funcIn, nFunc*sizeof(SensitivityFunc));
        CudaSafeCall(cudaMallocManaged((void**)&maxVal, sizeof(double)));
        *maxVal = 0.0;
    }

    void prefetch_memory(int dev, cudaStream_t *stream)
    {
        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(soln,
                                          N*N*sizeof(double),
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(func,
                                          nFunc*sizeof(SensitivityFunc),
                                          dev, *stream));
    }

    void init(unsigned blockSize, cudaStream_t *stream)
    {
        dim3 blocks(NBLOCKS(N, blockSize), NBLOCKS(N, blockSize));
        dim3 threads(blockSize, blockSize);

        init_carcin<<< blocks, threads, 0, stream ? *stream : 0 >>>(
            soln, maxVal, diffusion, ic, bc, influx - outflux, N, func,
            type == 1 ? 0 : funcIdx, type, maxTNoInflux != -1 ? true : false
        );
        CudaCheckError();
	}

    void free_resources(void)
    {
        if (soln != NULL) {
            CudaSafeCall(cudaFree(soln)); soln = NULL;
        }
        if (func != NULL) {
            CudaSafeCall(cudaFree(func)); func = NULL;
        }
        if (maxVal != NULL) {
            CudaSafeCall(cudaFree(maxVal)); maxVal = NULL;
        }
    }

    double set_influx(double tCurr, double maxTCurr, bool influxIn=true)
    {
        double influxOut = influx;

        if (tCurr < maxTCurr)
            if (nCycles % 2 == 0)
                influxOut = influxIn ? 0 : influx;
            else
                influxOut = influxIn ? influx : 0;
        else {
            if (tCurr > maxTCurr) {
                influxOut *= (1.0 - maxTCurr / tCurr);
                if (nCycles % 2 == influxIn ? 0 : 1)
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
            influxIn = set_influx(tCurr, maxTNoCurr, false);

        *maxVal = 0.0;
        carcin_space_step<<< blocks, threads, 0, *stream >>>(
            soln, maxVal, t, N, maxIter, bc, ic, diffusion,
            influxIn - outflux, deltaxy, deltat, func,
            type == 1 ? 0 : funcIdx, type
        );
        CudaCheckError();
    }
};

#endif // __CARCIN_H__