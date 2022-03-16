#ifndef __GENE_EXPR_NN_H__
#define __GENE_EXPR_NN_H__

__device__ void activation(unsigned idx, double *in, double alpha, double *out)
{
    /*  Computes the value of the ISQRU: f(x) = x/sqrt(1 + alpha*x^2).
        Inputs:
          idx: current element being computed
          in: array, the input array
          alpha: controls range of output, (-1/sqrt(alpha), 1/sqrt(alpha))
          out: array, the results of the computation are to be stored here
    */

    out[idx] = in[idx] / sqrt(1.0 + alpha * in[idx] * in[idx]);
}

__device__ double* dot(unsigned idx, double *m1, double *m2, double *out,
                       unsigned m1Rows , unsigned m1Cols, unsigned m2Cols)
{
    /*  Computes the product of two matrices: m1 x m2.
        Inputs:
          m1: array, left matrix of size m1_rows x m1_columns
          m2: array, right matrix of size m1_columns x m2_columns
          out: array, the results of the computation are to be stored here
          m1Rows: int, number of rows in the left matrix m1
          m1Cols: int, number of columns in the left matrix m1
          m2Cols: int, number of columns in the right matrix m2
    */

    unsigned i, k;
    int r, c;
    double tOut;

    for (i = idx; i < m1Rows * m2Cols; i += m2Cols * m1Rows) {
        r = i / m2Cols;
        c = i % m2Cols;
        tOut = 0.0;
        for(k = 0; k < m1Cols; k++)
            tOut += m1[r*m1Cols+k] * m2[k*m2Cols+c];
        out[i] = tOut;
	}

    return out;
}

__device__ double* vec_add(unsigned idx, double *m1, double *m2, double *out)
{
    /* Computes the (elementwise) addition between two arrays
       Inputs:
         m1: array
         m2: array
         output: array, the results of the computation are to be stored here
    */

    out[idx] = m1[idx] + m2[idx];

    return out;
}

__device__ void feedforward(double *in, double *WIn,
                            double *WOut, double *bOut, double *out,
                            unsigned nIn, unsigned nHid, unsigned nOut,
                            double alpha)
{
    unsigned i;
    double *hid = (double*)malloc(nHid*sizeof(double));
    memset(hid, 0, nHid*sizeof(double));

    for (i = 0; i < nHid; i++)
        activation(i, dot(i, WIn, in, hid, nHid, nIn, 1), alpha, hid);

    for (i = 0; i < nOut; i++)
        activation(i, vec_add(i, dot(i, WOut, hid, out, nHid, nOut, 1), bOut, out),
                   alpha, out);

    free(hid); hid = NULL;
}

struct GeneExprNN {
    int device;
    double *WIn;
    double *WOut;

    double alpha;
    double bias;

    unsigned nIn;
    unsigned nHid;
    unsigned nOut;

    GeneExprNN(int dev, unsigned nInput, unsigned nOutput,
               double alphaIn=1000000.0, double biasIn=0.001)
    {
        device = dev;

        alpha = alphaIn;
        bias = biasIn;

        nIn = nInput;
        nHid = nOutput;
        nOut = nOutput;
    }

    void memory_allocate(double *Wx, double *Wy)
    {
        size_t dbl = sizeof(double);

        CudaSafeCall(cudaMallocManaged((void**)&WIn, nHid*nIn*dbl));
        CudaSafeCall(cudaMallocManaged((void**)&WOut, nHid*nOut*dbl));

        memcpy(WIn, Wx, nHid*nIn*dbl);
        memcpy(WOut, Wy, nHid*nOut*dbl);
    }

    void free_resources(void)
    {
        if (WIn != NULL) {
            CudaSafeCall(cudaFree(WIn)); WIn = NULL;
        }
        if (WOut != NULL) {
            CudaSafeCall(cudaFree(WOut)); WOut = NULL;
        }
    }

    void prefetch_memory(int dev, cudaStream_t *stream)
    {
        size_t dbl = sizeof(double);

        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(WIn, nHid*nIn*dbl, dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(WOut, nHid*nOut*dbl, dev, *stream));
    }

    __device__ void evaluate(double *in, double *out, double *bOut,
                             double chanceUpreg, curandState_t *rndState)
    {
        unsigned i;
        double *WInTemp = (double*)malloc(nHid*nIn*sizeof(double));
        curandState_t localState = *rndState;

        memcpy(WInTemp, WIn, nHid*nIn*sizeof(double));
        for (i = 0; i < nHid; i++)
            if (curand_uniform_double(&localState) > chanceUpreg)
                WInTemp[i*nIn+(nIn-1)] *= -1;

        feedforward(in, WInTemp, WOut, bOut, out, nIn, nHid, nOut, alpha);

        *rndState = localState;
        free(WInTemp); WInTemp = NULL;
    }

    __device__ void mutate(double *bOut, double *geneExprs, double mutThresh)
    {
        unsigned i;
        double geneExpr;

        for (i = 0; i < nOut; i++) {
            geneExpr = geneExprs[i];
            // upregulated + mutated
            if (geneExpr > 0.0 && geneExpr >= mutThresh)
                bOut[i] = bias;
            // downregualted + mutated
            else if (geneExpr < 0.0 && abs(geneExpr) >= mutThresh)
                bOut[i] = -bias;
            // non mutated
            else
                bOut[i] = 0.0;
        }
    }
};

#endif // __GENE_EXPR_NN_H__
