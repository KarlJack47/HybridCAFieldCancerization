#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

// Copy gpu bitmap to unsigned char array so that pngs can be created.
__global__ void copy_frame(uchar4 *optr, unsigned char *frame)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned dim = gridDim.x * blockDim.x;
    unsigned idx = x + ((dim - 1) - y) * dim;

    frame[4*dim*y+4*x] = optr[idx].x;
    frame[4*dim*y+4*x+1] = optr[idx].y;
    frame[4*dim*y+4*x+2] = optr[idx].z;
    frame[4*dim*y+4*x+3] = optr[idx].w;
}

// Initalizes the carcinogen pde grid.
__global__ void init_pde(double *soln, double ic, double bc, unsigned N,
                         double cellVolume)
{
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x + y * gridDim.x * blockDim.x;

    if (!(x < N && y < N)) return;

    if (x == 0 || x == N-1 || y == 0 || y == N-1)
        soln[idx] = cellVolume * bc;
    else
        soln[idx] = cellVolume * ic;
}

// Spacial step for the carcinogen pde.
__global__ void pde_space_step(double *soln, unsigned t, unsigned N,
                               unsigned maxIter, double bc, double ic, double D,
                               double influx, double outflux, double cellVolume)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * gridDim.x * blockDim.x;
    unsigned n, m;
    double piDivN = M_PI / (double) N, piSquared = M_PI * M_PI,
           srcTerm = influx - outflux,
           icMinBc = ic - bc, NSquared = N * N, Dt = D * t;
    double sum, nOdd, mOdd, lambda, expResult;

    if (!(x < N && y < N)) return;

    if (!(x == 0 || x == N-1 || y == 0 || y == N-1)) {
        sum = 0.0;
        for (n = 0; n < maxIter; n++) {
            nOdd = 2.0 * n + 1.0;
            for (m = 0; m < maxIter; m++) {
                mOdd = 2.0 * m + 1.0;
                lambda = ((nOdd * nOdd + mOdd * mOdd) * piSquared) / NSquared;
                expResult = exp(-lambda * Dt);
                sum += ((srcTerm * (1.0 - expResult) / (lambda * D)
                         + expResult * icMinBc) * sin(x * nOdd * piDivN)
                        * sin(y * mOdd * piDivN)) / (nOdd * mOdd);
            }
        }
        soln[idx] = cellVolume * ((16.0 / piSquared) * sum + bc);
    } else soln[idx] = cellVolume * bc;
}

// CA related kernels
__global__ void cells_gpu_to_gpu_copy(Cell *src, Cell *dst, unsigned gSize,
                                      unsigned nGenes)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x + y * blockDim.x * gridDim.x;
	unsigned i;

    if (!(x < gSize && y < gSize)) return;

    dst[idx].state = src[idx].state;
    dst[idx].age = src[idx].age;

    for (i = 0; i < src[idx].params->nPheno; i++)
        dst[idx].phenotype[i] = src[idx].phenotype[i];
    for (i = 0; i < nGenes; i++) {
        dst[idx].geneExprs[i*2] = src[idx].geneExprs[i*2];
        dst[idx].geneExprs[i*2+1] = src[idx].geneExprs[i*2+1];
        dst[idx].bOut[i] = src[idx].bOut[i];
    }
}

__global__ void init_grid(Cell *prevG, unsigned gSize, GeneExprNN *NN,
                          unsigned numIter)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned i, j;
    curandState_t rndState;
    double rnd, *out = NULL;

    if (!(x < gSize && y < gSize)) return;

    if (prevG[idx].state == EMPTY) return;

    curand_init((unsigned long long) clock(), idx, 0, &rndState);

    for (i = 0; i < NN->nOut; i++) {
        rnd = curand_uniform_double(&rndState) * (0.1 - 0.0) + 0.0;
        prevG[idx].geneExprs[i*2] = rnd;
        rnd = curand_uniform_double(&rndState) * (0.1 - 0.0) + 0.0;
        prevG[idx].geneExprs[i*2+1] = rnd;
    }

    out = (double*)malloc(NN->nOut*sizeof(double));
    memset(out, 0, NN->nOut*sizeof(double));
    for (i = 0; i < numIter; i++) {
        for (j = 0; j < NN->nOut; j++) {
            rnd = curand_uniform_double(&rndState) * (0.001 - 0.0) + 0.0;
            out[j] = rnd;
        }
        prevG[idx].mutate(NN, out, &rndState);
    }
}

__global__ void mutate_grid(Cell *prevG, unsigned gSize, GeneExprNN *NN,
                            CarcinPDE *pdes, unsigned t)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned i;
    curandState_t rndState;
    double *in = NULL, *out = NULL;

    if (!(x < gSize && y < gSize)) return;

    in = (double*)malloc(NN->nIn*sizeof(double));
    out = (double*)malloc(NN->nOut*sizeof(double));
    memset(in, 0, NN->nIn*sizeof(double));
    memset(out, 0, NN->nOut*sizeof(double));
    for (i = 0; i < NN->nIn-1; i++)
        in[i] = pdes[i].soln[idx];
    in[NN->nIn-1] = prevG[idx].age;
    NN->evaluate(in, out, prevG[idx].bOut);
    free(in); in = NULL;

    curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
    prevG[idx].mutate(NN, out, &rndState);
    free(out); out = NULL;
}

__global__ void check_CSC_or_TC_formed(Cell *newG, Cell *prevG, unsigned gSize,
                                       unsigned t, unsigned *cscFormed,
                                       unsigned *tcFormed, unsigned exciseCount,
                                       unsigned timeTCDead)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;

    if (!(x < gSize && y < gSize)) return;

    if (prevG[idx].state != CSC && newG[idx].state == CSC && !newG[idx].moved) {
        if (*cscFormed == 1) {
            printf("A CSC formed at %d, (%d, %d).\n", t, y, x);
            return;
        }

        if (atomicCAS(cscFormed, 0, 1) == 1) {
            printf("A CSC formed at %d, (%d, %d).\n", t, y, x);
            return;
        }

        printf("The first CSC formed at %d, (%d, %d).\n", t, y, x);

        return;
    }
    if (prevG[idx].state != TC && newG[idx].state == TC & !newG[idx].moved) {
        if (tcFormed[exciseCount] == 1) {
            printf("A TC was formed at %d, (%d, %d).\n", t, y, x);
            return;
        }

        if (atomicCAS(&tcFormed[exciseCount], 0, 1) == 1) {
            printf("A TC was formed at %d, (%d, %d).\n", t, x, y);
            return;
        }

        if (exciseCount == 0) {
            printf("The first TC formed at %d, (%d, %d).\n", t, y, x);
            return;
        }

        printf("A TC recurred in %d steps after excision %d at %d, (%d, %d).\n",
               exciseCount, timeTCDead, t, y, x);
    }
}

__global__ void reset_rule_params(Cell *prevG, Cell *newG, unsigned gSize)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;

    if (!(x < gSize && y < gSize)) return;

    prevG[idx].chosenPheno = -1;
    prevG[idx].chosenCell = -1;
    prevG[idx].age++;
    prevG[idx].cellRebirth = false;
    newG[idx].moved = false;
    *newG[idx].inUse = 0;
    *prevG[idx].inUse = 0;
    *prevG[idx].actionApplied = 0;
    *prevG[idx].actionDone = 0;
}

__global__ void rule(Cell *newG, Cell *prevG, unsigned gSize,
                     unsigned nGenes, unsigned t, unsigned *count, bool kill)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned i, j;
    unsigned numEmpty, rnd, neighIdx, *rndIdx = NULL;
    int state = -2;
    curandState_t rndState;

    if (!(x < gSize && y < gSize)) return;

    if (kill && prevG[idx].state != TC && prevG[idx].state != CSC)
        return;

    if (kill && prevG[idx].chosenPheno == -1) return; // new cell

    if (*prevG[idx].actionDone == 1)
        return;

    if (prevG[idx].state == EMPTY) {
        if (!kill) atomicAdd(&count[0], 1);
        return;
    }

    if (prevG[idx].state == TC) {
        if (!kill) atomicAdd(&count[7], 1);
    }

    curand_init((unsigned long long) clock(), idx+t, 0, &rndState);

    if (prevG[idx].chosenPheno == -1) {
        prevG[idx].chosenPheno = prevG[idx].get_phenotype(&rndState);
        /*if (prevG[idx].state == TC
         && prevG[idx].params->phenoInit[TC*4+1] != prevG[idx].phenotype[1])
            printf("%d, %d, Prolif: %g, Quies: %g, Apop: %g, Diff: %g\n",
                   idx, prevG[idx].chosenPheno,
                   prevG[idx].phenotype[0], prevG[idx].phenotype[1],
                   prevG[idx].phenotype[2], prevG[idx].phenotype[3]);*/
        /*if (prevG[idx].state == SC && prevG[idx].geneExprs[0] != 0.0)
            printf("%d, (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g), (%.3g, %.3g)\n",
                   idx, prevG[idx].geneExprs[0], prevG[idx].geneExprs[1],
                   prevG[idx].geneExprs[ 2], prevG[idx].geneExprs[ 3],
                   prevG[idx].geneExprs[ 4], prevG[idx].geneExprs[ 5],
                   prevG[idx].geneExprs[ 6], prevG[idx].geneExprs[ 7],
                   prevG[idx].geneExprs[ 8], prevG[idx].geneExprs[ 9],
                   prevG[idx].geneExprs[10], prevG[idx].geneExprs[11],
                   prevG[idx].geneExprs[12], prevG[idx].geneExprs[13],
                   prevG[idx].geneExprs[14], prevG[idx].geneExprs[15],
                   prevG[idx].geneExprs[16], prevG[idx].geneExprs[17],
                   prevG[idx].geneExprs[18], prevG[idx].geneExprs[19]);*/
    }

    if (prevG[idx].chosenPheno == APOP && !kill) {
        newG[idx].apoptosis(nGenes);
        atomicCAS(prevG[idx].actionDone, 0, 1);
        atomicAdd(&count[3], 1);
        if (prevG[idx].state == TC)
            atomicAdd(&count[11], 1);
        return;
    }

    if (prevG[idx].state != CSC && prevG[idx].state != TC) {
        numEmpty = 0;
        for (i = 0; i < prevG[idx].params->nNeigh; i++)
             if (prevG[prevG[idx].neigh[i]].state == EMPTY) {
                 numEmpty++;
                 break;
             }
        if (numEmpty == 0) {
            if (!kill) {
                atomicAdd(&count[5], 1);
                atomicAdd(&count[6], 1);
            }
            return;
        }
    } else {
        unsigned numTCORCSC = 0;
        for (i = 0; i < prevG[idx].params->nNeigh; i++)
            if (prevG[prevG[idx].neigh[i]].state == TC
             || prevG[prevG[idx].neigh[i]].state == CSC)
                numTCORCSC++;
        if (numTCORCSC == prevG[idx].params->nNeigh) {
            if (!kill)
                atomicAdd(&count[9], 1);
            return;
        }
    }

    rndIdx = (unsigned*)malloc(prevG[idx].params->nNeigh*sizeof(unsigned));
    memset(rndIdx, 8, prevG[idx].params->nNeigh*sizeof(unsigned));
    j = 0;
    while (j != 8) {
        rnd = (unsigned) (ceil(curand_uniform_double(&rndState)
            * prevG[idx].params->nNeigh)) % prevG[idx].params->nNeigh;
        for (i = 0; i < prevG[idx].params->nNeigh; i++)
            if (rndIdx[i] == rnd) break;
        if (i < 8) continue;
        rndIdx[j++] = rnd;
    }
    __syncthreads();

    for (i = 0; i < prevG[idx].params->nNeigh; i++) {
        neighIdx = prevG[idx].neigh[rndIdx[i]];

        if (*prevG[idx].actionDone == 1) continue;
        if (*prevG[neighIdx].actionApplied == 1) continue;

        if (prevG[neighIdx].state != EMPTY && prevG[idx].state != CSC
         && prevG[idx].state != TC)
            continue;

        if ((prevG[neighIdx].state != EMPTY && !kill)
          || (prevG[neighIdx].state == EMPTY && kill))
            continue;

        if (prevG[neighIdx].state != EMPTY && (prevG[neighIdx].state == TC
         || prevG[neighIdx].state == CSC))
            continue;

        if (atomicCAS(prevG[neighIdx].inUse, 0, 1) == 1) {
            if (*prevG[neighIdx].actionApplied == 0)
                i--;
            continue;
        }

        if (*prevG[neighIdx].actionApplied == 1) continue;

        if (prevG[idx].chosenPheno == PROLIF) {
            state = newG[idx].proliferate(&newG[neighIdx], &rndState, nGenes);
            if (state != -2) {
                prevG[idx].cellRebirth = true;
                atomicAdd(&count[1], 1);
            }
        }
        else if (prevG[idx].chosenPheno == DIFF) {
            state = newG[idx].differentiate(&newG[neighIdx], &rndState, nGenes);
            if (state == -1) {
                atomicCAS(prevG[neighIdx].inUse, 1, 0);
                break;
            }
            if (state != -2) {
                prevG[idx].cellRebirth = true;
                atomicAdd(&count[4], 1);
            }
        }
        else if (prevG[idx].chosenPheno == QUIES) {
            state = newG[idx].move(&newG[neighIdx], &rndState, nGenes);
            if (state != -2) {
                atomicAdd(&count[2], 1);
                newG[neighIdx].moved = true;
            }
        }

        if (state != -2) {
            atomicCAS(prevG[neighIdx].actionApplied, 0, 1);
            atomicCAS(prevG[idx].actionDone, 0, 1);
            if (prevG[idx].state == TC)
                atomicAdd(&count[10], 1);
            break;
        }
        atomicCAS(prevG[neighIdx].inUse, 1, 0);
    }
    free(rndIdx); rndIdx = NULL;

    if (*prevG[idx].actionDone == 0) {
        if (prevG[idx].state == TC && kill)
            atomicAdd(&count[8], 1);
        if ((prevG[idx].state == TC || prevG[idx].state == CSC) && kill) {
            atomicAdd(&count[5], 1);
            return;
        }
        atomicAdd(&count[5], 1);
    }
}

__global__ void update_states(Cell *G, unsigned gSize, unsigned nGenes)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x + y * blockDim.x * gridDim.x;
	unsigned i;
    // Instead one could look at certain needing to be positively mutated
	unsigned minMut = 2;
	unsigned numMut = 0;

	if (!(x < gSize && y < gSize)) return;

    if (G[idx].state == CSC || G[idx].state == TC || G[idx].state == EMPTY)
        return;

    for (i = 0; i < nGenes; i++) {
        if (G[idx].positively_mutated((gene) i))
            numMut++;
        if (numMut == minMut) break;
    }

    if (numMut != minMut && G[idx].state == MNC)
        G[idx].change_state(NC);
    else if (numMut != minMut && G[idx].state == MSC)
        G[idx].change_state(SC);
    else if (numMut == minMut && G[idx].state == NC)
        G[idx].change_state(MNC);
    else if (numMut == minMut && G[idx].state == SC)
        G[idx].change_state(MSC);
}

__global__ void tumour_excision(Cell *newG, unsigned gSize, unsigned nGenes)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned i, neighIdx;

    if (!(x < gSize && y < gSize)) return;

    if (newG[idx].state != TC) return;

    for (i = 0; i < newG[idx].params->nNeigh; i++) {
        neighIdx = newG[idx].neigh[i];

        if (newG[neighIdx].state == TC) continue;

        if (atomicCAS(newG[neighIdx].inUse, 0, 1) == 1)
            continue;

        newG[neighIdx].apoptosis(nGenes);
    }

    newG[idx].apoptosis(nGenes);
}

__global__ void display_ca(uchar4 *optr, Cell *grid, unsigned gSize,
                           unsigned cellSize, unsigned dim, dim3 *stateColors)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned optrOffset = x * cellSize + y * cellSize * dim;
	unsigned i, j;

	if (!(x < gSize && y < gSize)) return;

    for (i = optrOffset; i < optrOffset + cellSize; i++)
        for (j = i; j < i + cellSize * dim; j += dim) {
            optr[j].x = stateColors[grid[idx].state].x;
            optr[j].y = stateColors[grid[idx].state].y;
            optr[j].z = stateColors[grid[idx].state].z;
            optr[j].w = 255;
        }
}

__global__ void display_genes(uchar4 *optr, Cell *G, unsigned gSize,
                              unsigned cellSize, unsigned dim,
                              unsigned nGenes, dim3 *geneColors)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned optrOffset = x * cellSize + y * cellSize * dim;
    unsigned i, j;
    gene M = (gene) 0;
    dim3 color(255, 255, 255);

    if (!(x < gSize && y < gSize)) return;

    for (i = 0; i < nGenes; i++) {
        if (!G[idx].positively_mutated((gene) i)) continue;

        if (!G[idx].positively_mutated(M)) {
            M = (gene) i;
            continue;
        }

        if (abs(G[idx].geneExprs[i*2] - G[idx].geneExprs[i*2+1])
          > abs(G[idx].geneExprs[M*2] - G[idx].geneExprs[M*2+1]))
			M = (gene) i;
    }

    if (G[idx].state != 0 && G[idx].state != 2 && G[idx].state != 6
     && G[idx].positively_mutated(M)) {
        color.x = geneColors[M].x;
        color.y = geneColors[M].y;
        color.z = geneColors[M].z;
    }

    for (i = optrOffset; i < optrOffset + cellSize; i++)
        for (j = i; j < i + cellSize * dim; j += dim) {
            optr[j].x = color.x;
            optr[j].y = color.y;
            optr[j].z = color.z;
            optr[j].w = 255;
        }
}

__global__ void display_carcin(uchar4 *optr, CarcinPDE *pde,
                               unsigned gSize, unsigned cellSize,
                               unsigned dim)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned optrOffset = x * cellSize + y * cellSize * dim;
    unsigned i, j;
    double carcinCon;

    if (!(x < gSize && y < gSize)) return;

    carcinCon = pde->soln[idx];

    for (i = optrOffset; i < optrOffset + cellSize; i++)
        for (j = i; j < i + cellSize * dim; j += dim) {
            optr[j].x = ceil(min(255.0, 255.0*carcinCon));
            optr[j].y = ceil(min(255.0, 255.0*carcinCon));
            optr[j].z = ceil(min(255.0, 255.0*carcinCon));
            optr[j].w = 255;
        }
}

__global__ void display_cell_data(uchar4 *optr, Cell *G, unsigned cellIdx,
                                  unsigned dim, unsigned nGenes,
                                  dim3 *stateColors)
{
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * blockDim.x * gridDim.x;
    unsigned gene = x / (dim / (double) (2 * nGenes));
    unsigned scl = 100;
    unsigned height = y / (dim / (double) (2 * scl + 1));
    int up, down;

    if (!(x < dim && y < dim)) return;

    optr[idx].x = 255;
    optr[idx].y = 255;
    optr[idx].z = 255;
    optr[idx].w = 255;

    if (abs((int) scl - (int) height)
     == trunc(G[cellIdx].params->mutThresh * (double) scl)) {
        optr[idx].x = 248;
        optr[idx].y = 222;
        optr[idx].z = 126;
    }

    if (abs((int) scl - (int) height) == 0) {
        optr[idx].x = 0;
        optr[idx].y = 0;
        optr[idx].z = 0;
    }

    if (gene % 2 == 1) gene = floor(gene / 2.0);
    else return;

    up   = G[cellIdx].geneExprs[gene*2] * scl;
    down = G[cellIdx].geneExprs[gene*2+1] * scl;

    optr[idx].x = 248;
    optr[idx].y = 222;
    optr[idx].z = 126;

    if ((up < down && height < scl && (scl - height) <= down)
     || (up > down && height > scl && (height - scl) <= up)) {
        optr[idx].x = stateColors[G[cellIdx].state].x;
        optr[idx].y = stateColors[G[cellIdx].state].y;
        optr[idx].z = stateColors[G[cellIdx].state].z;
    }
}

#endif // __CUDA_KERNELS_H__
