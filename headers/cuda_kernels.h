#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

// Carcinogen kernels
// Initalizes the carcinogen pde grid.
__global__ void init_carcin(double *soln, double *maxVal, double D,
                            double ic, double bc, double srcTerm, unsigned N,
                            SensitivityFunc *func, unsigned funcIdx, unsigned type,
                            bool noInflux=false)
{
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x * N + y;

    if (!(x < N && y < N)) return;

    if (type == 0) {
        noInflux ? soln[idx] = 0.0
                 : soln[idx] = (*func[funcIdx])(x, y, 1.0, N);
        atomicMax(maxVal, soln[idx]);
        return;
    }

    if (x == 0 || x == N-1 || y == 0 || y == N-1)
        noInflux ? soln[idx] = 0.0
                 : soln[idx] = (*func[funcIdx])(x, y, 1.0, N) * bc;
    else
        noInflux ? soln[idx] = 0.0
                 : soln[idx] = (*func[funcIdx])(x, y, 1.0, N) * ic;

    atomicMax(maxVal, soln[idx]);
}

// Spatial step for a carcinogen
__global__ void carcin_space_step(double *soln, double *maxVal, unsigned t,
                                  unsigned N, unsigned maxIter, double bc,
                                  double ic, double D, double srcTerm,
                                  double deltaxy, double deltat,
                                  SensitivityFunc *func, unsigned funcIdx,
                                  unsigned type)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * N + y;
    unsigned n, m;
    double nOdd, mOdd, piDivN, piSquared, NSquared,
           tNonDimen, sum, lambda, expResult, alpha;

    if (!(x < N && y < N)) return;

    if (type == 0) {
        srcTerm <= 0.0 ? soln[idx] = 0.0
                       : soln[idx] = (*func[funcIdx])(x, y, 1.0, N);
        atomicMax(maxVal, soln[idx]);
        return;
    }

    piDivN = M_PI / (double) N; piSquared = M_PI * M_PI;
    NSquared = deltaxy * deltaxy * N * N;
    tNonDimen = (D * t * deltat) / NSquared;

    if (x != 0 && x != N-1 && y != 0 && y != N-1) {
        sum = 0.0;
        for (n = 1; n <= maxIter; n++) {
            nOdd = 2 * n - 1;
            for (m = 1; m <= maxIter; m++) {
                mOdd = 2 * m - 1;
                lambda = (nOdd * nOdd + mOdd * mOdd) * piSquared;
                expResult = exp(-lambda * tNonDimen);
                alpha = (abs(ic) * D) / (abs(srcTerm) * NSquared);
                sum += (sin(nOdd * piDivN * x) * sin(mOdd * piDivN * y) * (((1 - expResult) / lambda) + alpha * expResult))
                     / (nOdd * mOdd);
            }
        }
        soln[idx] = (*func[funcIdx])(x, y, 1.0, N) * ((16.0 / piSquared) * sum);
        if (soln[idx] < 0.0) soln[idx] = 0.0;
    } else soln[idx] = (*func[funcIdx])(x, y, 1.0, N) * bc;

    atomicMax(maxVal, soln[idx]);
}

// CA related kernels
// Copies a CA grid over to another grid
__global__ void cells_gpu_to_gpu_cpy(Cell *dst, Cell *src, unsigned gSize,
                                     unsigned nGenes)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i;

    if (!(x < gSize && y < gSize)) return;

    dst[idx].state = src[idx].state;
    dst[idx].age = src[idx].age;
    dst[idx].lineage = src[idx].lineage;
    dst[idx].fitness = src[idx].fitness;
    dst[idx].isTAC = src[idx].isTAC;
    dst[idx].canMove = src[idx].canMove;
    dst[idx].canKill = src[idx].canKill;
    dst[idx].nTACProlif = src[idx].nTACProlif;

    for (i = 0; i < src[idx].params->nPheno; i++)
        dst[idx].phenotype[i] = src[idx].phenotype[i];
    for (i = 0; i < nGenes; i++) {
        dst[idx].geneExprs[i] = src[idx].geneExprs[i];
        dst[idx].bOut[i] = src[idx].bOut[i];
    }
}

// Used when initializing the CA with already mutated tissue cells
__global__ void init_grid(Cell *prevG, unsigned gSize, GeneExprNN *NN,
                          unsigned numIter)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i, j;
    curandState_t rndState;
    double rnd, *out = NULL;

    if (!(x < gSize && y < gSize)) return;

    if (prevG[idx].state == EMPTY) return;

    curand_init((unsigned long long) clock(), idx, 0, &rndState);

    for (i = 0; i < NN->nOut; i++) {
        rnd = curand_uniform_double(&rndState) * (0.1 - 0.0) + 0.0;
        prevG[idx].geneExprs[i] = rnd;
    }

    out = (double*)malloc(NN->nOut*sizeof(double));
    for (i = 0; i < numIter; i++) {
        for (j = 0; j < NN->nOut; j++) {
            rnd = curand_uniform_double(&rndState) * (0.001 - 0.0) + 0.0;
            out[j] = rnd;
        }
        prevG[idx].mutate(NN, out, &rndState);
    }
    free(out); out = NULL;
}

// Used to reset the grid so that a new simulation can be run with the same parameters
__global__ void reset_grid(Cell *prevG, unsigned gSize, unsigned nGenes,
                           double *weightStates, unsigned rTC, unsigned centerX,
                           unsigned centerY, double cellLifeSpan,
                           double cellCycleLen)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i, nPheno;
    double maxAge = ceil(cellLifeSpan / cellCycleLen);
    curandState_t rndState;
    ca_state state;
    double rnd, S;

    if (!(x < gSize && y < gSize)) return;

    curand_init((unsigned long long) clock(), idx, 0, &rndState);

    S = weightStates[NC];
    rnd = curand_uniform_double(&rndState);
	if (rnd <= S)
	    prevG[idx].state = NC;
	else if (rnd <= (S += weightStates[MNC]))
	    prevG[idx].state = MNC;
	else if (rnd <= (S += weightStates[SC]))
	    prevG[idx].state = SC;
	else if (rnd <= (S += weightStates[MSC]))
	    prevG[idx].state = MSC;
	else if (rnd <= (S += weightStates[CSC]))
	    prevG[idx].state = CSC;
	else if (rnd <= (S += weightStates[TC]))
	    prevG[idx].state = TC;
	else prevG[idx].state = EMPTY;

	if (rTC != 0 && check_in_circle(x, y, gSize, rTC, centerX, centerY)) {
	    rnd = curand_uniform_double(&rndState);
	    S = 0.89; // percentage of TC
	    if (rnd <= S) prevG[idx].state = TC;
	    else if (rnd <= (S += 0.01)) prevG[idx].state = CSC;
	    else prevG[idx].state = EMPTY;
	}

    nPheno = prevG[idx].params->nPheno;
    state = prevG[idx].state;
    for (i = 0; i < nPheno; i++)
	    prevG[idx].phenotype[i] = prevG[idx].params->phenoInit[state*nPheno+i];

    if (state == EMPTY) prevG[idx].age = 0;
    else prevG[idx].age = ceil(curand_uniform_double(&rndState) * maxAge);

    for (i = 0; i < nGenes; i++) {
        prevG[idx].geneExprs[i] = 0.0;
        prevG[idx].bOut[i] = 0.0;
    }

    prevG[idx].numNeighTested = 0;
    for (i = 0; i < prevG[idx].params->nNeigh; i++)
        prevG[idx].testedNeigh[i] = gSize * gSize;
    prevG[idx].chosenPheno = -1; prevG[idx].chosenCell = -1;
    prevG[idx].cellRebirth = false; prevG[idx].moved = false;
    prevG[idx].canMove = false; prevG[idx].canKill = false;
    *prevG[idx].inUse = 0;
    *prevG[idx].actionApplied = 0; *prevG[idx].actionDone = 0;
    prevG[idx].excised = 0;
    prevG[idx].lineage = -1;
    prevG[idx].isTAC = false; prevG[idx].nTACProlif = 0;
    prevG[idx].fitness = 0.0;
}

__global__ void mutate_grid(Cell *prevG, unsigned gSize, GeneExprNN *NN,
                            Carcin *carcins, bool *activeCarcin, unsigned t)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i;
    curandState_t rndState;
    double *in = NULL, *out = NULL;

    if (!(x < gSize && y < gSize)) return;

    if (prevG[idx].state == EMPTY) return;

    in = (double*)malloc(NN->nIn*sizeof(double));
    out = (double*)malloc(NN->nOut*sizeof(double));
    memset(out, 0, NN->nOut*sizeof(double));
    for (i = 0; i < NN->nIn-1; i++)
        if (activeCarcin[i])
            in[i] = carcins[i].soln[idx];
        else in[i] = 0.0;
    in[NN->nIn-1] = prevG[idx].age;
    curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
    NN->evaluate(in, out, prevG[idx].bOut, prevG[idx].params->chanceUpreg,
                 &rndState);
    free(in); in = NULL;

    prevG[idx].mutate(NN, out, &rndState);
    free(out); out = NULL;
}

__global__ void update_states(Cell *prevG, Cell *newG, unsigned gSize,
                              unsigned nGenes, unsigned t, unsigned *counts,
                              bool doDediff=true)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x * gSize + y;
	unsigned i, mutInfo, neighIdx, numMut = 0, numSC = 0, numEmpty = 0, step = 0;
	ca_state newState = SC;
	curandState_t rndState;
	bool rndInit = false, bySC = false, byEmpty = false, byRand = false;

	if (!(x < gSize && y < gSize)) return;

    if (prevG[idx].state == CSC || prevG[idx].state == EMPTY)
        return;

    for (i = 0; i < nGenes; i++) {
        mutInfo = prevG[idx].positively_mutated((gene) i);
        if (mutInfo == 1 || mutInfo == 3)
            numMut++;
        if (numMut == prevG[idx].params->minMut) break;
    }

    // MNC -> NC
    if (numMut != prevG[idx].params->minMut && prevG[idx].state == MNC) {
        newG[idx].change_state(NC);
        if (counts != NULL) {
            atomicAdd(&counts[EMPTY], 1);
            atomicAdd(&counts[MNC], 1);
        }
    // MSC -> SC
    } else if (numMut != prevG[idx].params->minMut && prevG[idx].state == MSC) {
        newG[idx].change_state(SC);
        if (counts != NULL) {
            atomicAdd(&counts[EMPTY], 1);
            atomicAdd(&counts[MSC], 1);
        }
    // NC -> MNC
    } else if (numMut == prevG[idx].params->minMut && prevG[idx].state == NC) {
        newG[idx].change_state(MNC);
        if (counts != NULL) {
            atomicAdd(&counts[EMPTY], 1);
            atomicAdd(&counts[NC], 1);
        }
    } else if (numMut == prevG[idx].params->minMut && prevG[idx].state == SC) {
        curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
        rndInit = true;
        if (curand_uniform_double(&rndState) <= prevG[idx].params->chanceCSCForm / 2.0) {
            // SC -> CSC
            newG[idx].change_state(CSC);
            if (counts != NULL) atomicAdd(&counts[TC], 1);
        } else {
            // SC -> MSC
            newG[idx].change_state(MSC);
            if (counts != NULL) atomicAdd(&counts[SC], 1);
        }
        if (counts != NULL) atomicAdd(&counts[EMPTY], 1);
    // MSC -> CSC
    } else if (numMut == prevG[idx].params->minMut && prevG[idx].state == MSC) {
        curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
        rndInit = true;
        if (curand_uniform_double(&rndState) <= prevG[idx].params->chanceCSCForm) {
            newG[idx].change_state(CSC);
            if (counts != NULL) {
                atomicAdd(&counts[CSC], 1);
                atomicAdd(&counts[EMPTY], 1);
            }
        }
    }

    if (!doDediff) return;

    if (prevG[idx].state == SC || prevG[idx].state == MSC)
        return;
    if (prevG[idx].state == MNC) newState = MSC;
    if (prevG[idx].state == TC) newState = CSC;
    for (i = 0; i < prevG[idx].params->nNeigh; i++) {
        neighIdx = prevG[idx].neigh[i];
        if (prevG[idx].state == NC && prevG[neighIdx].state == SC)
            numSC++;
        if (prevG[idx].state == MNC && prevG[neighIdx].state == MSC)
            numSC++;
        if (prevG[idx].state == TC && prevG[neighIdx].state == CSC)
            numSC++;
        if (prevG[neighIdx].state == EMPTY) numEmpty++;
    }
    if (numEmpty >= prevG[idx].params->maxNumEmpty) byEmpty = true;
    if (numSC <= prevG[idx].params->minNumSC) bySC = true;
    if (!bySC && !byEmpty) {
        if (!rndInit) curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
        byRand = ((prevG[idx].isTAC && curand_uniform_double(&rndState) <= prevG[idx].params->chanceDediff / 4.0)
              || (!prevG[idx].isTAC && curand_uniform_double(&rndState) <= prevG[idx].params->chanceDediff / 8.0));
        if (!byRand) return;
    }
    if (!rndInit) curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
    if (byRand || (prevG[idx].isTAC && curand_uniform_double(&rndState) <= prevG[idx].params->chanceDediff)
     || (!prevG[idx].isTAC && curand_uniform_double(&rndState) <= prevG[idx].params->chanceDediff / 2.0)) {
        if (prevG[idx].isTAC) {
            newG[idx].adjust_phenotype(0, -newG[idx].params->chanceTACProlif);
            newG[idx].isTAC = false;
            newG[idx].nTACProlif = 0;
        }
        newG[idx].change_state(newState);
        prevG[idx].deDifferentiated = true;

        if (bySC && !byEmpty) step = 0;
        else if (byEmpty && !bySC) step = 1;
        else if (bySC && byEmpty) step = 2;
        else if (byRand) step = 3;
        atomicAdd(&counts[31+step], 1);
        if (!prevG[idx].isTAC) step += 4;
        if (newState == SC) atomicAdd(&counts[7+step], 1);
        else if (newState == MSC) atomicAdd(&counts[15+step], 1);
        else if (newState == CSC) atomicAdd(&counts[23+step], 1);
    }
}

__global__ void reset_rule_params(Cell *prevG, Cell *newG, unsigned gSize)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i;

    if (!(x < gSize && y < gSize)) return;

    prevG[idx].chosenPheno = -1;
    prevG[idx].chosenCell = -1;
    if (newG[idx].state != EMPTY) newG[idx].age++;
    prevG[idx].cellRebirth = false;
    prevG[idx].deDifferentiated = false;
    newG[idx].moved = false;
    prevG[idx].canKill = false;
    prevG[idx].canMove = false;
    newG[idx].canKill = false;
    newG[idx].canMove = false;
    *newG[idx].inUse = 0;
    *prevG[idx].inUse = 0;
    *prevG[idx].actionApplied = 0;
    *prevG[idx].actionDone = 0;
    *prevG[idx].checked = 0;
    prevG[idx].numNeighTested = 0;
    for (i = 0; i < prevG[idx].params->nNeigh; i++)
        prevG[idx].testedNeigh[i] = gSize * gSize;
    newG[idx].excised = 0;
}

__global__ void rule(Cell *newG, Cell *prevG, unsigned gSize,
                     unsigned nGenes, unsigned t, unsigned *count,
                     unsigned *countKills, unsigned *countTAC)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i, j;
    unsigned numEmpty = 0, numLessFit = 0;
    unsigned step, rnd, neighIdx, numTries = 0;
    bool inUse = false, inUseNeigh = false, actionApplied = false;
    int state = -2;
    curandState_t rndState;

    if (!(x < gSize && y < gSize)) return;

    if (prevG[idx].state == EMPTY) return;

    // another cell has already done an action to it
    if (*prevG[idx].actionApplied == 1) return;

    if (prevG[idx].deDifferentiated) return;

    curand_init((unsigned long long) clock(), idx+t, 0, &rndState);
    if (prevG[idx].chosenPheno == -1) {
        prevG[idx].chosenPheno = prevG[idx].get_phenotype(&rndState);
        if (prevG[idx].chosenPheno == QUIES
         && !(newG[idx].canMove = curand_uniform_double(&rndState)
              <= newG[idx].params->chanceMove))
            return;
        if ((prevG[idx].state == TC || prevG[idx].state == CSC)
         && prevG[idx].chosenPheno != APOP)
            newG[idx].canKill = curand_uniform_double(&rndState)
                             <= newG[idx].params->chanceKill;
    }
    __syncthreads();

    if (prevG[idx].chosenPheno == APOP) {
        while (inUse = atomicCAS(prevG[idx].inUse, 0, 1) != 0)
            if (*prevG[idx].actionApplied == 1) break;
        // killed by another cell already
        if (atomicCAS(prevG[idx].actionApplied, 0, 1) == 0) {
            newG[idx].apoptosis(nGenes);
            atomicAdd(&count[APOP], 1);
            atomicAdd(&count[(3*prevG[idx].state+4)+APOP], 1);
            if (prevG[idx].isTAC) {
                step = 0;
                if (prevG[idx].state == TC) step = 3;
                atomicAdd(&countTAC[(prevG[idx].state-step)*3+APOP], 1);
            }
        }
        if (!inUse) atomicExch(prevG[idx].inUse, 0);
        return;
    }
    __syncthreads();

    for (i = 0; i < prevG[idx].params->nNeigh; i++) {
        if (prevG[prevG[idx].neigh[i]].state == EMPTY)
            numEmpty++;
        else if (prevG[prevG[idx].neigh[i]].fitness < prevG[idx].fitness)
            numLessFit++;
    }
    if (numEmpty == 0 && (prevG[idx].chosenPheno == QUIES
     || numLessFit == 0) && !newG[idx].canKill) {
        return;
    }
    __syncthreads();

    do {
        for (i = 0; i < prevG[idx].params->nNeigh; i++) {
            rnd = (unsigned) (ceil(curand_uniform_double(&rndState)
                * prevG[idx].params->nNeigh)) % prevG[idx].params->nNeigh;
            neighIdx = prevG[idx].neigh[rnd];
            for (j = 0; j < prevG[idx].params->nNeigh; j++)
                if (prevG[idx].testedNeigh[j] == neighIdx) break;
            if (j < prevG[idx].params->nNeigh) continue;

            // stops TC or CSC from killing a TC or CSC by chance
            if (prevG[neighIdx].state != EMPTY && (prevG[neighIdx].state == TC
             || prevG[neighIdx].state == CSC) && prevG[neighIdx].fitness >= prevG[idx].fitness)
                continue;

            // only TC and CSC can kill during movement
            if (prevG[neighIdx].state != EMPTY && (prevG[idx].chosenPheno == QUIES
             || prevG[idx].fitness <= prevG[neighIdx].fitness)
             && !newG[idx].canKill) {
                prevG[idx].testedNeigh[prevG[idx].numNeighTested++] = neighIdx;
                continue;
            }

            inUse = false; inUseNeigh = false; actionApplied = false;
            if ((inUse = atomicCAS(prevG[idx].inUse, 0, 1) == 0)
             && *prevG[idx].actionApplied == 0
             && (inUseNeigh = atomicCAS(prevG[neighIdx].inUse, 0, 1) == 0)
             && *prevG[neighIdx].actionApplied == 0
             && *prevG[neighIdx].actionDone == 0) {
                if (prevG[idx].chosenPheno == PROLIF)
                    state = newG[idx].proliferate(&newG[neighIdx], &rndState,
                                                  gSize, nGenes, countKills);
                else if (prevG[idx].chosenPheno == DIFF)
                    state = newG[idx].differentiate(&newG[neighIdx], &rndState,
                                                    gSize, nGenes, countKills);
                else if (prevG[idx].chosenPheno == QUIES)
                    state = newG[idx].move(&newG[neighIdx], &rndState,
                                           gSize, nGenes, countKills);

                if (state > -1) {
                    atomicExch(prevG[neighIdx].actionApplied, 1);
                    atomicExch(prevG[idx].actionDone, 1);
                    prevG[idx].chosenCell = neighIdx;

                    if (prevG[idx].chosenPheno == PROLIF
                     || prevG[idx].chosenPheno == DIFF)
                        prevG[idx].cellRebirth = true;
                    else newG[neighIdx].moved = true;
                    atomicAdd(&count[prevG[idx].chosenPheno], 1);
                    atomicAdd(&count[prevG[idx].chosenPheno == DIFF
                                     ? prevG[idx].state+20
                                     : (3*prevG[idx].state+4)
                                        +prevG[idx].chosenPheno],
                              1);
                    if (prevG[idx].isTAC) {
                        step = 0;
                        if (prevG[idx].state == TC) step = 3;
                        atomicAdd(&countTAC[(prevG[idx].state-step)*3+prevG[idx].chosenPheno], 1);
                    }
                } else state = 7; // once first time failed to do action stop

                prevG[idx].testedNeigh[prevG[idx].numNeighTested++] = neighIdx;
            }
            if (inUseNeigh) {
                if (*prevG[idx].actionDone == 0
                 && (*prevG[neighIdx].actionDone == 1
                  || *prevG[neighIdx].actionApplied == 1))
                    prevG[idx].testedNeigh[prevG[idx].numNeighTested++] = neighIdx;
                atomicExch(prevG[neighIdx].inUse, 0);
            }
            if (inUse) {
                actionApplied = *prevG[idx].actionApplied == 1;
                atomicExch(prevG[idx].inUse, 0);
            }

            if (state != -2 || actionApplied
             || prevG[idx].numNeighTested == prevG[idx].params->nNeigh)
                break;
        }

        if (prevG[idx].numNeighTested == prevG[idx].params->nNeigh
         || actionApplied || state != -2)
            break;
        numTries++;
    } while (numTries != 100);
}


__global__ void update_lineage_data(Cell *newG, unsigned *cellLineage,
                                    bool *stateInLineage, unsigned *nLineage,
                                    unsigned *nLineageCells, unsigned gSize,
                                    bool checkNLineageStates=false)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned result, i;
    int64_t lineage;

    if (!(x < gSize && y < gSize)) return;

    if (newG[idx].state == EMPTY) return;

    // Note if idx = lineage then it is where lineage started but
    // it doesn't necessaryily mean that cell type started the lineage
    // just that cell location
    lineage = newG[idx].lineage;

    if (!checkNLineageStates && lineage != -1 && ((result = atomicAdd(&cellLineage[lineage], 1)) == 0
     || result != 0))
    {
        if (result == 0) atomicAdd(&nLineage[EMPTY], 1);
        stateInLineage[lineage*6+newG[idx].state] = true;
        atomicAdd(nLineageCells, 1);
    }

    if (checkNLineageStates && cellLineage[idx] != 0) {
        for (i = 0; i < 6; i++)
            if (stateInLineage[idx*6+i])
                atomicAdd(&nLineage[i], 1);

        if (stateInLineage[idx*6+MNC] || stateInLineage[idx*6+MSC]
         || stateInLineage[idx*6+CSC] || stateInLineage[idx*6+TC])
            atomicAdd(&nLineage[7], 1);
        if (stateInLineage[idx*6+NC] || stateInLineage[idx*6+SC])
            atomicAdd(&nLineage[8], 1);
    }
}

// Finds the largest lineages for a cell class, where maxIdx is index of the
// next maximum value to be found
__global__ void max_lineage(Cell *G, unsigned *cellLineage, unsigned *max,
                            unsigned maxIdx, unsigned stateIdx,
                            unsigned gSize)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    int64_t lineage;
    unsigned i;

    if (!(x < gSize && y < gSize)) return;

    if (stateIdx != EMPTY && G[idx].state != stateIdx) return;

    if ((lineage = G[idx].lineage) == -1) return;

    for (i = 0; i < maxIdx; i++)
        if (cellLineage[lineage] == max[i]) return;

    atomicMax(&max[maxIdx], cellLineage[lineage]);
}

__global__ void check_CSC_or_TC_formed(Cell *newG, Cell *prevG, unsigned gSize,
                                       unsigned t, unsigned *cscFormed,
                                       unsigned *tcFormed, unsigned exciseCount,
                                       unsigned *timeTCDead, bool perfectExcision,
                                       unsigned *r, unsigned *cX, unsigned *cY,
                                       unsigned *numTC)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;

    if (!(x < gSize && y < gSize)) return;

    if (exciseCount == 0 && prevG[idx].state == TC) atomicAdd(&numTC[0], 1);
    if (exciseCount == 0 && newG[idx].state == TC) atomicAdd(&numTC[1], 1);

    if (exciseCount == 0 && prevG[idx].state != CSC
     && newG[idx].state == CSC && !newG[idx].moved) {
        if (*cscFormed == 1 || atomicCAS(cscFormed, 0, 1) == 1)
            return;

        printf("The first CSC formed at %d, (%d, %d).\n", t, x, y);

        return;
    }

    if (*prevG[idx].checked) return;

    if (prevG[idx].state != TC && newG[idx].state == TC && !newG[idx].moved) {
        if (tcFormed[exciseCount] == 1
         || (!perfectExcision && !check_in_circle(x, y, gSize, r[exciseCount],
                                                  cX[exciseCount], cY[exciseCount]))
         || atomicCAS(&tcFormed[exciseCount], 0, 1) == 1) {
            *prevG[idx].checked = 1;
            return;
        }

        if (exciseCount == 0) {
            printf("The first TC formed at %d, (%d, %d).\n", t, x, y);
            *prevG[idx].checked = 1;
            return;
        }

        printf("A TC recurred in %d steps after excision %d at %d, (%d, %d).\n",
               timeTCDead[exciseCount], exciseCount, t, x, y);
        *prevG[idx].checked = 1;
    }
}

__global__ void tumour_excision(Cell *newG, bool removeField,
                                unsigned gSize, unsigned nGenes,
                                unsigned maxNeighDepth=1)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i, neighIdx, neighDepth;
    bool isMutatedCell = false, dontKill = false
    ;

    if (!(x < gSize && y < gSize)) return;

    if (newG[idx].state != NC && newG[idx].state != SC
     && newG[idx].state != EMPTY)
        isMutatedCell = true;

    if ((!removeField && newG[idx].state != TC)
     || (removeField && !isMutatedCell))
        return;

    newG[idx].apoptosis(nGenes);
    newG[idx].excised = 1;

    if (maxNeighDepth == 0) return;

    for (i = 0; i < newG[idx].params->nNeigh; i++) {
        neighDepth = 0;
        neighIdx = newG[idx].neigh[i];
        while (neighDepth != maxNeighDepth) {
            if ((!removeField && newG[neighIdx].state == TC)
             || (removeField && newG[neighIdx].state != NC
              && newG[neighIdx].state != SC
              && newG[neighIdx].state != EMPTY))
               dontKill = true;

            if (atomicCAS(newG[neighIdx].inUse, 0, 1) == 1)
                dontKill = true;

            if (!dontKill) {
                newG[neighIdx].apoptosis(nGenes);
                newG[neighIdx].excised = 1;
            }

            neighIdx = newG[neighIdx].neigh[i];
            neighDepth++;
            dontKill = false;
        }
    }
}

__global__ void set_excision_circle(Cell *newG, unsigned gSize,
                                    unsigned *rTC, int *tcX, int *tcY)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned count = 0, prevDepth, depth = 0, maxDepth = depth, i;
    Cell *curr;

    if (!(x < gSize && y < gSize)) return;

    if (newG[idx].state != TC || *tcY != -1) return;

    for (i = 0; i < newG[idx].params->nNeigh; i++)
        if (newG[newG[idx].neigh[i]].state == TC
         || newG[newG[idx].neigh[i]].state == CSC
         || newG[newG[idx].neigh[i]].state == EMPTY)
            count++;

    if (count >= (double) newG[idx].params->nNeigh * 1.1) {
        atomicCAS(tcX, -1, x);
        if (atomicCAS(tcY, -1, y) == -1) {
            curr = (Cell*)malloc(sizeof(Cell));
            for (i = 0; i < newG[idx].params->nNeigh; i++) {
                curr = &newG[idx]; depth = 0;
                do {
                    prevDepth = depth;
                    if (newG[curr->neigh[i]].state == TC
                     || newG[curr->neigh[i]].state == CSC
                     || newG[curr->neigh[i]].state == EMPTY) {
                        depth++;
                        curr = &newG[curr->neigh[i]];
                    }

                    if (depth == gSize) break;
                } while (prevDepth != depth);
                if (depth > maxDepth) maxDepth = depth;
                if (maxDepth == gSize) break;
            }
            free(curr); curr = NULL;
            atomicAdd(rTC, maxDepth+1);
            return;
        }
    }
}

__global__ void excision(Cell *newG, unsigned gSize, unsigned nGenes,
                         unsigned r, unsigned cX, unsigned cY, unsigned *numTC)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;

    if (!(x < gSize && y < gSize)) return;

    if (newG[idx].state == EMPTY) return;

    if (check_in_circle(x, y, gSize, r, cX, cY))
    {
        if (newG[idx].state == TC) atomicSub(&numTC[1], 1);
        newG[idx].apoptosis(nGenes);
        newG[idx].excised = 1;
    }
}

// CUDA kernels related to creating visualizations for the CA
__device__ void display_cell(uchar4 *optr, unsigned x, unsigned y, dim3 color,
                             unsigned cellSize, unsigned dim)
{
    unsigned idx = x * cellSize + y * cellSize * dim;
    unsigned i, j;

    for (i = idx; i < idx + cellSize; i++)
        for (j = i; j < i + cellSize * dim; j+= dim) {
            optr[j].x = color.x;
            optr[j].y = color.y;
            optr[j].z = color.z;
            optr[j].w = 255;
        }
}

__global__ void display_carcin(uchar4 *optr, Carcin *carcin,
                               unsigned gSize, unsigned cellSize,
                               unsigned dim)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    double carcinCon, colorVal;

    if (!(x < gSize && y < gSize)) return;

    carcinCon = carcin->soln[idx];
    if (*carcin->maxVal != 0.0 && *carcin->maxVal < 0.11)
        carcinCon /= *carcin->maxVal;

    colorVal = ceil(min(255.0, 255.0*carcinCon));
    display_cell(optr, x, y, dim3(colorVal, colorVal, colorVal), cellSize, dim);
}

__global__ void display_ca(uchar4 *optr, Cell *grid, unsigned gSize,
                           unsigned cellSize, unsigned dim, dim3 *stateColors)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned idx = x * gSize + y;

	if (!(x < gSize && y < gSize)) return;

    display_cell(optr, x, y, stateColors[grid[idx].state], cellSize, dim);
}

__global__ void display_genes(uchar4 *optr, Cell *G, unsigned gSize,
                              unsigned cellSize, unsigned dim,
                              unsigned nGenes, dim3 *geneColors)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i;
    gene M = (gene) 0;
    unsigned mutInfo;
    bool posMutM;
    double geneExprM, geneExpr;
    dim3 color(255, 255, 255);

    if (!(x < gSize && y < gSize)) return;

    if (G[idx].state == NC || G[idx].state == SC || G[idx].state == EMPTY) {
        display_cell(optr, x, y, color, cellSize, dim);
        return;
    }

    mutInfo = G[idx].positively_mutated(M);
    posMutM = (mutInfo == 1 || mutInfo == 3);
    geneExprM = abs(G[idx].geneExprs[M]);
    for (i = 0; i < nGenes; i++) {
        mutInfo = G[idx].positively_mutated((gene) i);
        if (!(mutInfo == 1 || mutInfo == 3)) continue;

        geneExpr = abs(G[idx].geneExprs[i]);
        if (!posMutM || geneExpr > geneExprM) {
            M = (gene) i;
            posMutM = (mutInfo == 1 || mutInfo == 3);
            geneExprM = geneExpr;
        }
    }

    if (posMutM) {
        color.x = geneColors[M].x;
        color.y = geneColors[M].y;
        color.z = geneColors[M].z;
    }

    display_cell(optr, x, y, color, cellSize, dim);
}

__global__ void display_lineage_heatmap(uchar4 *optr, Cell *G, unsigned *cellLineage,
                                        unsigned *nLineageCells, unsigned *percentageCounts,
                                        unsigned gSize, unsigned cellSize, unsigned dim, dim3 *heatmap)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i;
    int64_t lineage;
    double limit = 0, result = 0;
    dim3 color = heatmap[0];

    if (!(x < gSize && y < gSize)) return;

    lineage = G[idx].lineage;

    if (*nLineageCells != 0 && lineage != -1)
        result = cellLineage[lineage] / (double) *nLineageCells;
    for (i = 0; i < 10; i++) {
        limit += 0.1;
        if (result <= limit) {
            color = heatmap[i];
            atomicAdd(&percentageCounts[i], 1);
            break;
        }
    }

    display_cell(optr, x, y, color, cellSize, dim);
}

__global__ void display_max_lineages(uchar4 *optr, Cell *G,
                                     unsigned *cellLineage, bool *stateInLineage,
                                     unsigned *maxLineages, unsigned stateIdx,
                                     unsigned gSize, unsigned cellSize,
                                     unsigned dim, dim3 *lineageColors)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i;
    int64_t lineage;
    dim3 color(255, 255, 255);

    if (!(x < gSize && y < gSize)) return;

    if (G[idx].state != EMPTY) color = dim3(185, 185, 185);

    lineage = G[idx].lineage;
    if (lineage == -1 || maxLineages[0] == 0) {
        display_cell(optr, x, y, color, cellSize, dim);
        return;
    }

    for (i = 0; i < 20; i++) {
        if (maxLineages[i] == 0) break;
        if (cellLineage[lineage] == maxLineages[i]) {
            if (stateIdx == EMPTY || G[idx].state == stateIdx)
               color = lineageColors[i];
            else if (stateInLineage[lineage*6+G[idx].state])
               color = dim3(128, 128, 128);
            break;
        }
    }

    display_cell(optr, x, y, color, cellSize, dim);
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
    int up = 0, down = 0;

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

    double geneExpr = G[cellIdx].geneExprs[gene];
    if (geneExpr > 0.0)
        up = geneExpr * scl;
    if (geneExpr < 0.0)
        down = abs(geneExpr) * scl;

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

// Used to save the gpu bitmap in a format that can be written as a jpeg
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

// CUDA kernels related to collecting data for graphs, analysis, and debugging
__global__ void collect_data(Cell *G, unsigned *counts, double *phenoSum,
                             double *geneExprSum, double *nPosMutSum, double *fitnessSum,
                             unsigned gSize, unsigned nGenes, unsigned nStates)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i, nPheno, step, mutInfo = 0, numMut = 0;
    double tmpPhenotype[4];

    if (!(x < gSize && y < gSize)) return;

    atomicAdd(&counts[G[idx].state], 1);
    if (G[idx].isTAC) {
        atomicAdd(&counts[nStates], 1);
        atomicAdd(&counts[nStates+(G[idx].state == TC ? 3 : G[idx].state+1)], 1);
    }

    if (G[idx].state == EMPTY) return;

    nPheno = G[idx].params->nPheno;
    memcpy(tmpPhenotype, G[idx].phenotype, nPheno*sizeof(double));
    if (G[idx].isTAC)
        G[idx].adjust_phenotype(PROLIF, -G[idx].params->chanceTACProlif,
                                tmpPhenotype);
    for (i = 0; i < nPheno; i++) {
        atomicAdd(&phenoSum[G[idx].state*nPheno+i], tmpPhenotype[i]);
        atomicAdd(&phenoSum[EMPTY*nPheno+i], tmpPhenotype[i]);
    }

    step = nStates + 4;
    for (i = 0; i < nGenes; i++) {
        mutInfo = G[idx].positively_mutated((gene) i);
        if (mutInfo == 1 || mutInfo == 3) {
            atomicAdd(&counts[i+step], 1);
            atomicAdd(&counts[G[idx].state*nGenes+nGenes+step+i], 1);
            numMut++;
        }
        atomicAdd(&geneExprSum[G[idx].state*nGenes+i], G[idx].geneExprs[i]);
        atomicAdd(&geneExprSum[EMPTY*nGenes+i], G[idx].geneExprs[i]);
    }

    atomicAdd(&nPosMutSum[G[idx].state], numMut);
    atomicAdd(&nPosMutSum[EMPTY], numMut);

    atomicAdd(&fitnessSum[G[idx].state], G[idx].fitness);
    atomicAdd(&fitnessSum[EMPTY], G[idx].fitness);
}

// Saves the state of a cell for debugging purposes. I would also like
// to eventually be able to use the saved cell data to create a resuming
// functionality, so for example if a computer crashes the simulation can be
// rerun starting from where it last stopped.
__global__ void save_cell_data(Cell *prevG, Cell *newG, char *cellData,
                               unsigned gSize, unsigned maxT,
                               double cellLifeSpan, double cellCycleLen,
                               unsigned nGenes, size_t bytesPerCell)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x * gSize + y;
    unsigned i, dataIdx = idx * bytesPerCell;
    unsigned nDigGSize = num_digits(gSize * gSize),
             nDigAge = num_digits(maxT + cellLifeSpan / cellCycleLen),
             nDigInt = 2, nDigFrac = 10, nDigDoub = nDigInt + nDigFrac + 1;
    size_t numChar;

    if (!(x < gSize && y < gSize)) return;

    num_to_string_with_padding(idx, nDigGSize, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string(newG[idx].state, &numChar, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string_with_padding(newG[idx].age, nDigAge, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string_with_padding(newG[idx].phenotype[PROLIF],
                               nDigDoub-1, cellData, &dataIdx, true);
    cellData[dataIdx++] = ',';
    num_to_string_with_padding(newG[idx].phenotype[QUIES],
                               nDigDoub-1, cellData, &dataIdx, true);
    cellData[dataIdx++] = ',';
    num_to_string_with_padding(newG[idx].phenotype[APOP],
                               nDigDoub-1, cellData, &dataIdx, true);
    cellData[dataIdx++] = ',';
    num_to_string_with_padding(newG[idx].phenotype[DIFF],
                               nDigDoub-1, cellData, &dataIdx, true);
    cellData[dataIdx++] = ',';

    cellData[dataIdx++] = '[';
    for (i = 0; i < nGenes; i++) {
        num_to_string_with_padding(newG[idx].geneExprs[i], nDigDoub+1,
                                   cellData, &dataIdx, true);
        if (i != nGenes-1) cellData[dataIdx++] = ',';
    }
    cellData[dataIdx++] = ']';
    cellData[dataIdx++] = ',';

    num_to_string_with_padding(prevG[idx].fitness, 6+nDigFrac+1+1,
                               cellData, &dataIdx, true);
    cellData[dataIdx++] = ',';

    num_to_string_with_padding(prevG[idx].chosenPheno, 2, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string_with_padding(prevG[idx].chosenCell, nDigGSize+1,
                               cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string(*prevG[idx].actionDone, &numChar, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string(newG[idx].excised, &numChar, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string_with_padding(newG[idx].lineage, nDigGSize,
                               cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string(newG[idx].isTAC, &numChar, cellData, &dataIdx);
    cellData[dataIdx++] = ',';

    num_to_string(newG[idx].nTACProlif, &numChar, cellData, &dataIdx);
    cellData[dataIdx] = '\n';
}

#endif // __CUDA_KERNELS_H__
