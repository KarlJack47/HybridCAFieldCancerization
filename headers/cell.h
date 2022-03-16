#ifndef __CELL_H__
#define __CELL_H__

struct CellParams {
    unsigned nPheno, nNeigh, minMut, minNumSC, maxNumEmpty, maxTACDivisions;
    double mutThresh, phenoIncr, exprAdjMaxIncr;
    double chancePhenoAdj, chanceGeneExprAdj, chanceMove, chanceKill,
           chanceUpreg, chanceTACProlif, chanceCSCForm, chanceDediff;
    gene CSCGeneIdx;

    effect *upregPhenoMap, *downregPhenoMap;
    double *phenoInit;
    ca_state *diffMap;
    gene_type *geneType;
    gene_related *geneRelations;

    CellParams(unsigned nStates, unsigned nGenes, unsigned minmut,
               double chancecscform, double alpha, effect *upregphenomap,
               effect *downregphenomap, ca_state *diffmap, gene_type *genetype,
               gene_related *generelations, double cellCycleLen,
               double cellLifeSpan, double *phenoinit=NULL)
    : nPheno(4), nNeigh(8), phenoIncr(1e-6), exprAdjMaxIncr(1.0/sqrt(alpha)),
      mutThresh(0.1), minMut(minmut), chanceCSCForm(chancecscform),
      chanceDediff(1e-4), minNumSC(0), maxNumEmpty(6),
      chancePhenoAdj(0.35), chanceGeneExprAdj(0.45), chanceMove(0.25),
      chanceKill(0.20), chanceUpreg(0.5),
      maxTACDivisions(2), chanceTACProlif(1.0 / 3.0)
    {
        size_t dbl = sizeof(double), eff = sizeof(effect),
               st = sizeof(ca_state), gType = sizeof(gene_type),
               gRel = sizeof(gene_related);
        double SCLifeSpan = 25550.0, apopFactor = 1.625,
               NCProlifFactor = 0.65, SCProlifFactor = 14.75,
               compFactor = 1.485, diffFactor = 1.0 / 6.0;

        CudaSafeCall(cudaMallocManaged((void**)&phenoInit, nStates*nPheno*dbl));
        memset(phenoInit, 0.0, nStates*nPheno*dbl);
        if (phenoinit == NULL) {
            phenoInit[ NC * nPheno +   APOP] = cellCycleLen / cellLifeSpan;
            phenoInit[MNC * nPheno +   APOP] = phenoInit[NC*nPheno+APOP]
                                             / apopFactor;
            phenoInit[ SC * nPheno +   APOP] = cellCycleLen / SCLifeSpan;
            phenoInit[MSC * nPheno +   APOP] = phenoInit[SC*nPheno+APOP]
                                             / apopFactor;
            phenoInit[CSC * nPheno +   APOP] = phenoInit[MSC*nPheno+APOP]
                                             / (5.0 * apopFactor);
            phenoInit[ TC * nPheno +   APOP] = phenoInit[MNC*nPheno+APOP]
                                             / (5.0 * apopFactor);

            phenoInit[ NC * nPheno + PROLIF] = NCProlifFactor * phenoInit[NC*nPheno+APOP];
            phenoInit[MNC * nPheno + PROLIF] = phenoInit[NC*nPheno+PROLIF];
            phenoInit[ TC * nPheno + PROLIF] = phenoInit[NC*nPheno+PROLIF];
            phenoInit[ SC * nPheno + PROLIF] = SCProlifFactor * phenoInit[SC*nPheno+APOP];
            phenoInit[MSC * nPheno + PROLIF] = phenoInit[SC*nPheno+PROLIF];
            phenoInit[CSC * nPheno + PROLIF] = phenoInit[SC*nPheno+PROLIF];

            phenoInit[ SC * nPheno + DIFF] = compFactor * diffFactor;
            phenoInit[MSC * nPheno + DIFF] = phenoInit[SC*nPheno+DIFF];
            phenoInit[CSC * nPheno + DIFF] = phenoInit[SC*nPheno+DIFF];

            phenoInit[ NC * nPheno +  QUIES] = 1.0
                                             - (phenoInit[NC*nPheno+PROLIF]
                                              + phenoInit[NC*nPheno+APOP]);
            phenoInit[MNC * nPheno +  QUIES] = 1.0
                                             - (phenoInit[MNC*nPheno+PROLIF]
                                              + phenoInit[MNC*nPheno+APOP]);
            phenoInit[ SC * nPheno +  QUIES] = 1.0
                                             - (phenoInit[SC*nPheno+PROLIF]
                                              + phenoInit[SC*nPheno+APOP]
                                              + phenoInit[SC*nPheno+DIFF]);
            phenoInit[MSC * nPheno +  QUIES] = 1.0
                                             - (phenoInit[MSC*nPheno+PROLIF]
                                              + phenoInit[MSC*nPheno+APOP]
                                              + phenoInit[MSC*nPheno+DIFF]);
            phenoInit[CSC * nPheno +  QUIES] = 1.0
                                             - (phenoInit[CSC*nPheno+PROLIF]
                                              + phenoInit[CSC*nPheno+APOP]
                                              + phenoInit[CSC*nPheno+DIFF]);
            phenoInit[ TC * nPheno +  QUIES] = 1.0
                                             - (phenoInit[TC*nPheno+PROLIF]
                                              + phenoInit[TC*nPheno+APOP]);
        } else memcpy(phenoInit, phenoinit, nStates*nPheno*dbl);

        CudaSafeCall(cudaMallocManaged((void**)&upregPhenoMap,
                                       nGenes*nPheno*eff));
        memcpy(upregPhenoMap, upregphenomap, nGenes*nPheno*eff);
        CudaSafeCall(cudaMallocManaged((void**)&downregPhenoMap,
                                        nGenes*nPheno*eff));
        memcpy(downregPhenoMap, downregphenomap, nGenes*nPheno*eff);
        CudaSafeCall(cudaMallocManaged((void**)&diffMap,
                                       (nStates-4)*st));
        memcpy(diffMap, diffmap, (nStates-4)*st);
        CudaSafeCall(cudaMallocManaged((void**)&geneType,
                                       nGenes*gType));
        memcpy(geneType, genetype, nGenes*gType);
        CudaSafeCall(cudaMallocManaged((void**)&geneRelations,
                                       nGenes*nGenes*gRel));
        memcpy(geneRelations, generelations, nGenes*nGenes*gRel);
    }

    void free_resources(void)
    {
        if (upregPhenoMap != NULL) {
            CudaSafeCall(cudaFree(upregPhenoMap)); upregPhenoMap = NULL;
        }
        if (downregPhenoMap != NULL) {
            CudaSafeCall(cudaFree(downregPhenoMap)); downregPhenoMap = NULL;
        }
        if (phenoInit != NULL) {
            CudaSafeCall(cudaFree(phenoInit)); phenoInit = NULL;
        }
        if (diffMap != NULL) {
            CudaSafeCall(cudaFree(diffMap)); diffMap = NULL;
        }
        if (geneType != NULL) {
            CudaSafeCall(cudaFree(geneType)); geneType = NULL;
        }
        if (geneRelations != NULL) {
            CudaSafeCall(cudaFree(geneRelations)); geneRelations = NULL;
        }
    }

    void prefetch_memory(int dev, unsigned nStates, unsigned nGenes,
                         cudaStream_t *stream)
    {
        size_t dbl = sizeof(double), eff = sizeof(effect),
               st = sizeof(ca_state), gType = sizeof(gene_type),
               gRel = sizeof(gene_related);

        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(upregPhenoMap,
                                          nGenes*nPheno*eff,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(downregPhenoMap,
                                          nGenes*nPheno*eff,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(phenoInit,
                                          nStates*nPheno*dbl,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(diffMap,
                                          (nStates-4)*st,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(geneType, nGenes*gType,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(geneRelations, nGenes*nGenes*gRel,
                                          dev, *stream));
    }
};

struct Cell {
    int device;
    ca_state state;
    unsigned location, age;
    unsigned *neigh, *testedNeigh, numNeighTested;
    double *phenotype;
    double *geneExprs;
    double *bOut;
    double fitness;

    int chosenPheno, chosenCell;
    int64_t lineage;
    bool cellRebirth, moved, isTAC, deDifferentiated, canKill, canMove;
    unsigned *inUse, *actionDone, *actionApplied, *checked, excised, nTACProlif;

    CellParams *params;

    Cell(void)
    {
        neigh = NULL; testedNeigh = NULL;
        phenotype = NULL;
        geneExprs = NULL; bOut = NULL;
        inUse = NULL; actionDone = NULL; actionApplied = NULL;
        params = NULL;
    }

    Cell(unsigned dev, CellParams *paramsIn,
         unsigned x, unsigned y, unsigned gridSize,
         unsigned nGenes, double cellCycleLen, double cellLifeSpan,
         double *weightStates=NULL, unsigned radius=0,
         unsigned cX=0, unsigned cY=0)
    {
        double rnd, S, weightStatesTemp[7] = { 0.0 };

        device = dev;
        params = paramsIn;
        location = x * gridSize + y;
        fitness = 0.0;
        lineage = -1;
        isTAC = false;

        if (weightStates == NULL) {
            weightStatesTemp[NC] = 0.645; weightStatesTemp[SC] = 0.065;
            weightStatesTemp[EMPTY] = 0.29;
        } else memcpy(weightStatesTemp, weightStates, 7*sizeof(double));
        S = weightStatesTemp[NC];

        rnd = rand() / (double) RAND_MAX;
	if (abs(rnd) <= S)
	    state = NC;
	else if (abs(rnd) <= (S += weightStatesTemp[MNC]))
	    state = MNC;
	else if (abs(rnd) <= (S += weightStatesTemp[SC]))
	    state = SC;
	else if (abs(rnd) <= (S += weightStatesTemp[MSC]))
	    state = MSC;
	else if (abs(rnd) <= (S += weightStatesTemp[CSC]))
	    state = CSC;
	else if (abs(rnd) <= (S += weightStatesTemp[TC]))
	    state = TC;
	else state = EMPTY;

	if (radius != 0 && check_in_circle(x, y, gridSize, radius, cX, cY)) {
	    rnd = rand() / (double) RAND_MAX;
	    S = 0.89; // percentage of TC
	    if (rnd <= S) state = TC;
	    else if (rnd <= (S += 0.01)) state = CSC;
	    else state = EMPTY;
	}

        init(x, y, gridSize, nGenes);

        phenotype[PROLIF] = params->phenoInit[state*params->nPheno+ PROLIF];
        phenotype[ QUIES] = params->phenoInit[state*params->nPheno+  QUIES];
        phenotype[  APOP] = params->phenoInit[state*params->nPheno+   APOP];
        phenotype[  DIFF] = params->phenoInit[state*params->nPheno+   DIFF];

        if (state == EMPTY) age = 0;
        else age = rand() % (int) ceil(cellLifeSpan / cellCycleLen);
    }

    void init(unsigned x, unsigned y, unsigned gridSize, unsigned nGenes)
    {
        size_t unsgn = sizeof(unsigned), dbl = sizeof(double);
        unsigned xIncr = (x + 1) % gridSize, yIncr = (y + 1) % gridSize,
                 xDecr = abs((int) (((int) x - 1) % gridSize)),
                 yDecr = abs((int) (((int) y - 1) % gridSize)), i;

        CudaSafeCall(cudaMallocManaged((void**)&phenotype, params->nPheno*dbl));

        CudaSafeCall(cudaMallocManaged((void**)&neigh, params->nNeigh*unsgn));
        neigh[NORTH]      = x     * gridSize + yIncr;
        neigh[EAST]       = xIncr * gridSize + y;
        neigh[SOUTH]      = x     * gridSize + yDecr;
        neigh[WEST]       = xDecr * gridSize + y;
        neigh[NORTH_EAST] = xIncr * gridSize + yIncr;
        neigh[SOUTH_EAST] = xIncr * gridSize + yDecr;
        neigh[SOUTH_WEST] = xDecr * gridSize + yDecr;
        neigh[NORTH_WEST] = xDecr * gridSize + yIncr;

        numNeighTested = 0;
        CudaSafeCall(cudaMallocManaged((void**)&testedNeigh, params->nNeigh*unsgn));
        for (i = 0; i < params->nNeigh; i++)
            testedNeigh[i] = gridSize * gridSize;

        CudaSafeCall(cudaMallocManaged((void**)&geneExprs,
                                       nGenes*dbl));
        memset(geneExprs, 0, nGenes*dbl);
        CudaSafeCall(cudaMallocManaged((void**)&bOut, nGenes*dbl));
        memset(bOut, 0, nGenes*dbl);

        chosenPheno = -1; chosenCell = -1;
        deDifferentiated = false;
        cellRebirth = false; moved = false; canKill = false; canMove = false;
        CudaSafeCall(cudaMallocManaged((void**)&inUse, unsgn));
        *inUse = 0;
        CudaSafeCall(cudaMallocManaged((void**)&actionApplied, unsgn));
        *actionApplied = 0;
        CudaSafeCall(cudaMallocManaged((void**)&actionDone, unsgn));
        *actionDone = 0;
        CudaSafeCall(cudaMallocManaged((void**)&checked, unsgn));
        *checked = 0;
        excised = 0;
        nTACProlif = 0;
    }

    void free_resources(void)
    {
        if (phenotype != NULL) {
            CudaSafeCall(cudaFree(phenotype)); phenotype = NULL;
        }
        if (neigh != NULL) {
            CudaSafeCall(cudaFree(neigh)); neigh = NULL;
        }
        if (geneExprs != NULL) {
            CudaSafeCall(cudaFree(geneExprs)); geneExprs = NULL;
        }
        if (bOut != NULL) {
            CudaSafeCall(cudaFree(bOut)); bOut = NULL;
        }
        if (inUse != NULL) {
            CudaSafeCall(cudaFree(inUse)); inUse = NULL;
        }
        if (actionApplied != NULL) {
            CudaSafeCall(cudaFree(actionApplied)); actionApplied = NULL;
        }
        if (actionDone != NULL) {
            CudaSafeCall(cudaFree(actionDone)); actionDone = NULL;
        }
        if (checked != NULL) {
            CudaSafeCall(cudaFree(checked)); checked = NULL;
        }
        if (testedNeigh != NULL) {
            CudaSafeCall(cudaFree(testedNeigh)); testedNeigh = NULL;
        }
    }

    void prefetch_memory(int dev, unsigned gSize, unsigned nGenes,
                         cudaStream_t *stream)
    {
        size_t dbl = sizeof(double);

        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(neigh,
                                          params->nNeigh*sizeof(unsigned),
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(testedNeigh,
                                          params->nNeigh*sizeof(unsigned),
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(geneExprs,
                                          nGenes*dbl,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(bOut,
                                          nGenes*dbl,
                                          dev, *stream));
        CudaSafeCall(cudaMemPrefetchAsync(phenotype,
                                          params->nPheno*dbl,
                                          dev, *stream));
    }

    __host__ __device__ void adjust_phenotype(unsigned pheno, double incr,
                                              double *tmpPhenotype=NULL)
    {
        size_t dbl = sizeof(double);
        double incrSign = 1, minIncr = abs(incr), chosenIncr = incr, sum = 0.0;
        double incrTest, tempIncr;
        double phenotypeCpy[4];

        if (incr == 0.0) return;
        if ((state == NC || state == MNC || state == TC) && pheno == DIFF)
            return;

        if (tmpPhenotype != NULL)
            memcpy(phenotypeCpy, tmpPhenotype, params->nPheno*dbl);
        else
            memcpy(phenotypeCpy, phenotype, params->nPheno*dbl);

        if (incr < 0.0) incrSign = -1;

        incrTest = phenotypeCpy[pheno] + chosenIncr;
        if ((isTAC && incrTest < 0.0) || (!isTAC && incrTest <= 0.0)
         && (tempIncr = phenotypeCpy[pheno] * 0.99)  < minIncr) {
            minIncr = tempIncr;
            chosenIncr = incrSign * minIncr;
        } else if (incrTest >= 1.0
               && (tempIncr = (1.0 - phenotypeCpy[pheno]) * 0.99) < minIncr) {
            minIncr = tempIncr;
            chosenIncr = incrSign * minIncr;
        }
        incrTest = phenotypeCpy[QUIES] - chosenIncr;
        if (pheno != QUIES && incrTest <= 0.0
         && (tempIncr = phenotypeCpy[QUIES] * 0.99) < minIncr) {
            minIncr = tempIncr;
            chosenIncr = incrSign * minIncr;
        } else if (pheno != QUIES && incrTest >= 1.0
               && (tempIncr = (1.0 - phenotypeCpy[QUIES]) * 0.99) < minIncr) {
            minIncr = tempIncr;
            chosenIncr = incrSign * minIncr;
        }

        if (pheno != QUIES) {
            phenotypeCpy[pheno] += chosenIncr;
            phenotypeCpy[QUIES] -= chosenIncr;
        } else {
            phenotypeCpy[pheno] += chosenIncr;
            sum = phenotypeCpy[PROLIF] + phenotypeCpy[APOP] + phenotypeCpy[DIFF];
            phenotypeCpy[PROLIF] -= chosenIncr * (phenotypeCpy[PROLIF] / sum);
            phenotypeCpy[  APOP] -= chosenIncr * (phenotypeCpy[APOP] / sum);
            phenotypeCpy[  DIFF] -= chosenIncr * (phenotypeCpy[DIFF] / sum);
        }

        if (tmpPhenotype != NULL)
            memcpy(tmpPhenotype, phenotypeCpy, params->nPheno*dbl);
        else
            memcpy(phenotype, phenotypeCpy, params->nPheno*dbl);
    }

    __device__ void normalize(double sumIn=0.0)
    {
        unsigned i;
        double S = sumIn == 0.0 ? 0.0 : sumIn;

        if (S == 0.0)
            for (i = 0; i < params->nPheno; i++) S += phenotype[i];

        for (i = 0; i < params->nPheno; i++)
            phenotype[i] /= S;
    }

    __device__ void change_state(ca_state newState)
    {
        unsigned i;
        unsigned newStatePheno = newState, statePheno = state;
        double delta, deltaSign = 1, S = 0.0, incrTest;
        bool nonSC = (newState == NC || newState == MNC || newState == TC);

        if (newState == ERROR) return;
        if (state == newState) return;

        for (i = 0; i < params->nPheno; i++) {
            if (nonSC && i == DIFF)
                continue;
            delta = phenotype[i] - params->phenoInit[statePheno*params->nPheno+i];
            if (delta < 0.0) deltaSign = -1;
            incrTest = params->phenoInit[newStatePheno*params->nPheno+i] + delta;
            if (incrTest <= 0.0)
                delta = deltaSign * params->phenoInit[newStatePheno*params->nPheno+i] * 0.99;
            else if (incrTest >= 1.0)
                delta = deltaSign * (1.0 - params->phenoInit[newStatePheno*params->nPheno+i]) * 0.99;
            phenotype[i] = params->phenoInit[newStatePheno*params->nPheno+i] + delta;
            S += phenotype[i];
        }

        state = newState;

        if (state == EMPTY) return;

        if (abs(S - 1.0) > FLT_EPSILON)
            normalize(S);
    }

    __device__ int get_phenotype(curandState_t *rndState)
    {
        double cmp = phenotype[PROLIF];
        double rnd = curand_uniform_double(rndState);

        if (state == EMPTY) return -1;

        if      (rnd <=  cmp)                      return PROLIF;
        else if (rnd <= (cmp += phenotype[QUIES])) return  QUIES;
        else if (rnd <= (cmp += phenotype[ APOP])) return   APOP;
        else if (rnd <= (cmp += phenotype[ DIFF])) return   DIFF;

        return -1;
    }

    __device__ void copy_mutations(Cell *c, unsigned nGenes)
    {
        unsigned i;
        unsigned statePheno = state;
        bool nonSC = (c->state == NC || c->state == MNC || c->state == TC);
        double delta, deltaSign = 1, S = 0.0, incrTest;

        if (c->state == EMPTY || state == EMPTY) return;

        for (i = 0; i < params->nPheno; i++) {
            if (nonSC && i == DIFF)
                continue;
            delta = phenotype[i] - params->phenoInit[statePheno*params->nPheno+i];
            if (delta == 0.0) { S += c->phenotype[i]; continue; }
            if (delta < 0.0) deltaSign = -1;
            incrTest = c->phenotype[i] + delta;
            if (incrTest <= 0.0)
                delta = deltaSign * c->phenotype[i] * 0.99;
            else if (incrTest >= 1.0)
                delta = deltaSign * (1.0 - c->phenotype[i]) * 0.99;
            c->phenotype[i] += delta;
            S += c->phenotype[i];
        }
        if (abs(S - 1.0) > FLT_EPSILON)
            c->normalize(S);

        for (i = 0; i < nGenes; i++) {
            c->geneExprs[i] = geneExprs[i];
            c->bOut[i] = bOut[i];
        }
    }

    __device__ unsigned positively_mutated(gene M)
    {
        double geneExpr = geneExprs[M];

        if (geneExpr < 0.0 && abs(geneExpr) >= params->mutThresh) {
            if (params->geneType[M] == SUPPR)
                return 1;
            return 2; // downregulated
        }

        if (geneExpr > 0.0 && geneExpr >= params->mutThresh) {
            if (params->geneType[M] == ONCO)
                return 3;
            return 4; // upregulated
        }

        return 0; // not mutated
    }

    __device__ int proliferate(Cell *c, curandState_t *rndState,
                               unsigned gSize, unsigned nGenes,
                               unsigned *countKills)
    {
        bool CSCorTC = (state == CSC || state == TC),
             lowerFitness = (fitness <= c->fitness);

        if (c->state != EMPTY && lowerFitness && !canKill)
            return -2;

        if (c->state != EMPTY && (!lowerFitness || canKill)) {
            if (lowerFitness && canKill) {// by chance (CSC or TC)
                // Proliferation by state via chance of c->state
                atomicAdd(&countKills[(state-4)*7+c->state+42], 1);
                // Proliferation by state via chance overall
                atomicAdd(&countKills[(state-4)*7+EMPTY+42], 1);
                // Kill by state via chance overall of c->state
                atomicAdd(&countKills[(state-4)*7+c->state+140], 1);
                // Kill by state via chance overall
                atomicAdd(&countKills[(state-4)*7+EMPTY+140], 1);
            } else {
                // Proliferation by state via competition of c->state
                atomicAdd(&countKills[state*7+c->state], 1);
                // Proliferation by state via competition overall
                atomicAdd(&countKills[state*7+EMPTY], 1);
            }
            if (CSCorTC) {
                // Proliferation by state overall of c->state (competition + chance)
                atomicAdd(&countKills[(state-4)*7+c->state+56], 1);
                // Proliferation by state overall (competition + chance)
                atomicAdd(&countKills[(state-4)*7+EMPTY+56], 1);
            }
            if (state != NC && state != MNC) {
                // Num kill c->state by state overall
                atomicAdd(&countKills[(state-2)*7+c->state+154], 1);
                // Num kill by state overall
                atomicAdd(&countKills[(state-2)*7+EMPTY+154], 1);
            }
            c->apoptosis(nGenes);
        }

        c->change_state(state);

        if (isTAC) {
            if (nTACProlif != params->maxTACDivisions) {
                nTACProlif++;
                c->isTAC = true;
                c->nTACProlif = nTACProlif;
            } else {
                adjust_phenotype(0, -params->chanceTACProlif);
                isTAC = false;
                nTACProlif = 0;
            }
        }
        copy_mutations(c, nGenes);

        if (lineage == -1) {
            c->lineage = location;
            lineage = c->lineage;
        } else c->lineage = lineage;
        c->age = 0; age = 0;

        return state;
    }

    __device__ int differentiate(Cell *c, curandState_t *rndState,
                                 unsigned gSize, unsigned nGenes,
                                 unsigned *countKills)
    {
        ca_state newState = ERROR;
        bool lowerFitness = (fitness <= c->fitness);

        if (state != SC && state != MSC && state != CSC)
            return -1;

        if (c->state != EMPTY && lowerFitness && !canKill)
            return -2;

        if (c->state != EMPTY && (!lowerFitness || canKill)) {
            if (lowerFitness && canKill) {// by chance CSC
                // Differentiation by state via chance of c->state
                atomicAdd(&countKills[c->state+91], 1);
                // Differentiation by state via chance overall
                atomicAdd(&countKills[EMPTY+91], 1);
                // Kill by state via chance overall of c->state
                atomicAdd(&countKills[(state-4)*7+c->state+140], 1);
                // Kill by state via chance overall
                atomicAdd(&countKills[(state-4)*7+EMPTY+140], 1);
            } else {
                // Differentiation by state via competition of c->state
                atomicAdd(&countKills[(state-2)*7+c->state+70], 1);
                // Differentiation by state via competition overall
                atomicAdd(&countKills[(state-2)*7+EMPTY+70], 1);
                // Kill by state via competition overall of c->state
                atomicAdd(&countKills[(state-2)*7+c->state+119], 1);
                // Kill by state via competition overall
                atomicAdd(&countKills[(state-2)*7+EMPTY+119], 1);
            }
            if (state == CSC) {
                // Differentiation by state overall of c->state (competition + chance)
                atomicAdd(&countKills[c->state+98], 1);
                // Differentiation by state overall (competition + chance)
                atomicAdd(&countKills[EMPTY+98], 1);
            }
            // Num kill c->state by state overall
            atomicAdd(&countKills[(state-2)*7+c->state+154], 1);
            // Num kill by state overall
            atomicAdd(&countKills[(state-2)*7+EMPTY+154], 1);
            c->apoptosis(nGenes);
        }

        newState = params->diffMap[state-2];
        c->change_state(newState);
        copy_mutations(c, nGenes);

        if (lineage == -1) {
            c->lineage = location;
            lineage = c->lineage;
        } else c->lineage = lineage;
        c->age = 0; age = 0;
        c->isTAC = true;
        c->adjust_phenotype(0, params->chanceTACProlif);

        return newState;
    }

    __device__ int move(Cell *c, curandState_t *rndState,
                        unsigned gSize, unsigned nGenes,
                        unsigned *countKills)
    {
        bool CSCorTC = (state == CSC || state == TC);

        if (!CSCorTC && c->state != EMPTY) return -2;

        if (!canMove)
            return -2;

        if (c->state != EMPTY && !canKill) return -2;

        // Only CSC and TC can do this
        if (c->state != EMPTY && canKill) {
            // Movement by state via chance of c->state
            atomicAdd(&countKills[(state-4)*7+c->state+105], 1);
            // Movement by state via chance overall
            atomicAdd(&countKills[(state-4)*7+EMPTY+105], 1);
            // Kill by state via chance overall of c->state
            atomicAdd(&countKills[(state-4)*7+c->state+140], 1);
            // Kill by state via chance overall
            atomicAdd(&countKills[(state-4)*7+EMPTY+140], 1);
            // Num kill c->state by state overall
            atomicAdd(&countKills[(state-2)*7+c->state+154], 1);
            // Num kill by state overall
            atomicAdd(&countKills[(state-2)*7+EMPTY+154], 1);
            c->apoptosis(nGenes);
        }

        c->change_state(state);
        copy_mutations(c, nGenes);
        c->age = age;
        c->lineage = lineage;
        if (isTAC) {
            c->isTAC = isTAC;
            c->nTACProlif = nTACProlif;
        }
        apoptosis(nGenes);

        return c->state;
    }

    __device__ void apoptosis(unsigned nGenes)
    {
        unsigned i;

        if (state == EMPTY) return;

        state = EMPTY;
        for (i = 0; i < params->nPheno; i++)
             phenotype[i] = params->phenoInit[EMPTY*params->nPheno+i];
        for (i = 0; i < nGenes; i++) {
            geneExprs[i] = 0.0;
            bOut[i] = 0.0;
        }
        age = 0;
        lineage = -1;
        if (isTAC) {
            isTAC = false;
            nTACProlif = 0;
        }
        fitness = 0.0;
    }

    __device__ void phenotype_mutate(gene M, curandState_t *rndState)
    {
        unsigned i;
        double incr = params->phenoIncr;
        unsigned mutInfo = positively_mutated(M);
        curandState_t localState = *rndState;

        // downregulation
        if (mutInfo == 1 || mutInfo == 2) {
            for (i = 0; i < params->nPheno; i++) {
                if (curand_uniform_double(&localState) <= params->chancePhenoAdj) {
                    incr *= curand_uniform_double(&localState);
                    incr *= params->downregPhenoMap[M*params->nPheno+i];
                    adjust_phenotype(i, incr);
                }
            }
        // upregulation
        } else if (mutInfo == 3 || mutInfo == 4) {
            for (i = 0; i < params->nPheno; i++) {
                if (curand_uniform_double(&localState) <= params->chancePhenoAdj) {
                    incr *= curand_uniform_double(&localState);
                    incr *= params->upregPhenoMap[M*params->nPheno+i];
                    adjust_phenotype(i, incr);
                }
            }
        }

        *rndState = localState;
    }

    __device__ void gene_regulation_adj(gene M, curandState_t *rndState,
                                        unsigned nGenes)
	{
	    unsigned m;
        double incr = params->exprAdjMaxIncr, factor,
               chanceFix = params->chanceGeneExprAdj * 0.80,
               rndSamp;
        unsigned mutInfoM = positively_mutated(M), mutInfom;
        bool posMutM;
        curandState_t localState = *rndState;

        for (m = 0; m < nGenes; m++) {
            if (m != M && params->geneRelations[M*nGenes+m] != YES) continue;
            mutInfom = m == M ? mutInfoM : positively_mutated((gene) m);
            // M positively mutated so mutate m towards cancer
            if (posMutM = (mutInfoM == 1 || mutInfoM == 3)
             && (rndSamp = curand_uniform_double(&localState)) <= params->chanceGeneExprAdj) {
                factor = curand_uniform_double(&localState);
                if (params->geneType[m] == SUPPR)
                    geneExprs[m] -= incr * factor;
                else geneExprs[m] += incr * factor;
            }
            // M is not positively mutated.
            // m is mutated so mutate m away from cancer
            if (!posMutM && mutInfom
             && rndSamp <= params->chanceGeneExprAdj) {
                factor = curand_uniform_double(&localState);
                if (mutInfom == 1 || mutInfom == 2)
                    geneExprs[m] += incr * factor;
                else geneExprs[m] -= incr * factor; // mutInfom = 3 or 4
            }

            // body fixing mutations
            if (m == M && mutInfom
             && curand_uniform_double(&localState) <= chanceFix) {
                factor = curand_uniform_double(&localState);
                if (mutInfom == 1 || mutInfom == 2) geneExprs[m] += incr * factor;
                else geneExprs[m] -= incr * factor;
            }
        }

        *rndState = localState;
    }

    __device__ void compute_fitness(unsigned nGenes)
    {
        unsigned m;
        double geneExprsSum = 0.0, SCFactor = 4.0;
        unsigned statePheno = state;

        fitness = 0.0;

        fitness -= age * phenotype[APOP];
        fitness += isTAC ? phenotype[PROLIF] / (params->phenoInit[statePheno*params->nPheno]
                                                + params->chanceTACProlif)
                         : phenotype[PROLIF] / params->phenoInit[statePheno*params->nPheno];
        fitness -= phenotype[APOP] / params->phenoInit[statePheno*params->nPheno+APOP];
        for (m = 0; m < nGenes; m++) {
            if (geneExprs[m] == 0.0) continue;
            if (geneExprs[m] < 0.0) {
                if (params->geneType[m] == SUPPR)
                    geneExprsSum += geneExprs[m] / params->mutThresh;
                else
                    geneExprsSum -= geneExprs[m] / params->mutThresh;
            } else {
                if (params->geneType[m] == ONCO)
                    geneExprsSum += geneExprs[m] / params->mutThresh;
                else
                    geneExprsSum -= geneExprs[m] / params->mutThresh;
            }
        }
        fitness += (state == NC || state == SC) ? -geneExprsSum : geneExprsSum;

        if (state == SC || state == MSC || state == CSC)
            fitness *= fitness < 0 ? (1.0 / SCFactor) : SCFactor;
        if (isTAC) fitness *= fitness < 0 ? (2.0 / SCFactor) : SCFactor / 2.0;
    }

    __device__ void mutate(GeneExprNN *NN, double *NNOut,
                           curandState_t *rndState)
    {
        unsigned m;
        double factor;
        curandState_t localState = *rndState;

        if (state == EMPTY) return;

        for (m = 0; m < NN->nOut; m++) {
            factor = curand_uniform_double(&localState);
            geneExprs[m] += factor * NNOut[m];
        }

        NN->mutate(bOut, geneExprs, params->mutThresh);
        for (m = 0; m < NN->nOut; m++)
            gene_regulation_adj((gene) m, rndState, NN->nOut);
        for (m = 0; m < NN->nOut; m++)
            phenotype_mutate((gene) m, rndState);

        compute_fitness(NN->nOut);
    }
};

#endif // __CELL_H__
