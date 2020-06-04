#ifndef __CELL_H__
#define __CELL_H__

struct CellParams {
    unsigned nPheno, nNeigh;
    double phenoIncr;
    double exprAdjMaxIncr;
    double mutThresh;
    gene CSCGeneIdx;
    double chanceMove;
    double chanceKill;
    double chanceUpreg;
    double chancePhenoMut;
    double chanceExprAdj;

    effect *upregPhenoMap, *downregPhenoMap;
    double *phenoInit;
    ca_state *stateMutMap, *prolifMutMap, *diffMutMap;
    gene_type *geneType;
    gene_related *geneRelations;

    CellParams(unsigned nStates, unsigned nGenes, double alpha,
               effect *upregphenomap, effect *downregphenomap,
               ca_state *statemutmap, ca_state *prolifmutmap,
               ca_state *diffmutmap, gene_type *genetype,
               gene_related *generelations, double *phenoinit=NULL)
    : nPheno(4), nNeigh(8), phenoIncr(1e-6), exprAdjMaxIncr(1.0/sqrt(alpha)),
      mutThresh(0.1), CSCGeneIdx(RB), chanceMove(0.25), chanceKill(0.4),
      chanceUpreg(0.5), chancePhenoMut(1.0), chanceExprAdj(1.0)
    {
        size_t dbl = sizeof(double), eff = sizeof(effect),
               st = sizeof(ca_state), gType = sizeof(gene_type),
               gRel = sizeof(gene_related);

        CudaSafeCall(cudaMallocManaged((void**)&phenoInit, nStates*nPheno*dbl));
        if (phenoinit == NULL) {
            phenoInit[ NC * nPheno + PROLIF] = 0.024000;
            phenoInit[MNC * nPheno + PROLIF] = 0.024000;
            phenoInit[ SC * nPheno + PROLIF] = 0.006399;
            phenoInit[MSC * nPheno + PROLIF] = 0.006399;
            phenoInit[CSC * nPheno + PROLIF] = 0.000625;
            phenoInit[ TC * nPheno + PROLIF] = 0.000625;
            phenoInit[ NC * nPheno +  QUIES] = 0.966000;
            phenoInit[MNC * nPheno +  QUIES] = 0.971000;
            phenoInit[ SC * nPheno +  QUIES] = 0.966101;
            phenoInit[MSC * nPheno +  QUIES] = 0.967351;
            phenoInit[CSC * nPheno +  QUIES] = 0.991750;
            phenoInit[ TC * nPheno +  QUIES] = 0.993125;
            phenoInit[ NC * nPheno +   APOP] = 0.010000;
            phenoInit[MNC * nPheno +   APOP] = 0.005000;
            phenoInit[ SC * nPheno +   APOP] = 0.002500;
            phenoInit[MSC * nPheno +   APOP] = 0.001250;
            phenoInit[CSC * nPheno +   APOP] = 0.000625;
            phenoInit[ TC * nPheno +   APOP] = 0.000625;
            phenoInit[ NC * nPheno +   DIFF] = 0.000000;
            phenoInit[MNC * nPheno +   DIFF] = 0.000000;
            phenoInit[ SC * nPheno +   DIFF] = 0.025000;
            phenoInit[MSC * nPheno +   DIFF] = 0.025000;
            phenoInit[CSC * nPheno +   DIFF] = 0.006250;
            phenoInit[ TC * nPheno +   DIFF] = 0.000000;
            memset(phenoInit+EMPTY*nPheno, 0, nPheno*dbl);
        } else memcpy(phenoInit, phenoinit, nStates*nPheno*dbl);

        CudaSafeCall(cudaMallocManaged((void**)&upregPhenoMap,
                                       nGenes*nPheno*eff));
        memcpy(upregPhenoMap, upregphenomap, nGenes*nPheno*eff);
        CudaSafeCall(cudaMallocManaged((void**)&downregPhenoMap,
                                        nGenes*nPheno*eff));
        memcpy(downregPhenoMap, downregphenomap, nGenes*nPheno*eff);
        CudaSafeCall(cudaMallocManaged((void**)&stateMutMap,
                                       (nStates-1)*nGenes*st));
        memcpy(stateMutMap, statemutmap, (nStates-1)*nGenes*st);
        CudaSafeCall(cudaMallocManaged((void**)&prolifMutMap,
                                       (nStates-1)*nGenes*st));
        memcpy(prolifMutMap, prolifmutmap, (nStates-1)*nGenes*st);
        CudaSafeCall(cudaMallocManaged((void**)&diffMutMap,
                                       (nStates-1)*(nGenes+1)*st));
        memcpy(diffMutMap, diffmutmap, (nStates-1)*(nGenes+1)*st);
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
        if (stateMutMap != NULL) {
            CudaSafeCall(cudaFree(stateMutMap)); stateMutMap = NULL;
        }
        if (prolifMutMap != NULL) {
            CudaSafeCall(cudaFree(prolifMutMap)); prolifMutMap = NULL;
        }
        if (diffMutMap != NULL) {
            CudaSafeCall(cudaFree(diffMutMap)); diffMutMap = NULL;
        }
        if (geneType != NULL) {
            CudaSafeCall(cudaFree(geneType)); geneType = NULL;
        }
        if (geneRelations != NULL) {
            CudaSafeCall(cudaFree(geneRelations)); geneRelations = NULL;
        }
    }

    void prefetch_memory(int dev, unsigned nStates, unsigned nGenes)
    {
        size_t dbl = sizeof(double), eff = sizeof(effect),
               st = sizeof(ca_state), gType = sizeof(gene_type),
               gRel = sizeof(gene_related);
        if (dev == -1) dev = cudaCpuDeviceId;

        CudaSafeCall(cudaMemPrefetchAsync(upregPhenoMap,
                                          nGenes*nPheno*eff,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(downregPhenoMap,
                                          nGenes*nPheno*eff,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(phenoInit,
                                          nStates*nPheno*dbl,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(stateMutMap,
                                          (nStates-1)*nGenes*st,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(prolifMutMap,
                                          (nStates-1)*nGenes*st,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(diffMutMap,
                                          (nStates-1)*(nGenes+1)*st,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(geneType, nGenes*gType,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(geneRelations, nGenes*nGenes*gRel,
                                          dev, NULL));
    }
};

struct Cell {
    int device;
    ca_state state;
    unsigned age;
    unsigned location;
    unsigned *neigh;
    double *phenotype;
    double *geneExprs;
    double *bOut;

    int chosenPheno, chosenCell;
    bool cellRebirth, moved;
    unsigned *inUse, *actionDone, *actionApplied;

    CellParams *params;

    Cell(void)
    {
        neigh = NULL;
        phenotype = NULL;
        geneExprs = NULL;
        bOut = NULL;
        params = NULL;
    }

    Cell(unsigned dev, CellParams *paramsIn,
         unsigned x, unsigned y, unsigned gridSize,
         unsigned nGenes, double cellCycleLen, double cellLifeSpan,
         double *weightStates=NULL)
    {
        double rnd;
        double S;
        bool statesIn = true;

        set_seed();

        device = dev;
        params = paramsIn;
        location = x * gridSize + y;

        if (weightStates == NULL) {
            weightStates = (double*)malloc(7*sizeof(double));
            weightStates[NC] = 0.70; weightStates[SC] = 0.01;
            weightStates[EMPTY] = 0.29;
            statesIn = false;
        }
        S = weightStates[NC];

        rnd = rand() / (double) RAND_MAX;
	    if (abs(rnd) <= S)
	        state = NC;
	    else if (abs(rnd) <= (S += weightStates[MNC]))
	        state = MNC;
	    else if (abs(rnd) <= (S += weightStates[SC]))
	        state = SC;
	    else if (abs(rnd) <= (S += weightStates[MSC]))
	        state = MSC;
	    else if (abs(rnd) <= (S += weightStates[CSC]))
	        state = CSC;
	    else if (abs(rnd) <= (S += weightStates[TC]))
	        state = TC;
	    else
	        state = EMPTY;

	    if (!statesIn) free(weightStates);

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
                 yDecr = abs((int) (((int) y - 1) % gridSize));

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

        CudaSafeCall(cudaMallocManaged((void**)&geneExprs,
                                       2*nGenes*dbl));
        memset(geneExprs, 0, 2*nGenes*dbl);
        CudaSafeCall(cudaMallocManaged((void**)&bOut, nGenes*dbl));
        memset(bOut, 0, nGenes*dbl);

        chosenPheno = -1; chosenCell = -1;
        cellRebirth = false; moved = false;
        CudaSafeCall(cudaMallocManaged((void**)&inUse, unsgn));
        *inUse = 0;
        CudaSafeCall(cudaMallocManaged((void**)&actionApplied, unsgn));
        *actionApplied = 0;
        CudaSafeCall(cudaMallocManaged((void**)&actionDone, unsgn));
        *actionDone = 0;
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
    }

    void prefetch_memory(int dev, unsigned gSize, unsigned nGenes)
    {
        size_t dbl = sizeof(double);

        if (dev == -1) dev = cudaCpuDeviceId;
        CudaSafeCall(cudaMemPrefetchAsync(neigh,
                                          params->nNeigh*sizeof(unsigned),
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(geneExprs,
                                          2*nGenes*dbl,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(bOut,
                                          nGenes*dbl,
                                          dev, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(phenotype,
                                          params->nPheno*dbl,
                                          dev, NULL));
    }

    __device__ void adjust_phenotype(unsigned pheno, double incr)
    {
        double incrSign;
        double minIncr = abs(incr), chosenIncr = incr;
        double sum = 0.0;

        if (incr == 0.0) return;
        if ((state == NC || state == MNC || state == TC) && pheno == DIFF)
            return;

        incrSign = incr / abs(incr);

        if (phenotype[pheno] + chosenIncr <= 0.0
         && phenotype[pheno] * 0.99  < minIncr) {
            minIncr = phenotype[pheno] * 0.99;
            chosenIncr = incrSign * minIncr;
        } else if (phenotype[pheno] + chosenIncr >= 1.0
               && (1.0 - phenotype[pheno]) * 0.99 < minIncr) {
            minIncr = (1.0 - phenotype[pheno]) * 0.99;
            chosenIncr = incrSign * minIncr;
        }
        if (pheno != QUIES && phenotype[QUIES] - chosenIncr <= 0.0
         && phenotype[QUIES] * 0.99 < minIncr) {
            minIncr = phenotype[QUIES] * 0.99;
            chosenIncr = incrSign * minIncr;
        } else if (pheno != QUIES && phenotype[QUIES] - chosenIncr >= 1.0
               && (1.0 - phenotype[QUIES]) * 0.99 < minIncr) {
            minIncr = (1.0 - phenotype[QUIES]) * 0.99;
            chosenIncr = incrSign * minIncr;
        }

        if (pheno != QUIES) {
            phenotype[pheno] += chosenIncr;
            phenotype[QUIES] -= chosenIncr;
        } else {
            phenotype[pheno] += chosenIncr;
            sum += phenotype[PROLIF] + phenotype[APOP] + phenotype[DIFF];
            phenotype[PROLIF] -= chosenIncr * (phenotype[PROLIF] / sum);
            phenotype[  APOP] -= chosenIncr * (phenotype[APOP] / sum);
            phenotype[  DIFF] -= chosenIncr * (phenotype[DIFF] / sum);
        }
    }

    __device__ void normalize()
    {
        unsigned i;
        double S = 0.0;

        for (i = 0; i < params->nPheno; i++) S += phenotype[i];

        for (i = 0; i < params->nPheno; i++)
            phenotype[i] /= S;
    }

    __device__ void change_state(ca_state newState)
    {
        unsigned i;
        double delta, deltaSign, S;

        if (newState == ERR) return;
        if (state == newState) return;

        for (i = 0; i < params->nPheno; i++) {
            if ((newState == NC || newState == MNC || newState == TC) && i == DIFF)
                continue;
            delta = phenotype[i] - params->phenoInit[state*params->nPheno+i];
            deltaSign = delta / abs(delta);
            if (params->phenoInit[newState*params->nPheno+i] + delta <= 0.0)
                delta = deltaSign * params->phenoInit[newState*params->nPheno+i] * 0.99;
            else if (params->phenoInit[newState*params->nPheno+i] + delta >= 1.0)
                delta = deltaSign * (1.0 - params->phenoInit[newState*params->nPheno+i]) * 0.99;
            phenotype[i] = params->phenoInit[newState*params->nPheno+i] + delta;
        }

        state = newState;

        if (state == EMPTY) return;

        S = 0.0;
        for (i = 0; i < params->nPheno; i++) S += phenotype[i];
        if (abs(S - 1.0) > FLT_EPSILON)
            normalize();
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
        double delta, deltaSign, S = 0.0;

        if (c->state == EMPTY || state == EMPTY) return;

        for (i = 0; i < params->nPheno; i++) {
            if ((c->state == NC || c->state == MNC || c->state == TC) && i == DIFF)
                continue;
            delta = phenotype[i] - params->phenoInit[state*params->nPheno+i];
            if (delta == 0.0) continue;
            deltaSign = delta / abs(delta);
            if (c->phenotype[i] + delta <= 0.0)
                delta = deltaSign * c->phenotype[i] * 0.99;
            else if (c->phenotype[i] + delta >= 1.0)
                delta = deltaSign * (1.0 - c->phenotype[i]) * 0.99;
            c->phenotype[i] += delta;
        }
        for (i = 0; i < params->nPheno; i++) S += c->phenotype[i];
        if (abs(S - 1.0) > FLT_EPSILON) c->normalize();
        for (i = 0; i < nGenes; i++) {
            c->geneExprs[i*2] = geneExprs[i*2];
            c->geneExprs[i*2+1] = geneExprs[i*2+1];
            c->bOut[i] = bOut[i];
        }
    }

    __device__ bool positively_mutated(gene M)
    {
        double geneExpr = geneExprs[M*2] - geneExprs[M*2+1];

        if ((params->geneType[M] == SUPPR && geneExpr < 0.0
          && geneExprs[M*2+1] >= params->mutThresh)
         || (params->geneType[M] == ONCO  && geneExpr > 0.0
          && geneExprs[M*2] >= params->mutThresh))
            return true;

        return false;
    }

    __device__ int proliferate(Cell *c, curandState_t *rndState,
                               unsigned nGenes)
    {
        ca_state newState = ERR;
        gene m;
        curandState_t localState;

        if (state != CSC && state != TC && c->state != EMPTY)
            return -2;
        if ((state == CSC || state == TC)
         && (c->state == TC || c->state == CSC))
            return -2;

        if (state == CSC && !positively_mutated(params->CSCGeneIdx))
            return -1;

        localState = *rndState;

        m = (gene) ((unsigned) ceil(curand_uniform_double(&localState)
          * (double) nGenes) % nGenes);

        if ((state == CSC && c->state != EMPTY
          && curand_uniform_double(&localState) <= params->chanceKill)
         || (state == TC && c->state != EMPTY
          && curand_uniform_double(&localState) <= params->chanceKill)
          || c->state == EMPTY) {
            if (!positively_mutated(m)) newState = state;
            else {
                if (curand_uniform_double(&localState) <= 0.5) {
                    change_state(params->stateMutMap[state*nGenes+m]);
                    newState = params->prolifMutMap[state*nGenes+m];
                } else {
                    newState = params->prolifMutMap[state*nGenes+m];
                    change_state(params->stateMutMap[state*nGenes+m]);
                }
            }
            if (c->state != EMPTY && (state == CSC || state == TC)) {
                printf("(%d, %d) killed by (%d, %d) via proliferation\n",
                       c->location, c->state, location, state);
                c->apoptosis(nGenes);
            }
            c->change_state(newState);
            copy_mutations(c, nGenes);

            c->age = 0; age = 0;
        }

        *rndState = localState;

        return newState;
    }

    __device__ int differentiate(Cell *c, curandState_t *rndState,
                                 unsigned nGenes)
    {
        ca_state newState = ERR;
        gene m;
        curandState_t localState;

        if (state != CSC && c->state != EMPTY)
            return -2;
        if (state == CSC && (c->state == CSC || c->state == TC))
            return -2;
        if (state == CSC && !positively_mutated(params->CSCGeneIdx))
            return -1;

        localState = *rndState;

        m = (gene) ((unsigned) ceil(curand_uniform_double(&localState)
          * (double) nGenes) % nGenes);

        if ((state == CSC && c->state != EMPTY
           && curand_uniform_double(&localState) <= params->chanceKill)
         || c->state == EMPTY) {
            if (!positively_mutated(m))
                newState = params->diffMutMap[state*(nGenes+1)];
            else {
                if (curand_uniform_double(&localState) <= 0.5) {
                    change_state(params->stateMutMap[state*nGenes+m]);
                    newState = params->diffMutMap[state*(nGenes+1)+(m+1)];
                } else {
                    newState = params->diffMutMap[state*(nGenes+1)+(m+1)];
                    change_state(params->stateMutMap[state*nGenes+m]);
                }
            }
            if (newState == -1) { *rndState = localState; return newState; }
            if (c->state != EMPTY && state == CSC) {
                printf("(%d, %d) killed by (%d, %d) via differentiation\n",
                       c->location, c->state, location, state);
                c->apoptosis(nGenes);
            }
            c->change_state(newState);
            copy_mutations(c, nGenes);

            c->age = 0; age = 0;
        }

        *rndState = localState;

        return newState;
    }

    __device__ int move(Cell *c, curandState_t *rndState, unsigned nGenes)
    {
        curandState_t localState;

        if ((state != CSC || state != TC) && c->state != EMPTY)
            return -2;
        if ((state == CSC || state == TC)
         && (c->state == CSC || c->state == TC))
            return -2;

        localState = *rndState;

        if (curand_uniform_double(&localState) <= params->chanceMove) {
            if (((state == CSC || state == TC) && c->state != EMPTY
               && curand_uniform_double(&localState) <= params->chanceKill)
			 || c->state == EMPTY) {
			    if (c->state != EMPTY && (state == CSC || state == TC)) {
			        printf("(%d, %d) killed by (%d, %d) via movement\n",
			               c->location, c->state, location, state);
			        c->apoptosis(nGenes);
			    }
                c->change_state(state);
                copy_mutations(c, nGenes);
                c->age = age;
                apoptosis(nGenes);
            }
        }

        *rndState = localState;

        return 0;
    }

    __device__ void apoptosis(unsigned nGenes)
    {
        unsigned i;

        state = EMPTY;
        for (i = 0; i < params->nPheno; i++)
             phenotype[i] = params->phenoInit[EMPTY*params->nPheno+i];
        for (i = 0; i < nGenes; i++) {
            geneExprs[i*2] = 0.0;
			geneExprs[i*2+1] = 0.0;
			bOut[i] = 0.0;
        }
        age = 0;
    }

    __device__ void phenotype_mutate(gene M, curandState_t *rndState)
    {
        unsigned i;
        double incr = params->phenoIncr;
        double geneExpr = geneExprs[M*2] - geneExprs[M*2+1];
        curandState_t localState;

        if (!(curand_uniform_double(rndState) <= params->chancePhenoMut))
            return;
        localState = *rndState;

        if (geneExprs[M*2] >= params->mutThresh
         || geneExprs[M*2+1] >= params->mutThresh) {
            // down-regulation
            if (geneExpr < 0.0) {
                for (i = 0; i < params->nPheno; i++) {
                    incr *= params->downregPhenoMap[M*params->nPheno+i];
                    adjust_phenotype(i, incr);
                }
            // up-regulation
            } else if (geneExpr > 0.0) {
                for (i = 0; i < params->nPheno; i++) {
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
        double incr = params->exprAdjMaxIncr;
        bool posMut = positively_mutated(M), posMutm;
        curandState_t localState = *rndState;

        for (m = 0; m < nGenes; m++) {
            if (curand_uniform_double(&localState) > params->chanceExprAdj)
                continue;
            posMutm = positively_mutated((gene) m);
            if ((posMut && posMutm) || (!posMut && !posMutm))
                continue;
            incr *= curand_uniform_double(&localState);
            if (params->geneRelations[M*nGenes+m] == YES) {
                if (posMut) {
                    if (params->geneType[m] == SUPPR) geneExprs[m*2+1] += incr;
                    else geneExprs[m*2] += incr;
                } else {
                    if (params->geneType[m] == SUPPR)
                        geneExprs[m*2+1] = max(0.0, geneExprs[m*2+1] - incr);
                    else
                        geneExprs[m*2] = max(0.0, geneExprs[m*2] - incr);
                }
            }
        }

        *rndState = localState;
    }

    __device__ void mutate(GeneExprNN *NN, double *NNOut,
                           curandState_t *rndState)
	{
        unsigned m;
        double rnd;

        if (state == EMPTY) return;

        for (m = 0; m < NN->nOut; m++) {
            if (curand_uniform_double(rndState) <= params->chanceUpreg)
                geneExprs[m*2] += NNOut[m];
            else
                geneExprs[m*2+1] += NNOut[m];
        }

        rnd = curand_uniform_double(rndState);
        if (rnd <= 0.1666666667) {
            for (m = 0; m < NN->nOut; m++)
                gene_regulation_adj((gene) m, rndState, NN->nOut);
            NN->mutate(bOut, geneExprs, params->mutThresh);
            for (m = 0; m < NN->nOut; m++)
                phenotype_mutate((gene) m, rndState);
        } else if (rnd > 0.1666666667 && rnd <= 0.3333333333) {
            NN->mutate(bOut, geneExprs, params->mutThresh);
            for (m = 0; m < NN->nOut; m++)
                gene_regulation_adj((gene) m, rndState, NN->nOut);
            for (m = 0; m < NN->nOut; m++)
                phenotype_mutate((gene) m, rndState);
        } else if (rnd > 0.3333333333 && rnd <= 0.5000000000) {
            NN->mutate(bOut, geneExprs, params->mutThresh);
            for (m = 0; m < NN->nOut; m++)
                phenotype_mutate((gene) m, rndState);
            for (m = 0; m < NN->nOut; m++)
                gene_regulation_adj((gene) m, rndState, NN->nOut);
        } else if (rnd > 0.5000000000 && rnd <= 0.6666666667) {
            for (m = 0; m < NN->nOut; m++)
                phenotype_mutate((gene) m, rndState);
            NN->mutate(bOut, geneExprs, params->mutThresh);
            for (m = 0; m < NN->nOut; m++)
                gene_regulation_adj((gene) m, rndState, NN->nOut);
        } else if (rnd > 0.6666666667 && rnd <= 0.8333333333) {
            for (m = 0; m < NN->nOut; m++)
                phenotype_mutate((gene) m, rndState);
            for (m = 0; m < NN->nOut; m++)
                gene_regulation_adj((gene) m, rndState, NN->nOut);
            NN->mutate(bOut, geneExprs, params->mutThresh);
        } else {
            for (m = 0; m < NN->nOut; m++)
                gene_regulation_adj((gene) m, rndState, NN->nOut);
            for (m = 0; m < NN->nOut; m++)
                phenotype_mutate((gene) m, rndState);
            NN->mutate(bOut, geneExprs, params->mutThresh);
        }
    }
};

#endif // __CELL_H__