#include "../headers/general.h"

void cleanup(CA *ca, GUI *gui, CellParams *params)
{
    CudaSafeCall(cudaDeviceSynchronize());

    params->free_resources();
    ca->free_resources();
    gui->free_resources();

    if (params != NULL) {
        CudaSafeCall(cudaFree(params)); params = NULL;
    }
    if (ca != NULL) {
        free(ca); ca = NULL;
    }
    if (gui != NULL) {
        free(gui); gui = NULL;
    }
}

int main(int argc, char *argv[])
{
    unsigned i, j, nGenes = 10, nCarcin = 1, rTC = 0, cX = 0, cY = 0;
    size_t dbl = sizeof(double), st = sizeof(ca_state),
           eff = sizeof(effect), gType = sizeof(gene_type),
           gRel = sizeof(gene_related);

    unsigned T = 8766, gridSize = 256, outSize = 1024;
    bool display = false, save = true, perfectExcision = false;
    int maxTTC = -1;
    unsigned initType = 0;

    CA *ca = (CA*)malloc(sizeof(CA));
    GUI *gui = (GUI*)malloc(sizeof(GUI));

    char **carcinNames = NULL;
    double *diffusion = NULL, *influx = NULL, *outflux = NULL,
           *ic = NULL, *bc = NULL;

    double *carcinMutMap = NULL, *Wx = NULL, *Wy = NULL;
    double alpha = 1000000.0, bias = 0.001;
    dim3 *geneColors = NULL;

    effect *upregPhenoMap = NULL, *downregPhenoMap = NULL;
    ca_state *stateMutMap = NULL, *prolifMutMap = NULL, *diffMutMap = NULL;
    gene_type *geneType = NULL;
    gene_related *geneRelations = NULL;
    CellParams *params = NULL;

    double start, end;

    set_seed();

    if (argc >= 2) display = atoi(argv[1]);
    if (argc >= 3) save = atoi(argv[2]);
    if (argc >= 4) T = atoi(argv[3]);
    if (argc >= 5) gridSize = atoi(argv[4]);
    if (argc >= 6) initType = atoi(argv[5]);
    if (argc >= 7) perfectExcision = atoi(argv[6]);
    if (argc == 8) maxTTC = atoi(argv[7]);

    printf("The CA will run for %d timesteps on a grid of size %dx%d,",
           T, gridSize, gridSize);
    printf(" init type %d, perfectExcision %d, max time TC alive %d.\n",
           initType, perfectExcision, maxTTC);

    start = omp_get_wtime();

    if (initType == 2) nCarcin = 2;
    else if (initType == 3) nCarcin = 0;

    *ca = CA(gridSize, T, nGenes, nCarcin, save, outSize, maxTTC, perfectExcision);
    ca->initialize_memory();

    upregPhenoMap = (effect*)malloc(nGenes*4*eff);
    memset(upregPhenoMap, NONE, nGenes*4*eff);
    upregPhenoMap[  TP53 * 4 + PROLIF] = NEG;
    upregPhenoMap[  TP53 * 4 +  QUIES] = POS;
    upregPhenoMap[  TP53 * 4 +   APOP] = POS;
    upregPhenoMap[  TP73 * 4 +   APOP] = POS;
    upregPhenoMap[    RB * 4 + PROLIF] = NEG;
    upregPhenoMap[    RB * 4 +  QUIES] = POS;
    upregPhenoMap[   P21 * 4 + PROLIF] = NEG;
    upregPhenoMap[   P21 * 4 +  QUIES] = POS;
    upregPhenoMap[   P21 * 4 +   DIFF] = NEG;
    upregPhenoMap[  TP16 * 4 + PROLIF] = NEG;
    upregPhenoMap[  EGFR * 4 + PROLIF] = POS;
    upregPhenoMap[ CCDN1 * 4 +   APOP] = NEG;
    upregPhenoMap[   MYC * 4 + PROLIF] = POS;
    upregPhenoMap[   MYC * 4 +   APOP] = NEG;
    upregPhenoMap[   MYC * 4 +   DIFF] = POS;
    upregPhenoMap[PIK3CA * 4 +   APOP] = NEG;
    upregPhenoMap[   RAS * 4 + PROLIF] = POS;
    upregPhenoMap[   RAS * 4 +   APOP] = NEG;
    upregPhenoMap[   RAS * 4 +   DIFF] = POS;

    downregPhenoMap = (effect*)malloc(nGenes*4*eff);
    memset(downregPhenoMap, NONE, nGenes*4*eff);
    for (i = 0; i < nGenes; i++)
        for (j = 0; j < 4; j++) {
            if (upregPhenoMap[i*4+j] == NONE) continue;
            downregPhenoMap[i*4+j] = (effect) -upregPhenoMap[i*4+j];
        }

    stateMutMap = (ca_state*)malloc((ca->nStates-1)*nGenes*st);
    for (i = 0; i < nGenes; i++) {
        stateMutMap[ NC*nGenes+i] = MNC;
        stateMutMap[MNC*nGenes+i] = MNC;
        stateMutMap[ SC*nGenes+i] = MSC;
        stateMutMap[MSC*nGenes+i] = MSC;
        stateMutMap[CSC*nGenes+i] = CSC;
        stateMutMap[ TC*nGenes+i] =  TC;
    }
    for (i = 2; i < 4; i++) {
        stateMutMap[i*nGenes+MYC] = CSC;
        stateMutMap[i*nGenes+RAS] = CSC;
    }
    prolifMutMap = (ca_state*)malloc((ca->nStates-1)*nGenes*st);
    memcpy(prolifMutMap, stateMutMap, (ca->nStates-1)*nGenes*st);

    diffMutMap = (ca_state*)malloc((ca->nStates-1)*(nGenes+1)*st);
    memset(diffMutMap, ERR, (ca->nStates-1)*(nGenes+1)*st);
    for (i = 0; i < nGenes+1; i++) {
        diffMutMap[ SC*(nGenes+1)+i] =  NC;
        diffMutMap[MSC*(nGenes+1)+i] = MNC;
        diffMutMap[CSC*(nGenes+1)+i] =  TC;
    }
    diffMutMap[SC*nGenes+(TP53+1)] = MNC;
    for (i = 2; i < 4; i++) {
        diffMutMap[i*(nGenes+1)+(MYC+1)] = CSC;
        diffMutMap[i*(nGenes+1)+(RAS+1)] = CSC;
    }

    geneType = (gene_type*)malloc(nGenes*gType);
    geneType[ TP53] = SUPPR; geneType[TP73] = SUPPR; geneType[    RB] = SUPPR;
    geneType[  P21] = SUPPR; geneType[TP16] = SUPPR; geneType[  EGFR] =  ONCO;
    geneType[CCDN1] =  ONCO; geneType[ MYC] =  ONCO; geneType[PIK3CA] =  ONCO;
    geneType[  RAS] =  ONCO;
    geneRelations = (gene_related*)malloc(nGenes*nGenes*gRel);
    memset(geneRelations, NO, nGenes*nGenes*gRel);
    for (i = 1; i < nGenes; i++) geneRelations[TP53*nGenes+i] = YES;
    geneRelations[   RB * nGenes +   TP53] = YES;
    geneRelations[   RB * nGenes +  CCDN1] = YES;
    geneRelations[CCDN1 * nGenes +    P21] = YES;
    geneRelations[  MYC * nGenes +    P21] = YES;
    geneRelations[  MYC * nGenes +    RAS] = YES;
    geneRelations[  RAS * nGenes +  CCDN1] = YES;
    geneRelations[  RAS * nGenes +    MYC] = YES;

    CudaSafeCall(cudaMallocManaged((void**)&params, sizeof(CellParams)));
    *params = CellParams(ca->nStates, nGenes, alpha, upregPhenoMap,
                         downregPhenoMap, stateMutMap, prolifMutMap,
                         diffMutMap, geneType, geneRelations);
    params->prefetch_memory(ca->devId2, ca->nStates, nGenes);
    CudaSafeCall(cudaMemPrefetchAsync(params,
                                      sizeof(CellParams),
                                      ca->devId2, NULL));

    free(upregPhenoMap); free(downregPhenoMap);
    upregPhenoMap = NULL; downregPhenoMap = NULL;
    free(stateMutMap); free(prolifMutMap); free(diffMutMap);
    stateMutMap = NULL; prolifMutMap = NULL; diffMutMap = NULL;
    free(geneType); free(geneRelations);
    geneType = NULL; geneRelations = NULL;

    if (nCarcin != 0)
        carcinMutMap = (double*)malloc(nCarcin*nGenes*dbl);
	Wx = (double*)malloc(ca->nGenes*(nCarcin+1)*dbl);
	Wy = (double*)malloc(ca->nGenes*nGenes*dbl);

    if (nCarcin != 0) {
        for (i = 0; i < nGenes; i++)
            for (j = 0; j < nCarcin; j++)
                carcinMutMap[j*nCarcin+i] = 1.0;
    }
    memset(Wx, 0, ca->nGenes*(nCarcin+1)*dbl);
    for (i = 0; i < nGenes; i++)
        for (j = 0; j < nCarcin+1; j++) {
            if (j == nCarcin) {
                Wx[i*(nCarcin+1)+j] = 1e-6;
                continue;
            }
            Wx[i*(nCarcin+1)+j] = carcinMutMap[j*nCarcin+i];
        }
    if (nCarcin != 0) { free(carcinMutMap); carcinMutMap = NULL; }

    memset(Wy, 0, nGenes*nGenes*dbl);
    Wy[  TP53 * nGenes +   TP53] =  1.0;
    Wy[  TP73 * nGenes +   TP73] =  0.1;
    Wy[    RB * nGenes +     RB] =  0.3;
    Wy[   P21 * nGenes +    P21] =  0.1;
    Wy[  TP16 * nGenes +   TP16] =  0.1;
    Wy[  EGFR * nGenes +   EGFR] =  0.1;
    Wy[ CCDN1 * nGenes +  CCDN1] =  0.2;
    Wy[   MYC * nGenes +    MYC] =  0.3;
    Wy[PIK3CA * nGenes + PIK3CA] =  0.1;
    Wy[   RAS * nGenes +    RAS] =  0.3;
    for (i = 1; i < nGenes; i++) Wy[i*nGenes+TP53] = 0.01;
    Wy[  TP53 * nGenes +     RB] =  0.01;
    Wy[ CCDN1 * nGenes +     RB] =  0.01;
    Wy[   P21 * nGenes +  CCDN1] =  0.01;
    Wy[   P21 * nGenes +    MYC] = -0.01;
    Wy[   RAS * nGenes +    MYC] =  0.01;
    Wy[ CCDN1 * nGenes +    RAS] =  0.01;
    Wy[   MYC * nGenes +    RAS] =  0.01;

    if (nCarcin != 0) {
        diffusion = (double*)malloc(nCarcin*dbl);
        diffusion[0] = 4.5590004e-2; // cm^2/h
        if (nCarcin == 2) diffusion[1] = 2.9487515e-2; // cm^2/h
        influx = (double*)malloc(nCarcin*dbl);
        influx[0] = 2.1755778; // microg/cm^3*h
        if (nCarcin == 2) influx[1] = 2.6498538e-2; // microg/cm^3*h
        outflux = (double*)malloc(nCarcin*dbl);
        outflux[0] = 0.0; // g/cm^3*h
        if (nCarcin == 2) outflux[1] = 3.9604607e-3; // microg/cm^3*h
        ic = (double*)malloc(nCarcin*dbl);
        memset(ic, 0, nCarcin*dbl);
        bc = (double*)malloc(nCarcin*dbl);
        memset(bc, 0, nCarcin*dbl);
    }

    geneColors = (dim3*)malloc(nGenes*sizeof(dim3));
    geneColors[  TP53] = dim3( 84,  48,   5); // Dark brown
    geneColors[  TP73] = dim3(140,  81,  10); // Light brown
    geneColors[    RB] = dim3(191, 129,  45); // Brown orange
    geneColors[   P21] = dim3(223, 194, 125); // Pale brown
    geneColors[  TP16] = dim3(246, 232, 195); // Pale
    geneColors[  EGFR] = dim3(199, 234, 229); // Baby blue
    geneColors[ CCDN1] = dim3(128, 205, 193); // Blueish green
    geneColors[   MYC] = dim3( 53, 151, 143); // Tourquois
    geneColors[PIK3CA] = dim3(  1, 102,  94); // Dark green
    geneColors[   RAS] = dim3(  0,  60,  48); // Forest green

    double weightStates[7] = { 0.45, 0.22, 0.01, 0.005, 0.005, 0.02, 0.29 };
    if (initType != 1) {
        memset(weightStates, 0, 7*dbl);
        weightStates[NC] = 0.7;
        weightStates[SC] = 0.01;
        weightStates[EMPTY] = 0.29;
    }

    if (initType == 1) {
        rTC = ca->gridSize / 10;
        cX = rand() % ca->gridSize;
        cY = rand() % ca->gridSize;
    }

    ca->init(diffusion, influx, outflux, ic, bc,
             Wx, Wy, alpha, bias, geneColors,
             params, weightStates, rTC, cX, cY);

    if (initType == 1) {
        dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                    NBLOCKS(ca->gridSize, ca->blockSize));
        dim3 threads(ca->blockSize, ca->blockSize);
        init_grid<<< blocks, threads >>>(ca->prevGrid, ca->gridSize,
                                         ca->NN, 100);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
        update_states<<< blocks, threads >>>(ca->prevGrid, ca->gridSize,
                                             ca->nGenes);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
        cells_gpu_to_gpu_copy<<< blocks, threads >>>(ca->prevGrid,
                                                     ca->newGrid,
                                                     ca->gridSize,
                                                     ca->nGenes);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    free(Wx); free(Wy); free(geneColors);
    Wx = NULL; Wy = NULL; geneColors = NULL;
    if (nCarcin != 0) {
        free(diffusion); free(influx); free(outflux); free(ic); free(bc);
        diffusion = NULL; influx = NULL; outflux = NULL; ic = NULL; bc = NULL;
    }

    if (nCarcin != 0) {
        carcinNames = (char**)malloc(sizeof(char*));
        carcinNames[0] = (char*)malloc(8);
        sprintf(carcinNames[0], "%s", "Alcohol");
        if (nCarcin == 2) {
            carcinNames[1] = (char*)malloc(8);
            sprintf(carcinNames[1], "%s", "Tobacco");
        }
    }

    *gui = GUI(outSize, outSize, ca, display, perfectExcision,
               gridSize, T, nCarcin, carcinNames);
    if (nCarcin != 0) {
        free(carcinNames[0]); carcinNames[0] = NULL;
        if (nCarcin == 2) {
            free(carcinNames[1]); carcinNames[1] = NULL;
        }
        free(carcinNames); carcinNames = NULL;
    }

    if (!gui->windows[0] || !gui->windows[1]
     || !gui->windows[2] || (nCarcin != 0 && !gui->windows[3])) {
        cleanup(ca, gui, params);
        return 1;
    }

    end = omp_get_wtime();
    printf("It took %f seconds to initialize the memory.\n", end - start);

    ca->animate(1, gui);

    cleanup(ca, gui, params);

    return 0;
}