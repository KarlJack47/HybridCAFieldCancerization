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
    bool display = false, save = false, perfectExcision = false;
    int opt, maxTTC = -1;
    unsigned initType = 0;

    CA *ca = (CA*)malloc(sizeof(CA));
    GUI *gui = (GUI*)malloc(sizeof(GUI));

    char **carcinNames = NULL;
    double *diffusion = NULL, *influx = NULL, *outflux = NULL,
           *ic = NULL, *bc = NULL;

    double *carcinMutMap = NULL, *Wx = NULL, *Wy = NULL;
    double mutRatePerMitosis = 1e-8, alpha = 1000000.0, bias = 0.001;
    dim3 *geneColors = NULL;

    effect *upregPhenoMap = NULL, *downregPhenoMap = NULL;
    ca_state *stateMutMap = NULL, *prolifMutMap = NULL, *diffMutMap = NULL;
    gene_type *geneType = NULL;
    gene_related *geneRelations = NULL;
    CellParams *params = NULL;

    double start, end;

    set_seed();

    while ((opt = getopt(argc, argv, ":dst:g:i:pe:")) != -1) {
        switch(opt)
        {
            case 'd':
                display = true;
                break;
            case 's':
                save = true;
                break;
            case 't':
                T = atoi(optarg);
                break;
            case 'g':
                gridSize = atoi(optarg);
                break;
            case 'i':
                initType = atoi(optarg);
                break;
            case 'p':
                perfectExcision = true;
                break;
            case 'e':
                maxTTC = atoi(optarg);
                break;
            case ':':
                printf("Option needs a value\n");
                break;
            case '?':
                printf("Unknown option: %c\n", optopt);
                break;
        }
    }
    for (; optind < argc; optind++)
        printf("extra argument: %s\n", argv[optind]);

    printf("The CA will run for %d timesteps on a grid of size %dx%d,",
           T, gridSize, gridSize);
    printf(" init type %d, perfectExcision %d, max time TC alive %d.\n",
           initType, perfectExcision, maxTTC);

    start = omp_get_wtime();

    if (initType == 2) nCarcin = 2;
    else if (initType == 3) nCarcin = 0;

    *ca = CA(gridSize, T, nGenes, nCarcin, save, outSize, maxTTC, perfectExcision);
    ca->initialize_memory();

    upregPhenoMap = (effect*)calloc(nGenes*4, eff);
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

    downregPhenoMap = (effect*)calloc(nGenes*4, eff);
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
    memset(diffMutMap, ERROR, (ca->nStates-1)*(nGenes+1)*st);
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
    geneRelations = (gene_related*)calloc(nGenes*nGenes, gRel);
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

    free(upregPhenoMap); free(downregPhenoMap);
    upregPhenoMap = NULL; downregPhenoMap = NULL;
    free(stateMutMap); free(prolifMutMap); free(diffMutMap);
    stateMutMap = NULL; prolifMutMap = NULL; diffMutMap = NULL;
    free(geneType); free(geneRelations);
    geneType = NULL; geneRelations = NULL;

    if (nCarcin != 0)
        carcinMutMap = (double*)malloc(nCarcin*nGenes*dbl);
	Wx = (double*)calloc(ca->nGenes*(nCarcin+1), dbl);
	Wy = (double*)calloc(ca->nGenes*nGenes, dbl);

    if (nCarcin != 0) {
        for (i = 0; i < nGenes; i++)
            for (j = 0; j < nCarcin; j++)
                carcinMutMap[j*nCarcin+i] = 1.0;
    }
    for (i = 0; i < nGenes; i++)
        for (j = 0; j < nCarcin+1; j++) {
            if (j == nCarcin) {
                Wx[i*(nCarcin+1)+j] = mutRatePerMitosis;
                continue;
            }
            Wx[i*(nCarcin+1)+j] = carcinMutMap[j*nCarcin+i];
        }
    if (nCarcin != 0) { free(carcinMutMap); carcinMutMap = NULL; }

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
        if (nCarcin == 2) diffusion[1] = 2.94875146e-2; // cm^2/h
        influx = (double*)malloc(nCarcin*dbl);
        influx[0] = 2.1755778; // microg/cm^3*h
        if (nCarcin == 2) influx[1] = 5.04734057e-2; // microg/cm^3*h
        outflux = (double*)malloc(nCarcin*dbl);
        outflux[0] = 0.0; // g/cm^3*h
        if (nCarcin == 2) outflux[1] = 7.54113434e-3; // microg/cm^3*h
        ic = (double*)calloc(nCarcin, dbl);
        bc = (double*)calloc(nCarcin, dbl);
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

        cells_gpu_to_gpu_cpy<<< blocks, threads >>>(
            ca->newGrid, ca->prevGrid, ca->gridSize, ca->nGenes
        );
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
        carcinNames[0] = (char*)calloc(8, 1);
        strcat(carcinNames[0], "Alcohol");
        if (nCarcin == 2) {
            carcinNames[1] = (char*)calloc(8, 1);
            strcat(carcinNames[1], "Tobacco");
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