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

__device__ double func1(double x, double y, unsigned N)
{
    return (sin(x) * sin(y) + 1.0) / 2.0;
}
__device__ double func2(double x, double y, unsigned N)
{
    double mux = (N / 2.0) - 1.0, muy = mux,
           sigmax = (N / 20.0), sigmay = sigmax;

    return exp(-0.5 * (pow(x - mux, 2) / (sigmax * sigmax)
                     + pow(y - muy, 2) / (sigmay * sigmay)));
}

__device__ SensitivityFunc pFunc[2] = { func1, func2 };

int main(int argc, char *argv[])
{
    unsigned i = 0, j = 0, k = 0, l = 0, nStates = 7, nPheno = 4, nGenes = 10,
             nCarcin = 1, maxNCarcin = 2, rTC = 0, cX = 0, cY = 0, nSim = 1;
    size_t dbl = sizeof(double), st = sizeof(ca_state),
           eff = sizeof(effect), gRel = sizeof(gene_related);

    unsigned T = 8766, gridSize = 256, outSize = 1024;
    bool display = false, save = false, perfectExcision = false;
    int opt, maxTTC = -1;
    unsigned initType = 0;

    CA *ca = (CA*)malloc(sizeof(CA));
    GUI *gui = (GUI*)malloc(sizeof(GUI));

    char **carcinNames = NULL;
    double diffusion[maxNCarcin], influx[maxNCarcin], outflux[maxNCarcin],
           ic[maxNCarcin], bc[maxNCarcin];
    bool carcinogens[maxNCarcin] = { false };
    int maxTInflux[maxNCarcin], maxTNoInflux[maxNCarcin];
    double exposureTime[maxNCarcin];
    unsigned nFunc = 2, funcIdx[maxNCarcin];
    SensitivityFunc func[nFunc];

    double carcinMutMap[maxNCarcin*nGenes], Wx[(maxNCarcin+1)*nGenes],
           Wy[nGenes*nGenes];
    double mutRatePerMitosis = 1e-7, alpha = 1000000.0, bias = 0.001;
    dim3 geneColors[nGenes];

    effect upregPhenoMap[nPheno*nGenes], downregPhenoMap[nPheno*nGenes];
    ca_state diffMap[nStates-4];
    gene_type geneType[nGenes];
    gene_related geneRelations[nGenes*nGenes];
    CellParams *params = NULL;
    double *weightStates = NULL;

    char outDir[22] = { '\0' }; struct stat dirStat = { 0 };
    time_t currTime; struct tm *timeinfo = NULL; char timeStamp[15] = { '\0' };
    char outSimDir[num_digits(nSim)] = { '\0' };

    dim3 blocks, threads;

    double start, end;

    set_seed();

    start = omp_get_wtime();

    CudaSafeCall(cudaMemcpyFromSymbol(func, pFunc,
                                      2 * sizeof(SensitivityFunc)));
    for (i = 0; i < maxNCarcin; i++) {
       maxTInflux[i] = -1;
       maxTNoInflux[i] = -1;
       exposureTime[i] = 24;
       funcIdx[i] = 0;
    }

    i = 0;
    while ((opt = getopt(argc, argv, ":dst:g:i:c:a:b:x:h:pe:n:f:")) != -1) {
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
            case 'c':
                if (atoi(optarg) < maxNCarcin)
                    carcinogens[atoi(optarg)] = true;
                break;
            case 'a':
                if (i < maxNCarcin)
                    maxTInflux[i++] = atoi(optarg);
                break;
            case 'b':
                if (j < maxNCarcin)
                    maxTNoInflux[j++] = atoi(optarg);
                break;
            case 'x':
                if (k < maxNCarcin)
                    exposureTime[k++] = atoi(optarg);
                break;
            case 'h':
                if (l < maxNCarcin && atoi(optarg) < nFunc)
                    funcIdx[l++] = atoi(optarg);
            case 'p':
                perfectExcision = true;
                break;
            case 'e':
                maxTTC = atoi(optarg);
                break;
            case 'n':
                nSim = atoi(optarg);
                break;
            case 'f':
                strcpy(outDir, optarg);
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

    if (strlen(outDir) == 0) {
        time(&currTime); timeinfo = localtime(&currTime);
        strftime(timeStamp, 15, "%Y%m%d%H%M%S", timeinfo);
        strcat(outDir, "output_");
        strcat(outDir, timeStamp);

        if (stat(outDir, &dirStat) == -1) {
            mkdir(outDir, 0700);
            chdir(outDir);
        }
    }

    printf("The CA will run for %d timesteps on a grid of size %dx%d,",
           T, gridSize, gridSize);
    printf(" init type %d, perfectExcision %d, max time TC alive %d.\n",
           initType, perfectExcision, maxTTC); 

    if (initType == 0) {
        for (i = 0; i < maxNCarcin; i++)
            if (carcinogens[i]) break;
        if (i == maxNCarcin) carcinogens[0] = true;
    } else if (initType == 2) {
        nCarcin = maxNCarcin;
        for (i = 0; i < maxNCarcin; i++)
            carcinogens[i] = true;
    } else if (initType == 3) {
        nCarcin = 0;
        for (i = 0; i < maxNCarcin; i++)
            carcinogens[i] = false;
    }

    *ca = CA(gridSize, T, nGenes, nCarcin, maxNCarcin,
             save, outSize, maxTTC, perfectExcision);
    ca->initialize_memory();
    blocks.x = NBLOCKS(gridSize, ca->blockSize); blocks.y = blocks.x;
    threads.x = ca->blockSize; threads.y = threads.x;

    memset(upregPhenoMap, NONE, nPheno*nGenes*eff);
    upregPhenoMap[  TP53 * nPheno + PROLIF] = NEG;
    upregPhenoMap[  TP53 * nPheno +  QUIES] = POS;
    upregPhenoMap[  TP53 * nPheno +   APOP] = POS;
    upregPhenoMap[  TP73 * nPheno +   APOP] = POS;
    upregPhenoMap[    RB * nPheno + PROLIF] = NEG;
    upregPhenoMap[    RB * nPheno +  QUIES] = POS;
    upregPhenoMap[   P21 * nPheno + PROLIF] = NEG;
    upregPhenoMap[   P21 * nPheno +  QUIES] = POS;
    upregPhenoMap[   P21 * nPheno +   DIFF] = NEG;
    upregPhenoMap[  TP16 * nPheno + PROLIF] = NEG;
    upregPhenoMap[  EGFR * nPheno + PROLIF] = POS;
    upregPhenoMap[ CCDN1 * nPheno +   APOP] = NEG;
    upregPhenoMap[   MYC * nPheno + PROLIF] = POS;
    upregPhenoMap[   MYC * nPheno +   APOP] = NEG;
    upregPhenoMap[   MYC * nPheno +   DIFF] = POS;
    upregPhenoMap[PIK3CA * nPheno +   APOP] = NEG;
    upregPhenoMap[   RAS * nPheno + PROLIF] = POS;
    upregPhenoMap[   RAS * nPheno +   APOP] = NEG;
    upregPhenoMap[   RAS * nPheno +   DIFF] = POS;

    memset(downregPhenoMap, NONE, nPheno*nGenes*eff);
    for (i = 0; i < nGenes; i++)
        for (j = 0; j < nPheno; j++) {
            if (upregPhenoMap[i*nPheno+j] == NONE) continue;
            downregPhenoMap[i*nPheno+j] = (effect) -upregPhenoMap[i*nPheno+j];
        }

    memset(diffMap, ERROR, (nStates-4)*st);
    diffMap[ SC-2] =  NC;
    diffMap[MSC-2] = MNC;
    diffMap[CSC-2] =  TC;

    geneType[ TP53] = SUPPR; geneType[TP73] = SUPPR; geneType[    RB] = SUPPR;
    geneType[  P21] = SUPPR; geneType[TP16] = SUPPR; geneType[  EGFR] =  ONCO;
    geneType[CCDN1] =  ONCO; geneType[ MYC] =  ONCO; geneType[PIK3CA] =  ONCO;
    geneType[  RAS] =  ONCO;
    memset(geneRelations, 0, nGenes*nGenes*gRel);
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
                         downregPhenoMap, diffMap, geneType, geneRelations);

	memset(Wx, 0, nGenes*(maxNCarcin+1)*dbl);
	memset(Wy, 0, nGenes*nGenes*dbl);
	memset(carcinMutMap, 0, maxNCarcin*nGenes*dbl);
	carcinMutMap[  TP53] =  1.0; carcinMutMap[nGenes +   TP53] = -1.0;
	carcinMutMap[  TP73] =  0.0; carcinMutMap[nGenes +   TP73] =  0.0;
	carcinMutMap[    RB] =  0.0; carcinMutMap[nGenes +     RB] = -1.0;
	carcinMutMap[   P21] =  1.0; carcinMutMap[nGenes +    P21] = -1.0;
	carcinMutMap[  TP16] =  1.0; carcinMutMap[nGenes +   TP16] =  0.0;
	carcinMutMap[  EGFR] =  1.0; carcinMutMap[nGenes +   EGFR] =  1.0;
	carcinMutMap[ CCDN1] =  1.0; carcinMutMap[nGenes +  CCDN1] =  1.0;
	carcinMutMap[   MYC] =  0.0; carcinMutMap[nGenes +    MYC] =  1.0;
	carcinMutMap[PIK3CA] =  0.0; carcinMutMap[nGenes + PIK3CA] =  1.0;
	carcinMutMap[   RAS] =  1.0; carcinMutMap[nGenes +    RAS] =  1.0;
    for (i = 0; i < nGenes; i++)
        for (j = 0; j < maxNCarcin+1; j++) {
            if (j == maxNCarcin) {
                Wx[i*(maxNCarcin+1)+j] = mutRatePerMitosis;
                continue;
            }
            Wx[i*(maxNCarcin+1)+j] = carcinMutMap[j*nGenes+i];
        }
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

    diffusion[0] = 4.5590004e-2; // cm^2/h
    influx[0] = 2.1755778; // microg/cm^3*h
    outflux[0] = 0.0; // g/cm^3*h
    diffusion[1] = 2.94875146e-2; // cm^2/h
    influx[1] = 5.04734057e-2; // microg/cm^3*h
    outflux[1] = 7.54113434e-3; // microg/cm^3*h
    memset(ic, 0, maxNCarcin*dbl);
    memset(bc, 0, maxNCarcin*dbl);

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

    CudaSafeCall(cudaMallocManaged((void**)&weightStates, nStates*dbl));
    weightStates[NC] = 0.45; weightStates[MNC] = 0.22; weightStates[SC] = 0.01; 
    weightStates[MSC] = 0.005; weightStates[CSC] = 0.005;
    weightStates[TC] = 0.02; weightStates[EMPTY] = 0.29;
    if (initType != 1) {
        memset(weightStates, 0, nStates*dbl);
        weightStates[NC] = 0.7;
        weightStates[SC] = 0.01;
        weightStates[EMPTY] = 0.29;
    }

    if (initType == 1) {
        rTC = ca->gridSize / 10;
        cX = rand() % ca->gridSize;
        cY = rand() % ca->gridSize;
    }

    ca->init(diffusion, influx, outflux, ic, bc, maxTInflux,
             maxTNoInflux, exposureTime, carcinogens, func, nFunc, funcIdx,
             Wx, Wy, alpha, bias, geneColors,
             params, weightStates, rTC, cX, cY);

    if (initType == 1) {
        init_grid<<< blocks, threads >>>(ca->prevGrid, ca->gridSize,
                                         ca->NN, 100);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        update_states<<< blocks, threads >>>(ca->prevGrid, ca->gridSize,
                                             ca->nGenes, 0);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        cells_gpu_to_gpu_cpy<<< blocks, threads >>>(
            ca->newGrid, ca->prevGrid, ca->gridSize, ca->nGenes
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    carcinNames = (char**)malloc(maxNCarcin*sizeof(char*));
    carcinNames[0] = (char*)calloc(8, 1);
    strcat(carcinNames[0], "Alcohol");
    carcinNames[1] = (char*)calloc(8, 1);
    strcat(carcinNames[1], "Tobacco");

    *gui = GUI(outSize, outSize, ca, display, &ca->perfectExcision,
               gridSize, &ca->maxT, &ca->nCarcin, maxNCarcin, 
               ca->carcinogens, carcinNames);
    for (i = 0; i < maxNCarcin; i++) {
        free(carcinNames[i]); carcinNames[i] = NULL;
    }
    free(carcinNames); carcinNames = NULL;

    if (!gui->windows[0] || !gui->windows[1]
     || !gui->windows[2] || !gui->windows[3]) {
        cleanup(ca, gui, params);
        return 1;
    }

    end = omp_get_wtime();
    printf("It took %f seconds to initialize the memory.\n", end - start);

    k = 1;
    do {
        sprintf(outSimDir, "%d", k); 
        if (stat(outSimDir, &dirStat) == -1) {
            mkdir(outSimDir, 0700);
            chdir(outSimDir);
        }

        printf("Starting simulation %d\n", k);

        ca->animate(1, gui);

        printf("Done simulation %d\n", k);
        k++;
        chdir("..");

        if (k == nSim) continue;

        cX = rand() % gridSize;
        cY = rand() % gridSize;
        reset_grid<<< blocks, threads >>>(
            ca->prevGrid, gridSize, nGenes, weightStates, rTC,
            cX, cY, ca->cellLifeSpan, ca->cellCycleLen
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        cells_gpu_to_gpu_cpy<<< blocks, threads >>>(ca->newGrid, ca->prevGrid, 
                                                    gridSize, nGenes);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        gui->windowsShouldClose = false;
        gui->display = display; ca->save = save;
        ca->perfectExcision = perfectExcision; ca->maxTTCAlive = maxTTC;
        ca->nCarcin = nCarcin;
        for (i = 0; i < maxNCarcin; i++) {
            ca->pdes[i].t = 0; ca->pdes[i].nCycles = 1;
            ca->pdes[i].maxTInflux = maxTInflux[i];
            ca->pdes[i].maxTNoInflux = maxTNoInflux[i];
            ca->pdes[i].exposureTime = exposureTime[i];
            ca->carcinogens[i] = carcinogens[i];
            ca->pdes[i].init(ca->blockSize, NULL);
            CudaSafeCall(cudaDeviceSynchronize());
        }
        *ca->cscFormed = false;
        for (i = 0; i < ca->exciseCount; i++) {
            ca->tcFormed[i] = false;
            ca->timeTCDead[i] = 1;
            ca->radius[i] = 0;
        }
        ca->exciseCount = 0;
        ca->radius[0] = gridSize;
        ca->centerX[0] = gridSize / 2 - 1; ca->centerY[0] = ca->centerX[0];
    } while(k <= nSim);

    CudaSafeCall(cudaFree(weightStates));
    cleanup(ca, gui, params);

    return 0;
}