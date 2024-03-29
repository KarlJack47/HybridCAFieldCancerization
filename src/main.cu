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
    int nGPU, retVal = 0;
    unsigned i = 0, j = 0, k = 0, l = 0, m = 0,
             nStates = 7, nPheno = 4, nGenes = 10, nCarcin = 0, maxNCarcin = 2,
             rTC = 0, cX = 0, cY = 0, nSim = 1;
    size_t dbl = sizeof(double), st = sizeof(ca_state),
           eff = sizeof(effect), gRel = sizeof(gene_related);

    unsigned T = 8766, gridSize = 256, outSize = 1024;
    bool display = false, save = false, saveCellData = false,
         pauseOnFirstTC = false, perfectExcision = false,
         removeField = false;
    int opt, maxTTC = -1, excisionTime = -1;
    unsigned initType = 0, maxNeighDepth=1;

    CA *ca = (CA*)malloc(sizeof(CA));
    GUI *gui = (GUI*)malloc(sizeof(GUI));

    char **carcinNames = NULL;
    unsigned carcinType[maxNCarcin];
    double diffusion[maxNCarcin], influx[maxNCarcin], outflux[maxNCarcin],
           ic[maxNCarcin], bc[maxNCarcin];
    bool activeCarcin[maxNCarcin];
    int maxTInflux[maxNCarcin], maxTNoInflux[maxNCarcin];
    double exposureTime[maxNCarcin];
    unsigned nFunc = 4, funcIdx[maxNCarcin];
    SensitivityFunc func[nFunc];

    unsigned minMut = 4; // Head and Neck (2-8 depending on cancer type)
    double chanceCSCForm = 2.5e-6;
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

    char outDir[500] = { '\0' }; struct stat dirStat = { 0 };
    time_t currTime; struct tm *timeinfo = NULL; char timeStamp[15] = { '\0' };
    char outSimDir[num_digits(nSim)+1] = { '\0' };

    dim3 blocks, threads;

    double start, end;

    set_seed();

    start = omp_get_wtime();

    CudaSafeCall(cudaMemcpyFromSymbol(func, pFunc,
                                      nFunc * sizeof(SensitivityFunc)));
    for (i = 0; i < maxNCarcin; i++) {
       carcinType[i] = 1;
       activeCarcin[i] = false;
       maxTInflux[i] = -1;
       maxTNoInflux[i] = -1;
       exposureTime[i] = 24;
       funcIdx[i] = 0;
    }

    i = 0;
    while ((opt = getopt(argc, argv, ":dsCt:g:i:c:a:b:x:h:o:m:r:kj:pqu:e:n:f:")) != -1) {
        switch(opt)
        {
            case 'd':
                display = true;
                break;
            case 's':
                save = true;
                break;
            case 'C':
                saveCellData = true;
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
                if (atoi(optarg) < maxNCarcin) {
                    activeCarcin[atoi(optarg)] = true;
                    nCarcin++;
                }
                break;
            case 'a':
                if (i < maxNCarcin) {
                    maxTInflux[i++] = atoi(optarg);
                    if (i == maxNCarcin) i = 0;
                }
                break;
            case 'b':
                if (j < maxNCarcin) {
                    maxTNoInflux[j++] = atoi(optarg);
                    if (j == maxNCarcin) j = 0;
                }
                break;
            case 'x':
                if (k < maxNCarcin) {
                    exposureTime[k++] = atoi(optarg);
                    if (k == maxNCarcin) k = 0;
                }
                break;
            case 'h':
                if (l < maxNCarcin && atoi(optarg) < nFunc) {
                    funcIdx[l++] = atoi(optarg);
                    if (l == maxNCarcin) l = 0;
                }
                break;
            case 'o':
                if (m < maxNCarcin && atoi(optarg) < 3) {
                    carcinType[m++] = atoi(optarg);
                    if (m == maxNCarcin) m = 0;
                }
                break;
            case 'm':
                if (atoi(optarg) > 1 && atoi(optarg) < 9)
                    minMut = atoi(optarg);
                break;
            case 'r':
                if (atof(optarg) <= 1.0)
                    chanceCSCForm = atof(optarg);
                break;
            case 'k':
                pauseOnFirstTC = true;
                break;
            case 'j':
                excisionTime = atoi(optarg);
                break;
            case 'p':
                perfectExcision = true;
                break;
            case 'q':
                removeField = true;
                break;
            case 'u':
                maxNeighDepth = atoi(optarg);
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

    if (initType == 0) {
        for (i = 0; i < maxNCarcin; i++)
            if (activeCarcin[i]) break;
        if (i == maxNCarcin) activeCarcin[0] = true;
    } else if (initType == 2) {
        nCarcin = maxNCarcin;
        for (i = 0; i < maxNCarcin; i++)
            activeCarcin[i] = true;
    } else if (initType == 3) {
        nCarcin = 0;
        for (i = 0; i < maxNCarcin; i++)
            activeCarcin[i] = false;
    }

    printf("The CA will run %d times for %d timesteps on a grid of size %dx%d,\n",
           nSim, T, gridSize, gridSize);
    printf("init type %d, perfectExcision %d, max time TC alive %d,\n",
           initType, perfectExcision, maxTTC);
    printf("number of carcinogens %d", nCarcin);
    for (i = 0; i < maxNCarcin; i++) {
        if (!activeCarcin[i]) continue;
        printf(", carcin type (%d, %d), exposure time (%d, %g),",
               i, carcinType[i], i, exposureTime[i]);
        printf(" time influx (%d, %d), time no influx (%d, %d),",
               i, maxTInflux[i], i, maxTNoInflux[i]);
        printf(" function index (%d, %d)", i, funcIdx[i]);
    }
    printf(",\n");
    printf("chance CSC Form %g, minimum number of mutations %d, ",
           chanceCSCForm, minMut);
    printf("pauseOnFirstTC %d, excisionTime %d, removeField %d, maxNeighDepth %d\n",
           pauseOnFirstTC, excisionTime, removeField, maxNeighDepth);

    *ca = CA(gridSize, T, nGenes, nCarcin, maxNCarcin, save, saveCellData,
             outSize, maxTTC, perfectExcision, removeField, maxNeighDepth,
             pauseOnFirstTC, excisionTime);
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
    *params = CellParams(ca->nStates, nGenes, minMut, chanceCSCForm, alpha,
                         upregPhenoMap, downregPhenoMap, diffMap, geneType,
                         geneRelations, ca->cellCycleLen, ca->cellLifeSpan);

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

    // ethanol
    diffusion[0] = 2.18e-2; // cm^2/h
    influx[0] = 2.009e-3; // g/cm^3*h
    outflux[0] = 0.0; // g/cm^3*h
    // nicotine
    diffusion[1] = 1.56e-2; // cm^2/h
    influx[1] = 7.01e-6; // g/cm^3*h
    outflux[1] = 6.98e-6; // g/cm^3*h
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
    weightStates[NC] = 0.43; weightStates[MNC] = 0.19;
    weightStates[SC] = 0.043333333; weightStates[MSC] = 0.019147287;
    weightStates[CSC] = 0.00251938; weightStates[TC] = 0.025;
    weightStates[EMPTY] = 0.29;
    if (initType != 1) {
        memset(weightStates, 0, nStates*dbl);
        weightStates[NC] = 0.70; weightStates[SC] = 0.065;
        weightStates[EMPTY] = 1.0 - weightStates[NC] - weightStates[SC];
    }

    if (initType == 1) {
        rTC = ca->gridSize / 10;
        cX = rand() % gridSize;
        cY = rand() % gridSize;
    }

    ca->init(carcinType, diffusion, influx, outflux, ic, bc, maxTInflux,
             maxTNoInflux, exposureTime, activeCarcin, func, nFunc, funcIdx,
             Wx, Wy, alpha, bias, geneColors,
             params, weightStates, rTC, cX, cY);

    if (initType == 1) {
        init_grid<<< blocks, threads >>>(ca->prevGrid, gridSize,
                                         ca->NN, 100);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        update_states<<< blocks, threads >>>(ca->prevGrid, ca->prevGrid, gridSize,
                                             nGenes, 0, NULL, false);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        cells_gpu_to_gpu_cpy<<< blocks, threads >>>(
            ca->newGrid, ca->prevGrid, gridSize, nGenes
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    carcinNames = (char**)malloc(maxNCarcin*sizeof(char*));
    carcinNames[0] = (char*)calloc(8, 1);
    strcat(carcinNames[0], "Ethanol");
    carcinNames[1] = (char*)calloc(9, 1);
    strcat(carcinNames[1], "Nicotine");

    *gui = GUI(outSize, outSize, ca, display, &ca->perfectExcision,
               gridSize, &ca->maxT, &ca->nCarcin, maxNCarcin,
               ca->activeCarcin, carcinNames);
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
        chdir("..");

        if (gui->windowsShouldClose) {
            retVal = 1;
            break;
        }
        if (k == nSim) continue;

        k++;

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
        gui->earlyStop = false;
        gui->display = display; ca->save = save;
        ca->perfectExcision = perfectExcision; ca->maxTTCAlive = maxTTC;
        ca->nCarcin = nCarcin;
        for (i = 0; i < maxNCarcin; i++) {
            ca->carcins[i].t = 0; ca->carcins[i].nCycles = 1;
            ca->carcins[i].maxTInflux = maxTInflux[i];
            ca->carcins[i].maxTNoInflux = maxTNoInflux[i];
            ca->carcins[i].exposureTime = exposureTime[i];
            ca->activeCarcin[i] = activeCarcin[i];
            ca->carcins[i].init(ca->blockSize, NULL);
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
    } while(k != nSim);

    CudaSafeCall(cudaFree(weightStates));
    cleanup(ca, gui, params);

    system("tput init");

    CudaSafeCall(cudaGetDeviceCount(&nGPU));
    for (i = 0; i < nGPU; i++) {
        CudaSafeCall(cudaSetDevice(i));
        CudaSafeCall(cudaDeviceReset());
    }

    return retVal;
}
