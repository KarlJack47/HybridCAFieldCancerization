#ifndef __CA_H__
#define __CA_H__

struct CA;
void anim_gpu_ca(uchar4*,unsigned,CA*,unsigned,bool,bool,bool,
                 unsigned,unsigned,unsigned,bool*,bool*);
void anim_gpu_lineage(uchar4*,unsigned,CA*,unsigned,
                      unsigned,unsigned,bool,bool);
void anim_gpu_carcin(uchar4*,unsigned,CA*,unsigned,
                     unsigned,bool,bool);
void anim_gpu_cell(uchar4*,unsigned,CA*,
                   unsigned,unsigned,bool);
void anim_gpu_timer_and_saver(CA*,bool,unsigned,bool,bool);

struct CA {
    unsigned devId1, devId2;
    unsigned blockSize;

    Cell *prevGrid, *newGrid;
    Carcin *carcins;
    GeneExprNN *NN;

    unsigned gridSize, cellSize, dim;
    unsigned maxT;
    int maxTTCAlive;

    unsigned nStates, nGenes, nCarcin, maxNCarcin;

    double cellVolume, cellCycleLen, cellLifeSpan;

    unsigned *cscFormed, *tcFormed, exciseCount, maxExcise,
             timeTCAlive, *timeTCDead, *cellLineage, *nLineage, *nLineageCells,
             *maxLineages, *percentageCounts;
    bool perfectExcision, *activeCarcin, *stateInLineage;
    unsigned *radius, *centerX, *centerY;

    clock_t start, end, startStep, endStep;

    unsigned framerate;
    dim3 *stateColors, *geneColors, *heatmap, *lineageColors;
    bool save;
    char **prefixes, **outNames, *headerCount, **countFiles,
         *headerCellData, *cellData;
    size_t cellDataSize, bytesPerCell;

    CA(unsigned gSize, unsigned T, unsigned ngenes, unsigned ncarcin,
       unsigned maxncarcin, bool saveIn, unsigned dim, int maxTTC=-1,
       bool perfectexcision=false)
    {
        blockSize = 16;

        gridSize = gSize; maxT = T;
        nStates = 7; nGenes = ngenes;
        nCarcin = ncarcin; maxNCarcin = maxncarcin;

        cellVolume = 1.596e-9; // relative to cm, epithelial cell
        cellCycleLen = 10.0; // in hours, for tastebuds
        cellLifeSpan = 250.0; // in hours, for tastebuds

        exciseCount = 0; maxExcise = 100;
        timeTCAlive = 0; maxTTCAlive = maxTTC;
        perfectExcision = perfectexcision;

        cellSize = dim / gridSize; save = saveIn;
    }

    void free_resources(void)
    {
        unsigned i, j, k;
        int nt = omp_get_num_procs(), counts[nt] = { 0 }, numFinished = 0;
        double start;

        if (prevGrid != NULL && newGrid != NULL) {
            start = omp_get_wtime();
            printf("Grid freeing progress:   0.00/100.00");
            for (i = 0; i < gridSize; i++)
                for (j = 0; j < gridSize; j+=2) {
                    #pragma omp parallel sections num_threads(2)
                    {
                        #pragma omp section
                        {
                            prevGrid[i*gridSize+j].free_resources();
                            newGrid[i*gridSize+j].free_resources();

                            counts[omp_get_thread_num()]++;
                        }

                        #pragma omp section
                        {
                            prevGrid[i*gridSize+(j+1)].free_resources();
                            newGrid[i*gridSize+(j+1)].free_resources();

                            counts[omp_get_thread_num()]++;
                        }
                    }

                    numFinished = 0;
                    for (k = 0; k < nt; k++) numFinished += counts[k];
                    print_progress(numFinished, gridSize*gridSize);
                }
            printf("\n");
            fflush(stdout);
            printf("It took %f seconds to finish freeing the memory.\n",
                   omp_get_wtime() - start);
        }
        if (prevGrid != NULL) {
            CudaSafeCall(cudaFree(prevGrid)); prevGrid = NULL;
        }
        if (newGrid != NULL) {
            CudaSafeCall(cudaFree(newGrid)); newGrid = NULL;
        }

        if (carcins != NULL) {
            for (i = 0; i < maxNCarcin; i++)
                carcins[i].free_resources();
            CudaSafeCall(cudaFree(carcins)); carcins = NULL;
        }
        if (activeCarcin != NULL) {
            CudaSafeCall(cudaFree(activeCarcin)); activeCarcin = NULL;
        }
        if (NN != NULL) {
            NN->free_resources();
            CudaSafeCall(cudaFree(NN)); NN = NULL;
        }

        if (cscFormed != NULL) {
            CudaSafeCall(cudaFree(cscFormed)); cscFormed = NULL;
        }
        if (tcFormed != NULL) {
            CudaSafeCall(cudaFree(tcFormed)); tcFormed = NULL;
        }
        if (timeTCDead != NULL) {
            CudaSafeCall(cudaFree(timeTCDead)); timeTCDead = NULL;
        }
        if (radius != NULL) {
            CudaSafeCall(cudaFree(radius)); radius = NULL;
        }
        if (centerX != NULL) {
            CudaSafeCall(cudaFree(centerX)); centerX = NULL;
        }
        if (centerY != NULL) {
            CudaSafeCall(cudaFree(centerY)); centerY = NULL;
        }
        if (cellLineage != NULL) {
            CudaSafeCall(cudaFree(cellLineage)); cellLineage = NULL;
        }
        if (stateInLineage != NULL) {
            CudaSafeCall(cudaFree(stateInLineage)); stateInLineage = NULL;
        }
        if (nLineage != NULL) {
            CudaSafeCall(cudaFree(nLineage)); nLineage = NULL;
        }
        if (nLineageCells != NULL) {
            CudaSafeCall(cudaFree(nLineageCells)); nLineageCells = NULL;
        }
        if (maxLineages != NULL) {
            CudaSafeCall(cudaFree(maxLineages)); maxLineages = NULL;
        }
        if (percentageCounts != NULL) {
            CudaSafeCall(cudaFree(percentageCounts)); percentageCounts = NULL;
        }
        if (outNames != NULL && prefixes != NULL) {
            for (i = 0; i < maxNCarcin+10; i++) {
                free(outNames[i]); outNames[i] = NULL;
                if (i < maxNCarcin+9) { free(prefixes[i]); prefixes[i] = NULL; }
            }
            free(outNames); outNames = NULL;
            free(prefixes); prefixes = NULL;
        }
        if (countFiles != NULL) {
            for (i = 0; i < 4*nStates+nStates*nGenes+4; i++) {
                free(countFiles[i]); countFiles[i] = NULL;
            }
            free(countFiles); countFiles = NULL;
        }
        if (headerCount != NULL) {
            free(headerCount); headerCount = NULL;
        }
        if (cellData != NULL) {
            CudaSafeCall(cudaFree(cellData)); cellData = NULL;
        }
        if (headerCellData != NULL) {
            free(headerCellData); headerCellData = NULL;
        }
        if (stateColors != NULL) {
            CudaSafeCall(cudaFree(stateColors)); stateColors = NULL;
        }
        if (geneColors != NULL) {
            CudaSafeCall(cudaFree(geneColors)); geneColors = NULL;
        }
        if (heatmap != NULL) {
            CudaSafeCall(cudaFree(heatmap)); heatmap = NULL;
        }
        if (lineageColors != NULL) {
            CudaSafeCall(cudaFree(lineageColors)); lineageColors = NULL;
        }
    }

    void initialize_memory(void)
    {
        int i, j, numDev;
        unsigned nPheno = 4, nDigGene = num_digits(nGenes),
                 nDigInt = 2, nDigFrac = 10;

        CudaSafeCall(cudaGetDeviceCount(&numDev));
        for (i = numDev-1; i > -1; i--) {
            CudaSafeCall(cudaSetDevice(i));
            for (j = i-1; j > -1; j--)
                CudaSafeCall(cudaDeviceEnablePeerAccess(j, 0));
            CudaSafeCall(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 33554432));
            CudaSafeCall(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 50331648));
        }

        devId1 = 0; devId2 = 0;
        if (numDev == 2) devId2 = 1;

        CudaSafeCall(cudaMallocManaged((void**)&carcins,
                                       maxNCarcin*sizeof(Carcin)));
        CudaSafeCall(cudaMallocManaged((void**)&activeCarcin,
                                       maxNCarcin*sizeof(bool)));
        memset(activeCarcin, false, maxNCarcin*sizeof(bool));
        CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(GeneExprNN)));
        CudaSafeCall(cudaMallocManaged((void**)&prevGrid,
                                       gridSize*gridSize*sizeof(Cell)));
        CudaSafeCall(cudaMallocManaged((void**)&newGrid,
                                       gridSize*gridSize*sizeof(Cell)));

        CudaSafeCall(cudaMallocManaged((void**)&cscFormed, sizeof(unsigned)));
        *cscFormed = false;
        CudaSafeCall(cudaMallocManaged((void**)&tcFormed,
                                       (maxExcise+1)*sizeof(unsigned)));
        memset(tcFormed, 0, (maxExcise+1)*sizeof(unsigned));
        CudaSafeCall(cudaMallocManaged((void**)&timeTCDead,
                                       (maxExcise+1)*sizeof(unsigned)));
        for (i = 0; i < maxExcise+1; i++) timeTCDead[i] = 1;
        CudaSafeCall(cudaMallocManaged((void**)&radius,
                                       (maxExcise+1)*sizeof(unsigned)));
        CudaSafeCall(cudaMallocManaged((void**)&centerX,
                                       (maxExcise+1)*sizeof(unsigned)));
        CudaSafeCall(cudaMallocManaged((void**)&centerY,
                                       (maxExcise+1)*sizeof(unsigned)));
        memset(radius, 0, (maxExcise+1)*sizeof(unsigned));
        radius[0] = gridSize;
        centerX[0] = gridSize / 2 - 1; centerY[0] = centerX[0];
        CudaSafeCall(cudaMallocManaged((void**)&cellLineage,
                                       gridSize*gridSize*sizeof(unsigned)));
        memset(cellLineage, 0, gridSize*gridSize*sizeof(unsigned));
        CudaSafeCall(cudaMallocManaged((void**)&stateInLineage,
                                       (nStates-1)*gridSize*gridSize*sizeof(bool)));
        memset(stateInLineage, false, (nStates-1)*gridSize*gridSize*sizeof(bool));
        CudaSafeCall(cudaMallocManaged((void**)&nLineage, sizeof(unsigned)));
        *nLineage = 0;
        CudaSafeCall(cudaMallocManaged((void**)&nLineageCells, sizeof(unsigned)));
        *nLineageCells = 0;
        CudaSafeCall(cudaMallocManaged((void**)&maxLineages, 10*sizeof(unsigned)));
        CudaSafeCall(cudaMallocManaged((void**)&percentageCounts, 10*sizeof(unsigned)));

        CudaSafeCall(cudaMallocManaged((void**)&stateColors, nStates*sizeof(dim3)));
        stateColors[   NC] = dim3(146, 111,  98); // brown
        stateColors[  MNC] = dim3( 50, 200, 118); // green
        stateColors[   SC] = dim3(  0,  84, 147); // blue
        stateColors[  MSC] = dim3(240, 200,   0); // yellow
        stateColors[  CSC] = dim3(200,  62, 255); // purple
        stateColors[   TC] = dim3(255,  61,  62); // red
        stateColors[EMPTY] = dim3(255, 255, 255); // white
        CudaSafeCall(cudaMallocManaged((void**)&geneColors, nGenes*sizeof(dim3)));
        CudaSafeCall(cudaMallocManaged((void**)&heatmap, 10*sizeof(dim3)));
        heatmap[0] = dim3(255,   0,   0); // 0-10%
        heatmap[1] = dim3(255, 104, 104); // 11-20%
        heatmap[2] = dim3(255, 153, 153); // 21-30%
        heatmap[3] = dim3(255, 195, 195); // 31-40%
        heatmap[4] = dim3(255, 234, 234); // 41-50%
        heatmap[5] = dim3(231, 255, 231); // 51-60%
        heatmap[6] = dim3(214, 255, 214); // 61-70%
        heatmap[7] = dim3(187, 255, 187); // 71-80%
        heatmap[8] = dim3(145, 255, 145); // 81-90%
        heatmap[9] = dim3(  0, 255,   0); // 91-100%
        CudaSafeCall(cudaMallocManaged((void**)&lineageColors, 10*sizeof(dim3)));
        lineageColors[0] = dim3(221, 219,   0);
        lineageColors[1] = dim3(107, 221,   0);
        lineageColors[2] = dim3(  0, 203, 152);
        lineageColors[3] = dim3(  0, 181, 179);
        lineageColors[4] = dim3(  0, 159, 197);
        lineageColors[5] = dim3( 75, 121, 255);
        lineageColors[6] = dim3(193,   0, 230);
        lineageColors[7] = dim3(188,   0, 143);
        lineageColors[8] = dim3(172,   0,  55);
        lineageColors[9] = dim3(105,  60,   0);

        prefixes = (char**)malloc((maxNCarcin+9)*sizeof(char*));
        outNames = (char**)malloc((maxNCarcin+10)*sizeof(char*));
        prefixes[0] = (char*)calloc(7, 1);
        strcat(prefixes[0], "genes_");
        prefixes[1] = (char*)calloc(9, 1);
        strcat(prefixes[1], "heatmap_");
        for (i = 0; i < nStates; i++) {
            prefixes[i+2] = (char*)calloc(6, 1);
            sprintf(prefixes[i+2], "max%d_", i);
        }
        outNames[0] = (char*)calloc(11, 1);
        strcat(outNames[0], "out_ca.mp4");
        outNames[1] = (char*)calloc(14, 1);
        strcat(outNames[1], "out_genes.mp4");
        outNames[2] = (char*)calloc(16, 1);
        strcat(outNames[2], "out_heatmap.mp4");
        for (i = 0; i < nStates; i++) {
            outNames[i+3] = (char*)calloc(13, 1);
            sprintf(outNames[i+3], "out_max%d.mp4", i);
        }
        for (i = 0; i < maxNCarcin; i++) {
            prefixes[i+9] = (char*)calloc(15, 1);
            sprintf(prefixes[i+9], "carcin%d_", i);
            outNames[i+10] = (char*)calloc(25, 1);
            sprintf(outNames[i+10], "out_carcin%d.mp4", i);
        }

        countFiles = (char**)malloc((4*nStates+nStates*nGenes+4)*sizeof(char*));
        for (i = 0; i < nStates; i++) {
            countFiles[i] = (char*)calloc(15, 1);
            sprintf(countFiles[i], "numState%d.data", i);
            if (i == EMPTY) continue;
            for (j = 0; j < 3; j++) {
                countFiles[i*3+(nGenes+11)+j] = (char*)calloc(22, 1);
                sprintf(countFiles[i*3+(nGenes+11)+j],
                        "numPheno%d_State%d.data", j, i);
            }
            for (j = 0; j < nGenes; j++) {
                countFiles[i*nGenes+(3*nStates+nGenes+11)+j] = (char*)calloc(nDigGene+20, 1);
                sprintf(countFiles[i*nGenes+(3*nStates+nGenes+11)+j],
                        "numGene%d_State%d.data", j, i);
            }
        }
        j = SC;
        for (i = 0; i < 3; i++) {
            countFiles[i+(3*nStates+nGenes+8)] = (char*)calloc(22, 1);
            sprintf(countFiles[i+(3*nStates+nGenes+8)],
                    "numPheno%d_State%d.data", 3, j++);
        }
        for (i = 0; i < 4; i++) {
            countFiles[i+7] = (char*)calloc(15, 1);
            sprintf(countFiles[i+7], "numPheno%d.data", i);
        }
        for (i = 0; i < nGenes; i++) {
            countFiles[i+11] = (char*)calloc(nDigGene+13, 1);
            sprintf(countFiles[i+11], "numGene%d.data", i);
        }
        headerCount = (char*)calloc(53, 1);
        strcat(headerCount, "# t\tcount\tcolour (2^16*red+2^8*blue+green)\n");

        bytesPerCell = 3 * num_digits(gridSize * gridSize)
                     + num_digits(maxT + cellLifeSpan / cellCycleLen)
                     + (nDigFrac + 2) * nPheno
                     + (2 * (nDigInt + nDigFrac) + 4) * nGenes + 20;
        cellDataSize = bytesPerCell * gridSize * gridSize + 1;
        CudaSafeCall(cudaMallocManaged((void**)&cellData, cellDataSize));
        cellData[cellDataSize-1] = '\0';
        headerCellData = (char*)calloc(93, 1);
        strcat(headerCellData, "# idx,state,age,prolif,quies,apop,diff");
        strcat(headerCellData, ",[geneExprs],chosenPheno,chosenCell");
        strcat(headerCellData, ",actionDone,excised,lineage\n");
	}

    void init_grid_cell(Cell *G, unsigned i, unsigned j, int dev,
                        CellParams *params, double *weightStates,
                        unsigned rTC, unsigned cX, unsigned cY,
                        cudaStream_t *stream)
    {
        G[i*gridSize+j] = Cell(dev, params, i, j, gridSize, nGenes,
                               cellCycleLen, cellLifeSpan, weightStates,
                               rTC, cX, cY);
        G[i*gridSize+j].prefetch_memory(dev, gridSize, nGenes, stream);
    }

    void init(unsigned *carcinType, double *diffusion, double *influx,
              double *outflux, double *ic, double *bc, int *maxTInflux,
              int *maxTNoInflux, double *exposureTime, bool *activecarcin,
              SensitivityFunc *func, unsigned nFunc, unsigned *funcIdx,
              double *Wx, double *Wy, double alpha, double bias,
              dim3 *genecolors, CellParams *params, double *weightStates=NULL,
              unsigned rTC=0, unsigned cX=0, unsigned cY=0)
    {
        unsigned i, j, k, numFinished, dev;
        int nt = omp_get_num_procs(), counts[nt] = { 0 };
        dim3 blocks(NBLOCKS(gridSize, blockSize), NBLOCKS(gridSize, blockSize));
        dim3 threads(blockSize, blockSize);
        cudaStream_t prefetch[maxNCarcin+5]; cudaStream_t kernel[maxNCarcin+1];

        for (i = 0; i < nGenes; i++) {
            geneColors[i].x = genecolors[i].x;
            geneColors[i].y = genecolors[i].y;
            geneColors[i].z = genecolors[i].z;
        }

        for (i = 0; i < maxNCarcin+5; i++) {
            CudaSafeCall(cudaStreamCreate(&prefetch[i]));
            if (i < maxNCarcin+1) CudaSafeCall(cudaStreamCreate(&kernel[i]));
        }

        printf("Grid initialization progress:   0.00/100.00");
        for (i = 0; i < gridSize; i++)
            for (j = 0; j < gridSize; j+=2) {
                #pragma omp parallel sections num_threads(2)
                {
                    #pragma omp section
                    {
                        init_grid_cell(prevGrid, i, j, devId1, params,
                                       weightStates, rTC, cX, cY, &prefetch[0]);
                        init_grid_cell(newGrid, i, j, devId2, params,
                                       weightStates, rTC, cX, cY, &prefetch[1]);
                        counts[omp_get_thread_num()]++;
                    }

                    #pragma omp section
                    {
                        init_grid_cell(prevGrid, i, j+1, devId1, params,
                                       weightStates, rTC, cX, cY, &prefetch[2]);
                        init_grid_cell(newGrid, i, j+1, devId2, params,
                                       weightStates, rTC, cX, cY, &prefetch[3]);
                        counts[omp_get_thread_num()]++;
                    }
                }

                numFinished = 0;
                for (k = 0; k < nt; k++) numFinished += counts[k];
                print_progress(numFinished, gridSize*gridSize);
            }
        printf("\n");
        fflush(stdout);

        cells_gpu_to_gpu_cpy<<< blocks, threads, 0, kernel[0] >>>(
            newGrid, prevGrid, gridSize, nGenes
        );
        CudaCheckError();

        for (k = 0; k < maxNCarcin; k++) {
            dev = k % 2 == 1 ? devId1 : devId2;
            carcins[k] = Carcin(dev, k, carcinType[k], gridSize, diffusion[k],
                                influx[k], outflux[k], ic[k], bc[k],
                                maxTInflux[k], maxTNoInflux[k], cellVolume,
                                cellCycleLen, exposureTime[k], func, nFunc,
                                funcIdx[k]);
            carcins[k].prefetch_memory(dev, &prefetch[k+5]);
            carcins[k].init(blockSize, &kernel[k+1]);
            if (activecarcin[k]) activeCarcin[k] = true;
        }

        *NN = GeneExprNN(devId2, maxNCarcin+1, nGenes, alpha, bias);
        NN->memory_allocate(Wx, Wy);
        NN->prefetch_memory(devId2, &prefetch[4]);

        params->prefetch_memory(devId2, nStates, nGenes, &prefetch[0]);
        CudaSafeCall(cudaMemPrefetchAsync(params, sizeof(CellParams),
                                          devId2, prefetch[1]));

        CudaSafeCall(cudaMemPrefetchAsync(prevGrid,
                                          gridSize*gridSize*sizeof(Cell),
                                          devId1, prefetch[2]));
        CudaSafeCall(cudaMemPrefetchAsync(newGrid,
                                          gridSize*gridSize*sizeof(Cell),
                                          devId2, prefetch[3]));
        CudaSafeCall(cudaMemPrefetchAsync(cellLineage,
                                          gridSize*gridSize*sizeof(unsigned),
                                          devId2, prefetch[4]));
        CudaSafeCall(cudaMemPrefetchAsync(stateInLineage,
                                          (nStates-1)*gridSize*gridSize*sizeof(bool),
                                          devId2, prefetch[0]));
        CudaSafeCall(cudaMemPrefetchAsync(maxLineages,
                                          10*sizeof(unsigned),
                                          devId2, prefetch[1]));
        CudaSafeCall(cudaMemPrefetchAsync(percentageCounts,
                                          10*sizeof(unsigned),
                                          devId2, prefetch[2]));
        CudaSafeCall(cudaMemPrefetchAsync(carcins,
                                          maxNCarcin*sizeof(Carcin),
                                          devId2, prefetch[3]));
        CudaSafeCall(cudaMemPrefetchAsync(activeCarcin,
                                          maxNCarcin*sizeof(bool),
                                          devId2, prefetch[4]));

        CudaSafeCall(cudaMemPrefetchAsync(NN,
                                          sizeof(GeneExprNN),
                                          devId2, prefetch[0]));
        CudaSafeCall(cudaMemPrefetchAsync(tcFormed,
                                          (maxExcise+1)*sizeof(unsigned),
                                          devId1, prefetch[1]));
        CudaSafeCall(cudaMemPrefetchAsync(timeTCDead,
                                          (maxExcise+1)*sizeof(unsigned),
                                          devId1, prefetch[2]));
        CudaSafeCall(cudaMemPrefetchAsync(radius,
                                          (maxExcise+1)*sizeof(unsigned),
                                          devId1, prefetch[3]));
        CudaSafeCall(cudaMemPrefetchAsync(centerX,
                                          (maxExcise+1)*sizeof(unsigned),
                                          devId1, prefetch[4]));
        CudaSafeCall(cudaMemPrefetchAsync(centerY,
                                          (maxExcise+1)*sizeof(unsigned),
                                          devId1, prefetch[0]));
        CudaSafeCall(cudaMemPrefetchAsync(stateColors,
                                          nStates*sizeof(dim3),
                                          devId1, prefetch[1]));
        CudaSafeCall(cudaMemPrefetchAsync(geneColors,
                                          nGenes*sizeof(dim3),
                                          devId1, prefetch[2]));
        CudaSafeCall(cudaMemPrefetchAsync(heatmap,
                                          10*sizeof(dim3),
                                          devId1, prefetch[3]));
        CudaSafeCall(cudaMemPrefetchAsync(lineageColors,
                                          10*sizeof(dim3),
                                          devId1, prefetch[4]));
        CudaSafeCall(cudaMemPrefetchAsync(cellData,
                                          cellDataSize,
                                          devId2, prefetch[0]));

        for (i = 0; i < maxNCarcin+5; i++) {
            CudaSafeCall(cudaStreamSynchronize(prefetch[i]));
            CudaSafeCall(cudaStreamDestroy(prefetch[i]));
            if (i < maxNCarcin+1) {
                CudaSafeCall(cudaStreamSynchronize(kernel[i]));
                CudaSafeCall(cudaStreamDestroy(kernel[i]));
            }
        }
    }

    void animate(unsigned frameRate, GUI *gui)
    {
        framerate = frameRate;
        gui->anim((void (*)(uchar4*,unsigned,void*,unsigned,bool,bool,bool,
                            unsigned,unsigned,unsigned,bool*,bool*))anim_gpu_ca,
                  (void (*)(uchar4*,unsigned,void*,unsigned,
                            unsigned,unsigned,bool,bool))anim_gpu_lineage,
                  (void (*)(uchar4*,unsigned,void*,unsigned,
                            unsigned,bool,bool))anim_gpu_carcin,
                  (void (*)(uchar4*,unsigned,void*,
                            unsigned,unsigned,bool))anim_gpu_cell,
                  (void (*)(void*,bool,unsigned,
                            bool,bool))anim_gpu_timer_and_saver);
    }
};

#endif // __CA_H__