#ifndef __CA_H__
#define __CA_H__

struct CA;
void print_progress(unsigned,unsigned);
void anim_gpu_ca(uchar4*,unsigned,CA*,unsigned,bool,bool*,bool,
                 unsigned*,unsigned*,unsigned*,unsigned,bool*,bool*);
void anim_gpu_lineage(uchar4*,unsigned,CA*,unsigned,
                      unsigned,unsigned,bool,bool);
void anim_gpu_carcin(uchar4*,unsigned,CA*,unsigned,
                     unsigned,bool,bool);
void anim_gpu_cell(uchar4*,unsigned,CA*,
                   unsigned,unsigned,bool);
void anim_gpu_timer_and_saver(CA*,bool,unsigned,bool,bool,bool);

struct CA {
    unsigned devId1, devId2;
    unsigned blockSize;

    Cell *prevGrid, *newGrid;
    Carcin *carcins;
    GeneExprNN *NN;

    unsigned gridSize, cellSize, dim;
    unsigned maxT;
    int maxTTCAlive, excisionTime;

    unsigned nStates, nGenes, nCarcin, maxNCarcin;

    double cellDiameter, cellCycleLen, cellLifeSpan;

    unsigned *cscFormed, *tcFormed, exciseCount, maxExcise, maxNeighDepth,
             timeTCAlive, *timeTCDead, *cellLineage, *nLineage, *nLineageCells,
             *maxLineages, *percentageCounts;
    bool pauseOnFirstTC, perfectExcision, removeField, *activeCarcin, *stateInLineage;
    unsigned *radius, *centerX, *centerY;

    clock_t start, end, startStep, endStep;

    unsigned framerate;
    dim3 *stateColors, *geneColors, *heatmap, *lineageColors;
    bool save, saveCellData;
    char **prefixes, **outNames, *headerCount, **countFiles,
         *headerCellData, *cellData;
    size_t cellDataSize, bytesPerCell;

    CA(unsigned gSize, unsigned T, unsigned ngenes, unsigned ncarcin,
       unsigned maxncarcin, bool saveIn, bool saveCellDataIn, unsigned dim, int maxTTC=-1,
       bool perfectexcision=false, bool removefield=false,
       unsigned maxneighdepth=1, bool pauseonfirsttc=false, int excisiontime=-1)
    {
        blockSize = 16;

        gridSize = gSize; maxT = T;
        nStates = 7; nGenes = ngenes;
        nCarcin = ncarcin; maxNCarcin = maxncarcin;

        cellDiameter = 1.45e-3; // relative to cm, epithelial cell
        cellCycleLen = 10.0; // in hours, for tastebuds
        cellLifeSpan = 250.0; // in hours, for tastebuds

        pauseOnFirstTC = pauseonfirsttc;
        excisionTime = excisiontime;
        exciseCount = 0; maxExcise = 100;
        timeTCAlive = 0; maxTTCAlive = maxTTC;
        perfectExcision = perfectexcision;
        removeField = removefield;
        maxNeighDepth = maxneighdepth;

        cellSize = dim / gridSize;
        save = saveIn; saveCellData = saveCellDataIn;
    }

    void free_resources(void)
    {
        unsigned i, nCells = gridSize * gridSize;
        int nFinished = 0;
        double start;

        if (prevGrid != NULL && newGrid != NULL) {
            start = omp_get_wtime();
            printf("Grid freeing progress:   0.00/100.00");
            #pragma omp parallel for num_threads(4) schedule(guided, 16)\
                    default(shared) private(i)
            for (i = 0; i < nCells; i++) {
                prevGrid[i].free_resources();
                newGrid[i].free_resources();
                #pragma omp atomic
                nFinished++;
                print_progress(nFinished, nCells);
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
                if (i != 0) { free(prefixes[i]); prefixes[i] = NULL; }
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
                 nDigInt = 2, nDigFrac = 10, nCells = gridSize * gridSize;
        size_t unsgn = sizeof(unsigned), chrptr = sizeof(char*),
               dim = sizeof(dim3), bl = sizeof(bool), cell = sizeof(Cell);

        CudaSafeCall(cudaGetDeviceCount(&numDev));
        for (i = numDev-1; i > -1; i--) {
            CudaSafeCall(cudaSetDevice(i));
            for (j = i-1; j > -1; j--)
                CudaSafeCall(cudaDeviceEnablePeerAccess(j, 0));
            CudaSafeCall(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                            692108864));
            CudaSafeCall(cudaDeviceSetLimit(cudaLimitPrintfFifoSize,
                                            11073741824));
        }

        devId1 = 0; devId2 = 0;
        if (numDev == 2) devId2 = 1;

        CudaSafeCall(cudaMallocManaged((void**)&carcins, maxNCarcin*sizeof(Carcin)));
        CudaSafeCall(cudaMallocManaged((void**)&activeCarcin, maxNCarcin*bl));
        memset(activeCarcin, false, maxNCarcin*bl);
        CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(GeneExprNN)));
        CudaSafeCall(cudaMallocManaged((void**)&prevGrid, nCells*cell));
        CudaSafeCall(cudaMallocManaged((void**)&newGrid, nCells*cell));

        CudaSafeCall(cudaMallocManaged((void**)&cscFormed, unsgn));
        *cscFormed = false;
        CudaSafeCall(cudaMallocManaged((void**)&tcFormed, (maxExcise+1)*unsgn));
        memset(tcFormed, 0, (maxExcise+1)*unsgn);
        CudaSafeCall(cudaMallocManaged((void**)&timeTCDead, (maxExcise+1)*unsgn));
        for (i = 0; i < maxExcise+1; i++) timeTCDead[i] = 1;
        CudaSafeCall(cudaMallocManaged((void**)&radius, (maxExcise+1)*unsgn));
        CudaSafeCall(cudaMallocManaged((void**)&centerX, (maxExcise+1)*unsgn));
        CudaSafeCall(cudaMallocManaged((void**)&centerY, (maxExcise+1)*unsgn));
        memset(radius, 0, (maxExcise+1)*unsgn);
        radius[0] = gridSize;
        centerX[0] = gridSize / 2 - 1; centerY[0] = centerX[0];
        CudaSafeCall(cudaMallocManaged((void**)&cellLineage, nCells*unsgn));
        memset(cellLineage, 0, nCells*unsgn);
        CudaSafeCall(cudaMallocManaged((void**)&stateInLineage, (nStates-1)*nCells*bl));
        memset(stateInLineage, false, (nStates-1)*nCells*bl);
        CudaSafeCall(cudaMallocManaged((void**)&nLineage, (nStates+2)*unsgn));
        memset(nLineage, 0, nStates*unsgn);
        CudaSafeCall(cudaMallocManaged((void**)&nLineageCells, unsgn));
        *nLineageCells = 0;
        CudaSafeCall(cudaMallocManaged((void**)&maxLineages, 20*unsgn));
        CudaSafeCall(cudaMallocManaged((void**)&percentageCounts, 10*unsgn));

        CudaSafeCall(cudaMallocManaged((void**)&stateColors, nStates*dim));
        stateColors[   NC] = dim3(146, 111,  98); // brown
        stateColors[  MNC] = dim3( 50, 200, 118); // green
        stateColors[   SC] = dim3(  0,  84, 147); // blue
        stateColors[  MSC] = dim3(240, 200,   0); // yellow
        stateColors[  CSC] = dim3(200,  62, 255); // purple
        stateColors[   TC] = dim3(255,  61,  62); // red
        stateColors[EMPTY] = dim3(255, 255, 255); // white
        CudaSafeCall(cudaMallocManaged((void**)&geneColors, nGenes*dim));
        CudaSafeCall(cudaMallocManaged((void**)&heatmap, 10*dim));
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
        CudaSafeCall(cudaMallocManaged((void**)&lineageColors, 20*dim));
        lineageColors[ 0] = dim3(255, 134, 161);
        lineageColors[ 1] = dim3(194, 165,  28);
        lineageColors[ 2] = dim3( 41, 185, 109);
        lineageColors[ 3] = dim3( 55, 173, 191);
        lineageColors[ 4] = dim3(178, 132, 241);
        lineageColors[ 5] = dim3(241, 102, 158);
        lineageColors[ 6] = dim3(177, 141,  64);
        lineageColors[ 7] = dim3( 67, 160,  77);
        lineageColors[ 8] = dim3( 76, 149, 154);
        lineageColors[ 9] = dim3(134, 121, 217);
        lineageColors[10] = dim3(212,  81, 153);
        lineageColors[11] = dim3(156, 119,  75);
        lineageColors[12] = dim3( 91, 132,  75);
        lineageColors[13] = dim3( 80, 125, 124);
        lineageColors[14] = dim3( 91, 109, 185);
        lineageColors[15] = dim3(161,  83, 133);
        lineageColors[16] = dim3(132,  99,  77);
        lineageColors[17] = dim3( 91, 107,  76);
        lineageColors[18] = dim3( 78, 103, 100);
        lineageColors[19] = dim3( 80,  95, 121);

        prefixes = (char**)malloc((maxNCarcin+10)*chrptr);
        outNames = (char**)malloc((maxNCarcin+10)*chrptr);
        prefixes[0] = NULL;
        prefixes[1] = (char*)calloc(7, 1);
        strcat(prefixes[1], "genes_");
        prefixes[2] = (char*)calloc(9, 1);
        strcat(prefixes[2], "heatmap_");
        for (i = 0; i < nStates; i++) {
            prefixes[i+3] = (char*)calloc(6, 1);
            sprintf(prefixes[i+3], "max%d_", i);
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
            prefixes[i+10] = (char*)calloc(15, 1);
            sprintf(prefixes[i+10], "carcin%d_", i);
            outNames[i+10] = (char*)calloc(25, 1);
            sprintf(outNames[i+10], "out_carcin%d.mp4", i);
        }

        countFiles = (char**)malloc((4*nStates+nStates*nGenes+4+4)*chrptr);
        countFiles[nStates] = (char*)calloc(12, 1);
        strcat(countFiles[nStates], "numTAC.data");
        for (i = 0; i < nStates; i++) {
            countFiles[i] = (char*)calloc(15, 1);
            sprintf(countFiles[i], "numState%d.data", i);
            if (i == EMPTY) continue;
            for (j = 0; j < 3; j++) {
                countFiles[i*3+(nGenes+15)+j] = (char*)calloc(22, 1);
                sprintf(countFiles[i*3+(nGenes+15)+j],
                        "numPheno%d_State%d.data", j, i);
            }
            for (j = 0; j < nGenes; j++) {
                countFiles[i*nGenes+(3*nStates+nGenes+15)+j] = (char*)calloc(nDigGene+20, 1);
                sprintf(countFiles[i*nGenes+(3*nStates+nGenes+15)+j],
                        "numGene%d_State%d.data", j, i);
            }
            if (i == SC || i == MSC || i == CSC) continue;
            countFiles[nStates+(i == TC ? 3 : i+1)] = (char*)calloc(19, 1);
            sprintf(countFiles[nStates+(i == TC ? 3 : i+1)],
                    "numTAC_State%d.data", i);
        }
        j = SC;
        for (i = 0; i < 3; i++) {
            countFiles[i+(3*nStates+nGenes+12)] = (char*)calloc(22, 1);
            sprintf(countFiles[i+(3*nStates+nGenes+12)],
                    "numPheno%d_State%d.data", 3, j++);
        }
        for (i = 0; i < 4; i++) {
            countFiles[i+nStates+4] = (char*)calloc(15, 1);
            sprintf(countFiles[i+nStates+4], "numPheno%d.data", i);
        }
        for (i = 0; i < nGenes; i++) {
            countFiles[i+15] = (char*)calloc(nDigGene+13, 1);
            sprintf(countFiles[i+15], "numGene%d.data", i);
        }
        headerCount = (char*)calloc(53, 1);
        strcat(headerCount, "# t\tcount\tcolour (2^16*red+2^8*blue+green)\n");

        bytesPerCell = 3 * num_digits(nCells)
                     + num_digits(maxT + cellLifeSpan / cellCycleLen)
                     + (nDigFrac + 2) * nPheno
                     + (nDigInt + nDigFrac + 3) * nGenes
                     + (6 + nDigFrac + 2) // fitness
                     + 25;
        cellDataSize = bytesPerCell * nCells + 1;
        CudaSafeCall(cudaMallocManaged((void**)&cellData, cellDataSize));
        cellData[cellDataSize-1] = '\0';
        headerCellData = (char*)calloc(128, 1);
        strcat(headerCellData, "# idx,state,age,prolif,quies,apop,diff");
        strcat(headerCellData, ",[geneExprs],fitness,chosenPheno,chosenCell");
        strcat(headerCellData, ",actionDone,excised,lineage,isTAC,nTACProlif\n");
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
        int nt = 4;
        unsigned i, k, nFinished = 0, dev, nCells = gridSize * gridSize;
        size_t unsgn = sizeof(unsigned), dim = sizeof(dim3), bl = sizeof(bool),
               cell = sizeof(Cell);
        dim3 blocks(NBLOCKS(gridSize, blockSize), NBLOCKS(gridSize, blockSize));
        dim3 threads(blockSize, blockSize);
        cudaStream_t prefetch[maxNCarcin+nt+5], kernel[maxNCarcin+1];

        for (i = 0; i < nGenes; i++) {
            geneColors[i].x = genecolors[i].x;
            geneColors[i].y = genecolors[i].y;
            geneColors[i].z = genecolors[i].z;
        }

        for (i = 0; i < maxNCarcin+nt+5; i++) {
            CudaSafeCall(cudaStreamCreate(&prefetch[i]));
            if (i < maxNCarcin+1) CudaSafeCall(cudaStreamCreate(&kernel[i]));
        }

        set_seed();

        printf("Grid initialization progress:   0.00/100.00");
        #pragma omp parallel for num_threads(nt) schedule(guided, 16)\
                default(shared) private(i)
        for (i = 0; i < nCells; i++) {
            init_grid_cell(prevGrid, i / gridSize, i % gridSize,
                           devId1, params, weightStates, rTC, cX,
                           cY, &prefetch[omp_get_thread_num()]);
            init_grid_cell(newGrid, i / gridSize, i % gridSize,
                           devId2, params, weightStates, rTC, cX,
                           cY, &prefetch[omp_get_thread_num()]);
            #pragma omp atomic
            nFinished++;
            print_progress(nFinished, nCells);
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
                                maxTInflux[k], maxTNoInflux[k], cellDiameter,
                                cellCycleLen, exposureTime[k], func, nFunc,
                                funcIdx[k]);
            carcins[k].prefetch_memory(dev, &prefetch[k+nt+5]);
            carcins[k].init(blockSize, &kernel[k+1]);
            if (activecarcin[k]) activeCarcin[k] = true;
        }

        *NN = GeneExprNN(devId2, maxNCarcin+1, nGenes, alpha, bias);
        NN->memory_allocate(Wx, Wy);
        NN->prefetch_memory(devId2, &prefetch[nt]);

        params->prefetch_memory(devId2, nStates, nGenes, &prefetch[nt+1]);
        CudaSafeCall(cudaMemPrefetchAsync(params, sizeof(CellParams),
                                          devId2, prefetch[nt+2]));

        CudaSafeCall(cudaMemPrefetchAsync(prevGrid, nCells*cell,
                                          devId1, prefetch[nt+3]));
        CudaSafeCall(cudaMemPrefetchAsync(newGrid, nCells*cell,
                                          devId2, prefetch[nt+4]));
        CudaSafeCall(cudaMemPrefetchAsync(cellLineage, nCells*unsgn,
                                          devId2, prefetch[nt]));
        CudaSafeCall(cudaMemPrefetchAsync(stateInLineage, (nStates-1)*nCells*bl,
                                          devId2, prefetch[nt+1]));
        CudaSafeCall(cudaMemPrefetchAsync(maxLineages, 20*unsgn,
                                          devId2, prefetch[nt+2]));
        CudaSafeCall(cudaMemPrefetchAsync(percentageCounts, 10*unsgn,
                                          devId2, prefetch[nt+3]));
        CudaSafeCall(cudaMemPrefetchAsync(carcins, maxNCarcin*sizeof(Carcin),
                                          devId2, prefetch[nt+4]));
        CudaSafeCall(cudaMemPrefetchAsync(activeCarcin, maxNCarcin*bl,
                                          devId2, prefetch[nt]));

        CudaSafeCall(cudaMemPrefetchAsync(NN, sizeof(GeneExprNN),
                                          devId2, prefetch[nt+1]));
        CudaSafeCall(cudaMemPrefetchAsync(tcFormed, (maxExcise+1)*unsgn,
                                          devId1, prefetch[nt+2]));
        CudaSafeCall(cudaMemPrefetchAsync(timeTCDead, (maxExcise+1)*unsgn,
                                          devId1, prefetch[nt+3]));
        CudaSafeCall(cudaMemPrefetchAsync(radius, (maxExcise+1)*unsgn,
                                          devId1, prefetch[nt+4]));
        CudaSafeCall(cudaMemPrefetchAsync(centerX, (maxExcise+1)*unsgn,
                                          devId1, prefetch[nt]));
        CudaSafeCall(cudaMemPrefetchAsync(centerY, (maxExcise+1)*unsgn,
                                          devId1, prefetch[nt+1]));
        CudaSafeCall(cudaMemPrefetchAsync(stateColors, nStates*dim,
                                          devId1, prefetch[nt+2]));
        CudaSafeCall(cudaMemPrefetchAsync(geneColors, nGenes*dim,
                                          devId1, prefetch[nt+3]));
        CudaSafeCall(cudaMemPrefetchAsync(heatmap, 10*dim,
                                          devId1, prefetch[nt+4]));
        CudaSafeCall(cudaMemPrefetchAsync(lineageColors, 20*dim,
                                          devId1, prefetch[nt]));
        CudaSafeCall(cudaMemPrefetchAsync(cellData, cellDataSize,
                                          devId2, prefetch[nt+1]));

        for (i = 0; i < maxNCarcin+nt+5; i++) {
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
        gui->anim((void (*)(uchar4*,unsigned,void*,unsigned,bool,bool*,bool,
                            unsigned*,unsigned*,unsigned*,unsigned,bool*,bool*))anim_gpu_ca,
                  (void (*)(uchar4*,unsigned,void*,unsigned,
                            unsigned,unsigned,bool,bool))anim_gpu_lineage,
                  (void (*)(uchar4*,unsigned,void*,unsigned,
                            unsigned,bool,bool))anim_gpu_carcin,
                  (void (*)(uchar4*,unsigned,void*,
                            unsigned,unsigned,bool))anim_gpu_cell,
                  (void (*)(void*,bool,unsigned,
                            bool,bool,bool))anim_gpu_timer_and_saver);
    }
};

#endif // __CA_H__
