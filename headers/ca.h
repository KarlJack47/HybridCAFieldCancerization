#ifndef __CA_H__
#define __CA_H__

struct CA;
void anim_gpu_ca(uchar4*,unsigned,CA*,unsigned,bool,bool,bool,
                 unsigned,unsigned,unsigned,bool*,bool*);
void anim_gpu_genes(uchar4*,unsigned,CA*,
                    unsigned,bool,bool);
void anim_gpu_carcin(uchar4*,unsigned,CA*,unsigned,
                     unsigned,bool,bool);
void anim_gpu_cell(uchar4*,unsigned,CA*,
                   unsigned,unsigned,bool);
void anim_gpu_timer_and_saver(CA*,bool,unsigned,bool,bool);

struct CA {
    unsigned devId1, devId2;
    unsigned blockSize;

    Cell *prevGrid, *newGrid;
    CarcinPDE *pdes;
    GeneExprNN *NN;

    unsigned gridSize, cellSize, dim;
    unsigned maxT;
    int maxTTCAlive;

    unsigned nGenes, nCarcin, nStates;

    double cellVolume, cellCycleLen, cellLifeSpan;

    unsigned *cscFormed, *tcFormed, exciseCount, maxExcise,
             timeTCAlive, *timeTCDead;
    bool perfectExcision;
    unsigned *radius, *centerX, *centerY;

    clock_t start, end, startStep, endStep;

    unsigned framerate;
    dim3 *stateColors, *geneColors;
    bool save;
    char **prefixes, **outNames, *headerCount, **countFiles,
         *headerCellData, *cellData;
    size_t cellDataSize, bytesPerCell;

    CA(unsigned gSize, unsigned T, unsigned ngenes, unsigned ncarcin,
       bool saveIn, unsigned dim, int maxTTC=-1, bool perfectexcision=false)
    {
        blockSize = 16;

        gridSize = gSize; maxT = T;
        nStates = 7; nGenes = ngenes; nCarcin = ncarcin;

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
            printf("It took %f seconds to finish freeing the memory.\n",
                   omp_get_wtime() - start);
        }
        if (prevGrid != NULL) {
            CudaSafeCall(cudaFree(prevGrid)); prevGrid = NULL;
        }
        if (newGrid != NULL) {
            CudaSafeCall(cudaFree(newGrid)); newGrid = NULL;
        }

        if (pdes != NULL) {
            for (i = 0; i < nCarcin; i++)
                pdes[i].free_resources();
            CudaSafeCall(cudaFree(pdes)); pdes = NULL;
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
            CudaSafeCall(cudaFree(timeTCDead));
        }
        if (radius != NULL) {
            CudaSafeCall(cudaFree(radius));
        }
        if (centerX != NULL) {
            CudaSafeCall(cudaFree(centerX));
        }
        if (centerY != NULL) {
            CudaSafeCall(cudaFree(centerY));
        }
        if (outNames != NULL && prefixes != NULL) {
            for (i = 0; i < nCarcin+2; i++) {
                free(outNames[i]); outNames[i] = NULL;
                if (i < nCarcin+1) { free(prefixes[i]); prefixes[i] = NULL; }
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
    }

    void initialize_memory(void)
    {
        int i, j, numDev;
        unsigned nPheno = 4, nDigGene = num_digits(nGenes),
                 nDigInt = 1, nDigFrac = 10;

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

        if (nCarcin != 0)
            CudaSafeCall(cudaMallocManaged((void**)&pdes,
                                           nCarcin*sizeof(CarcinPDE)));
        else
            pdes = NULL;
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

        CudaSafeCall(cudaMallocManaged((void**)&stateColors, nStates*sizeof(dim3)));
        stateColors[   NC] = dim3(  0,   0,   0); // black
        stateColors[  MNC] = dim3( 87, 207,   0); // green
        stateColors[   SC] = dim3(244, 131,   0); // orange
        stateColors[  MSC] = dim3(  0,   0, 255); // blue
        stateColors[  CSC] = dim3( 89,  35, 112); // purple
        stateColors[   TC] = dim3(255,   0,   0); // red
        stateColors[EMPTY] = dim3(255, 255, 255); // white
        CudaSafeCall(cudaMallocManaged((void**)&geneColors, nGenes*sizeof(dim3)));

        prefixes = (char**)malloc((nCarcin+1)*sizeof(char*));
        outNames = (char**)malloc((nCarcin+2)*sizeof(char*));
        prefixes[0] = (char*)calloc(7, 1);
        strcat(prefixes[0], "genes_");
        outNames[0] = (char*)calloc(11, 1);
        strcat(outNames[0], "out_ca.mp4");
        outNames[1] = (char*)calloc(14, 1);
        strcat(outNames[1], "out_genes.mp4");
        for (i = 0; i < nCarcin; i++) {
            prefixes[i+1] = (char*)calloc(15, 1);
            sprintf(prefixes[i+1], "carcin%d_", i);
            outNames[i+2] = (char*)calloc(25, 1);
            sprintf(outNames[i+2], "out_carcin%d.mp4", i);
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

        bytesPerCell = 2 * num_digits(gridSize * gridSize)
                     + num_digits(maxT + cellLifeSpan / cellCycleLen)
                     + (nDigInt + nDigFrac + 1) * nPheno
                     + (2 * (nDigInt + nDigFrac) + 4) * nGenes + 19;
        cellDataSize = bytesPerCell * gridSize * gridSize + 1;
        CudaSafeCall(cudaMallocManaged((void**)&cellData, cellDataSize));
        cellData[cellDataSize-1] = '\0';
        headerCellData = (char*)calloc(93, 1);
        strcat(headerCellData, "# idx,state,age,prolif,quies,apop,diff");
        strcat(headerCellData, ",[geneExprs],chosenPheno,chosenCell");
        strcat(headerCellData, ",actionDone,excised\n");
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

    void init(double *diffusion, double *influx, double *outflux, double *ic,
              double *bc, double *Wx, double *Wy, double alpha, double bias,
              dim3 *genecolors, CellParams *params, double *weightStates=NULL,
              unsigned rTC=0, unsigned cX=0, unsigned cY=0)
    {
        unsigned i, j, k, numFinished, dev;
        int nt = omp_get_num_procs(), counts[nt] = { 0 };
        dim3 blocks(NBLOCKS(gridSize, blockSize), NBLOCKS(gridSize, blockSize));
        dim3 threads(blockSize, blockSize);
        cudaStream_t prefetch[nCarcin+5]; cudaStream_t kernel[nCarcin+1];

        for (i = 0; i < nGenes; i++) {
            geneColors[i].x = genecolors[i].x;
            geneColors[i].y = genecolors[i].y;
            geneColors[i].z = genecolors[i].z;
        }

        for (i = 0; i < nCarcin+5; i++) {
            CudaSafeCall(cudaStreamCreate(&prefetch[i]));
            if (i < nCarcin+1) CudaSafeCall(cudaStreamCreate(&kernel[i]));
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

        cells_gpu_to_gpu_cpy<<< blocks, threads, 0, kernel[0] >>>(
            newGrid, prevGrid, gridSize, nGenes
        );
        CudaCheckError();

        for (k = 0; k < nCarcin; k++) {
            dev = k % 2 == 1 ? devId1 : devId2;
            pdes[k] = CarcinPDE(dev, k, gridSize, diffusion[k], influx[k],
                                outflux[k], ic[k], bc[k], cellVolume,
                                cellCycleLen);
            pdes[k].prefetch_memory(dev, &prefetch[k+5]);
            pdes[k].init(blockSize, &kernel[k+1]);
        }

        *NN = GeneExprNN(devId2, nCarcin+1, nGenes, alpha, bias);
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
        if (nCarcin != 0)
            CudaSafeCall(cudaMemPrefetchAsync(pdes,
                                              nCarcin*sizeof(CarcinPDE),
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
        CudaSafeCall(cudaMemPrefetchAsync(cellData,
                                          cellDataSize,
                                          devId2, prefetch[3]));

        for (i = 0; i < nCarcin+5; i++) {
            CudaSafeCall(cudaStreamSynchronize(prefetch[i]));
            CudaSafeCall(cudaStreamDestroy(prefetch[i]));
            if (i < nCarcin+1) {
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
                  (void (*)(uchar4*,unsigned,void*,
                            unsigned,bool,bool))anim_gpu_genes,
                  (void (*)(uchar4*,unsigned,void*,unsigned,
                            unsigned,bool,bool))anim_gpu_carcin,
                  (void (*)(uchar4*,unsigned,void*,
                            unsigned,unsigned,bool))anim_gpu_cell,
                  (void (*)(void*,bool,unsigned,
                            bool,bool))anim_gpu_timer_and_saver);
    }
};

#endif // __CA_H__