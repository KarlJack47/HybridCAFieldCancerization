#ifndef __CA_H__
#define __CA_H__

struct CA {
    unsigned devId1, devId2;
    unsigned blockSize;

    Cell *prevGrid, *newGrid;
    CarcinPDE *pdes;
    GeneExprNN *NN;

    unsigned gridSize, cellSize, dim;
    unsigned maxT;
    int maxTTCAlive;

    unsigned nGenes;
    unsigned nCarcin;
    unsigned nStates;

    double cellVolume;
    double cellCycleLen;
    double cellLifeSpan;

    bool *tcFormed;
    bool *cscFormed;
    unsigned exciseCount;
    unsigned timeTCAlive;
    unsigned timeTCDead;
    unsigned maxExcise;

    clock_t start, end;
    clock_t startStep, endStep;

    unsigned framerate;
    dim3 *stateColors, *geneColors;
    bool save;
    char **prefixes, **outNames;

    CA(unsigned gSize, unsigned T, unsigned ngenes, unsigned ncarcin,
       bool saveIn, int maxTTC, unsigned dim)
    {
        unsigned i;

        blockSize = 16;

        gridSize = gSize;
        cellSize = dim / gridSize;
        maxT = T;
        save = saveIn;
        maxTTCAlive = maxTTC;

        nGenes = ngenes;
        nCarcin = ncarcin;
        nStates = 7;

        cellVolume = 1.596e-9; // relative to cm, epithelial cell
        cellCycleLen = 10.0; // in hours, for tastebuds
        cellLifeSpan = 250.0; // in hours, for tastebuds

        exciseCount = 0;
        timeTCAlive = 0;
        timeTCDead = 1;
        maxExcise = 100;

        prefixes = (char**)malloc((nCarcin+1)*sizeof(char*));
        outNames = (char**)malloc((nCarcin+2)*sizeof(char*));
        prefixes[0] = (char*)malloc(7*sizeof(char));
        memset(prefixes[0], '\0', 7*sizeof(char));
        strcat(prefixes[0], "genes_");
        outNames[0] = (char*)malloc(11*sizeof(char));
        memset(outNames[0], '\0', 11*sizeof(char));
        strcat(outNames[0], "out_ca.mp4");
        outNames[1] = (char*)malloc(14*sizeof(char));
        memset(outNames[1], '\0', 14*sizeof(char));
        strcat(outNames[1], "out_genes.mp4");

        for (i = 0; i < nCarcin; i++) {
            prefixes[i+1] = (char*)malloc(15*sizeof(char));
            sprintf(prefixes[i+1], "carcin%d_", i);
            outNames[i+2] = (char*)malloc(25*sizeof(char));
            sprintf(outNames[i+2], "out_carcin%d.mp4", i);
        }
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
        if (outNames != NULL && prefixes != NULL) {
            for (i = 0; i < nCarcin+2; i++) {
                free(outNames[i]); outNames[i] = NULL;
                if (i < nCarcin+1) { free(prefixes[i]); prefixes[i] = NULL; }
            }
            free(outNames); outNames = NULL;
            free(prefixes); prefixes = NULL;
        }
        if (stateColors != NULL) {
            CudaSafeCall(cudaFree(stateColors)); stateColors = NULL;
        }
        if (geneColors != NULL) {
            CudaSafeCall(cudaFree(geneColors)); geneColors = NULL;
        }
    }

    void set_params(double volume, double cycleLen, double lifeSpan)
    {
        cellVolume = volume;
        cellCycleLen = cycleLen;
        cellLifeSpan = lifeSpan;
    }

    void initialize_memory(void)
    {
        int i, j;
        int numDev;

        CudaSafeCall(cudaGetDeviceCount(&numDev));
        for (i = numDev-1; i > -1; i--) {
            CudaSafeCall(cudaSetDevice(i));
            for (j = i-1; j > -1; j--)
                CudaSafeCall(cudaDeviceEnablePeerAccess(j, 0));
            CudaSafeCall(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 33554432));
        }

        devId1 = 0; devId2 = 0;
        if (numDev == 2) devId2 = 1;

        CudaSafeCall(cudaMallocManaged((void**)&NN, sizeof(GeneExprNN)));
        if (nCarcin != 0)
            CudaSafeCall(cudaMallocManaged((void**)&pdes,
                                           nCarcin*sizeof(CarcinPDE)));
        else
            pdes = NULL;
        CudaSafeCall(cudaMallocManaged((void**)&prevGrid,
                                       gridSize*gridSize*sizeof(Cell)));
        CudaSafeCall(cudaMallocManaged((void**)&newGrid,
                                       gridSize*gridSize*sizeof(Cell)));

        CudaSafeCall(cudaMallocManaged((void**)&cscFormed, sizeof(bool)));
        *cscFormed = false;
        CudaSafeCall(cudaMallocManaged((void**)&tcFormed,
                                       (maxExcise+1)*sizeof(bool)));
        memset(tcFormed, false, (maxExcise+1)*sizeof(bool));
        CudaSafeCall(cudaMallocManaged((void**)&stateColors, nStates*sizeof(dim3)));
        CudaSafeCall(cudaMallocManaged((void**)&geneColors, nGenes*sizeof(dim3)));
	}

    void init(double *diffusion, double *influx, double *outflux, double *ic,
              double *bc, double *Wx, double *Wy, double alpha, double bias,
              dim3 *genecolors, CellParams *params, double *weightStates=NULL)
    {
        unsigned i, j, k;
        int nt = omp_get_num_procs(), counts[nt] = { 0 };
        int numFinished;
        dim3 blocks(NBLOCKS(gridSize, blockSize), NBLOCKS(gridSize, blockSize));
        dim3 threads(blockSize, blockSize);

        printf("Grid initialization progress:   0.00/100.00");
        for (i = 0; i < gridSize; i++)
            for (j = 0; j < gridSize; j+=2) {
                #pragma omp parallel sections num_threads(2)
                {
                    #pragma omp section
                    {
                        prevGrid[i*gridSize+j] = Cell(devId1, params, i, j,
                                                      gridSize, nGenes,
                                                      cellCycleLen,
                                                      cellLifeSpan,
                                                      weightStates);
                        prevGrid[i*gridSize+j].prefetch_memory(devId1,
                                                               gridSize,
                                                               nGenes);

                        newGrid[i*gridSize+j] = Cell(devId2, params, i, j,
                                                     gridSize, nGenes,
                                                     cellCycleLen,
                                                     cellLifeSpan,
                                                     weightStates);
                        newGrid[i*gridSize+j].prefetch_memory(devId2,
                                                              gridSize,
                                                              nGenes);
                        counts[omp_get_thread_num()]++;
                    }

                    #pragma omp section
                    {
                        prevGrid[i*gridSize+(j+1)] = Cell(devId1, params, i, j+1,
                                                          gridSize, nGenes,
                                                          cellCycleLen,
                                                          cellLifeSpan,
                                                          weightStates);
                        prevGrid[i*gridSize+(j+1)].prefetch_memory(devId1,
                                                                   gridSize,
                                                                   nGenes);

                        newGrid[i*gridSize+(j+1)] = Cell(devId2, params, i, j+1,
                                                         gridSize, nGenes,
                                                         cellCycleLen,
                                                         cellLifeSpan,
                                                         weightStates);
                        newGrid[i*gridSize+(j+1)].prefetch_memory(devId2,
                                                                  gridSize,
                                                                  nGenes);
                        counts[omp_get_thread_num()]++;
                    }
                }

                numFinished = 0;
                for (k = 0; k < nt; k++) numFinished += counts[k];
                print_progress(numFinished, gridSize*gridSize);
            }
        printf("\n");

        cells_gpu_to_gpu_copy<<< blocks, threads >>>(prevGrid, newGrid,
                                                     gridSize, nGenes);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

	    for (k = 0; k < nCarcin; k++) {
	        pdes[k] = CarcinPDE(devId2, k, gridSize, diffusion[k],
	                            influx[k], outflux[k], ic[k], bc[k]);
	        pdes[k].prefetch_memory(devId2);
	        pdes[k].init(cellVolume, blockSize);
	    }

        *NN = GeneExprNN(devId2, nCarcin+1, nGenes, alpha, bias);
        NN->memory_allocate(Wx, Wy);

        stateColors[   NC] = dim3(  0,   0,   0); // black
        stateColors[  MNC] = dim3( 87, 207,   0); // green
        stateColors[   SC] = dim3(244, 131,   0); // orange
        stateColors[  MSC] = dim3(  0,   0, 255); // blue
        stateColors[  CSC] = dim3( 89,  35, 112); // purple
        stateColors[   TC] = dim3(255,   0,   0); // red
        stateColors[EMPTY] = dim3(255, 255, 255); // white

        for (i = 0; i < nGenes; i++) {
            geneColors[i].x = genecolors[i].x;
            geneColors[i].y = genecolors[i].y;
            geneColors[i].z = genecolors[i].z;
        }

        CudaSafeCall(cudaMemPrefetchAsync(prevGrid,
                                          gridSize*gridSize*sizeof(Cell),
                                          devId1, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(newGrid,
                                          gridSize*gridSize*sizeof(Cell),
                                          devId2, NULL));
        if (nCarcin != 0)
            CudaSafeCall(cudaMemPrefetchAsync(pdes,
                                              nCarcin*sizeof(CarcinPDE),
                                              devId2, NULL));
        NN->prefetch_memory(devId1);
        CudaSafeCall(cudaMemPrefetchAsync(NN,
                                          sizeof(GeneExprNN),
                                          devId2, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(cscFormed,
                                          sizeof(bool),
                                          devId2, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(tcFormed,
                                          (maxExcise+1)*sizeof(bool),
                                          devId2, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(stateColors,
                                          7*sizeof(dim3),
                                          devId1, NULL));
        CudaSafeCall(cudaMemPrefetchAsync(geneColors,
                                          nGenes*sizeof(dim3),
                                          devId2, NULL));

        CudaSafeCall(cudaDeviceSynchronize());
    }

    void animate(unsigned frameRate, GUI *gui)
    {
        framerate = frameRate;
        gui->anim((void (*)(uchar4*,unsigned,void*,unsigned,bool,bool,bool,bool,cudaStream_t))anim_gpu_ca,
                  (void (*)(uchar4*,unsigned,void*,unsigned,bool,bool,bool,cudaStream_t))anim_gpu_genes,
                  (void (*)(uchar4*,unsigned,void*,unsigned,unsigned,bool,bool,bool,cudaStream_t))anim_gpu_carcin,
                  (void (*)(uchar4*,unsigned,void*,unsigned,unsigned,bool,cudaStream_t))anim_gpu_cell,
                  (void (*)(void*,bool,unsigned,bool,bool))anim_gpu_timer_and_saver);
    }
};

#endif // __CA_H__