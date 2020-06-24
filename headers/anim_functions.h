#ifndef __ANIM_FUNCTIONS_H__
#define __ANIM_FUNCTIONS_H__

void anim_gpu_ca(uchar4* outputBitmap, unsigned dim, CA *ca,
                 unsigned ticks, bool display, bool paused, bool excise,
                 unsigned radius, unsigned centerX, unsigned centerY,
                 bool *windowsShouldClose, bool *keys)
{
    unsigned i, j, carcinIdx, deactivated, *numTC, *countData, *rTC, numCells;
    int *tcX, *tcY;
    bool excisionPerformed = false, activate = false;
    char answer[9] = { '\0' };
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);
    cudaStream_t streams[ca->maxNCarcin+2], *streamsExcise;

    for (i = 0; i < ca->maxNCarcin+2; i++)
        CudaSafeCall(cudaStreamCreate(&streams[i]));

    if (keys[0])
        ca->save ? ca->save = false : ca->save = true;
    else if (keys[1]) {
        printf("Enter how many time steps TC are alive: ");
        scanf("%d", &ca->maxTTCAlive);
    } else if (keys[2])
        ca->perfectExcision ? ca->perfectExcision = false
                            : ca->perfectExcision = true;
    else if (keys[3]) {
        printf("Do you want to activate or deactivate? ");
        fflush(stdout);
        scanf("%8s", answer);
        if (strcmp("activate", answer) == 0) activate = true;
        printf("Enter how many carcinogens (0-%d): ", ca->maxNCarcin);
        scanf("%d", &ca->nCarcin);
        if (ca->nCarcin == 0) {
            do {
                printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
                scanf("%d", &carcinIdx);
                if (carcinIdx < ca->maxNCarcin) {
                    ca->carcinogens[carcinIdx] = activate;
                    if (activate) ca->nCarcin++;
                    else {
                        ca->pdes[carcinIdx].t = 0;
                        ca->pdes[carcinIdx].nCycles = 0;
                    }
                }
                printf("Do you want to enter another carcinogen index? (yes/no) ");
                fflush(stdout);
                scanf("%3s", answer);
            } while(strcmp("no", answer) != 0);
        } else if (ca->nCarcin > 0) {
            deactivated = 0;
            for (i = 0; i < ca->nCarcin; i++)
                if (i < ca->maxNCarcin) {
                    ca->carcinogens[i] = activate;
                    if (!activate) {
                        deactivated++;
                        ca->pdes[i].t = 0;
                        ca->pdes[i].nCycles = 0;
                    }
                } else ca->nCarcin = ca->maxNCarcin;
            if (!activate) ca->nCarcin -= deactivated;
        }
    } else if (keys[4]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the number of time steps influx occurs: ");
                scanf("%d", &ca->pdes[carcinIdx].maxTInflux);
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while(strcmp("no", answer) != 0);
    } else if (keys[5]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the number of time steps no influx occurs: ");
                scanf("%d", &ca->pdes[carcinIdx].maxTNoInflux);
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while(strcmp("no", answer) != 0);
    } else if (keys[6]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the change in time step for the influx (hours): ");
                scanf("%lf", &ca->pdes[carcinIdx].exposureTime);
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while(strcmp("no", answer) != 0);
    }

    if (ticks <= ca->maxT) {
        if (ticks == 0 && (display || ca->save)) {
            display_ca<<< blocks, threads, 0, streams[0] >>>(
                outputBitmap, ca->newGrid, ca->gridSize,
                ca->cellSize, dim, ca->stateColors
            );
            CudaCheckError();
            if (!paused)
                save_cell_data_to_file(ca, ticks, blocks, threads, &streams[1]);
            CudaSafeCall(cudaStreamSynchronize(streams[0]));
        } else if (!paused) {
            mutate_grid<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->gridSize, ca->NN,
                ca->pdes, ca->carcinogens, ticks
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            if (ca->nCarcin != 0)
                for (i = 0; i < ca->maxNCarcin; i++)
                    if (ca->carcinogens[i])
                        ca->pdes[i].time_step(ca->blockSize, &streams[i+2]);

            cells_gpu_to_gpu_cpy<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            cudaMallocManaged((void**)&countData,
                              (3*ca->nStates+4)*sizeof(unsigned));
            memset(countData, 0, (3*ca->nStates+4)*sizeof(unsigned));
            rule<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize,
                ca->nGenes, ticks, countData, false
            );
            CudaCheckError();
            rule<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize,
                ca->nGenes, ticks, countData, true
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            for (i = 0; i < 4; i++) {
                save_count_data(ca->countFiles[i+7], ca->headerCount, ticks,
                                countData[i], 0, 0, 0);
                for (j = 0; j < ca->nStates; j++) {
                    if (j == EMPTY || (i == DIFF && (j == NC || j == MNC || j == TC)))
                        continue;
                    if (i != DIFF)
                        save_count_data(ca->countFiles[j*3+(ca->nGenes+11)+i],
                                        ca->headerCount, ticks,
                                        countData[(j*3+4)+i], 0, 0, 0);
                    else
                        save_count_data(ca->countFiles[j+(3*ca->nStates+ca->nGenes+6)],
                                        ca->headerCount, ticks,
                                        countData[(3*ca->nStates-1)+j], 0, 0, 0);
                }
            }
            CudaSafeCall(cudaFree(countData)); countData = NULL;

            CudaSafeCall(cudaMallocManaged((void**)&numTC,
                                           sizeof(unsigned)));
            *numTC = 0;
            streamsExcise = (cudaStream_t*)malloc((ca->exciseCount+1)*sizeof(cudaStream_t));
            for (i = 0; i < ca->exciseCount+1; i++) {
                CudaSafeCall(cudaStreamCreate(&streamsExcise[i]));
                if (i != 0 && ca->tcFormed[i]) continue;
                check_CSC_or_TC_formed<<< blocks, threads, 0, streamsExcise[i] >>>(
                    ca->newGrid, ca->prevGrid, ca->gridSize, ticks, ca->cscFormed,
                    ca->tcFormed, i, ca->timeTCDead, ca->perfectExcision,
                    ca->radius, ca->centerX, ca->centerY, numTC
                );
                CudaCheckError();
            }

            update_states<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            for (i = 0; i < ca->exciseCount+1; i++) {
                CudaSafeCall(cudaStreamSynchronize(streamsExcise[i]));
                CudaSafeCall(cudaStreamDestroy(streamsExcise[i]));
            }
            free(streamsExcise); streamsExcise = NULL;

            if (*numTC != 0) ca->timeTCAlive++;
            else ca->timeTCAlive = 0;

            for (i = 0; i < ca->exciseCount+1; i++)
                if (!ca->tcFormed[i])
                    ca->timeTCDead[i]++;

            if ((excise || (ca->maxTTCAlive != -1
             && ca->timeTCAlive != 0 && ca->timeTCAlive % ca->maxTTCAlive == 0))
             && ca->exciseCount == ca->maxExcise)
                printf("The maximum number of excisions have been performed.\n");

            if (ca->exciseCount < ca->maxExcise
             && (excise || (ca->maxTTCAlive != -1
             && ca->timeTCAlive != 0 && ca->timeTCAlive % ca->maxTTCAlive == 0))) {
                if (!ca->perfectExcision) {
                    CudaSafeCall(cudaMallocManaged((void**)&rTC,
                                                   sizeof(unsigned)));
                    *rTC = 0;
                    CudaSafeCall(cudaMallocManaged((void**)&tcX, sizeof(int)));
                    CudaSafeCall(cudaMallocManaged((void**)&tcY, sizeof(int)));
                    *tcX = -1; *tcY = -1;
                    if (radius == 0 && *numTC != 0) {
                        set_excision_circle<<< blocks, threads, 0, streams[0] >>>(
                            ca->newGrid, ca->gridSize, rTC, tcX, tcY
                        );
                        CudaCheckError();
                        CudaSafeCall(cudaStreamSynchronize(streams[0]));
                    } else if (radius != 0) {
                        *rTC = radius; *tcX = centerX; *tcY = centerY;
                    }
                    if (*rTC != 0) {
                        excision<<< blocks, threads, 0, streams[0] >>>(
                            ca->newGrid, ca->gridSize, ca->nGenes,
                            *rTC, *tcX, *tcY, numTC
                        );
                        CudaCheckError();
                        CudaSafeCall(cudaStreamSynchronize(streams[0]));
                        printf("Radius of excision: %d, center: (%d, %d)\n",
                               *rTC, *tcX, *tcY);

                        excisionPerformed = true;
                        ca->radius[ca->exciseCount+1] = *rTC;
                        ca->centerX[ca->exciseCount+1] = *tcX;
                        ca->centerY[ca->exciseCount+1] = *tcY;
                        if (*numTC == 0) ca->timeTCAlive = 0;
                    }
                    CudaSafeCall(cudaFree(rTC)); rTC = NULL;
                    CudaSafeCall(cudaFree(tcX)); tcX = NULL;
                    CudaSafeCall(cudaFree(tcY)); tcY = NULL;
                } else if (ca->perfectExcision && ca->tcFormed[ca->exciseCount]) {
                    tumour_excision<<< blocks, threads, 0, streams[0] >>>(
                        ca->newGrid, ca->gridSize, ca->nGenes
                    );
                    CudaCheckError();

                    excisionPerformed = true;
                    ca->timeTCAlive = 0;
                }

                if (excisionPerformed) {
                    printf("Excision was performed at time step %d.\n",
                           ticks);
                    ca->exciseCount++;
                }
            }
            CudaSafeCall(cudaFree(numTC)); numTC = NULL;

            save_cell_data_to_file(ca, ticks, blocks, threads, &streams[0]);

            reset_rule_params<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize
            );
            CudaCheckError();

            cells_gpu_to_gpu_cpy<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            if ((display || ca->save)
             && ticks % ca->framerate == 0) {
                display_ca<<< blocks, threads, 0, streams[0] >>>(
                    outputBitmap, ca->newGrid, ca->gridSize,
                    ca->cellSize, dim, ca->stateColors);
                CudaCheckError();
            }
        }

        if (!paused) {
            CudaSafeCall(cudaMallocManaged((void**)&countData,
                                           (ca->nStates+ca->nStates*ca->nGenes)*sizeof(unsigned)));
            memset(countData, 0,
                   (ca->nStates+ca->nStates*ca->nGenes)*sizeof(unsigned));
            collect_data<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, countData, ca->gridSize, ca->nGenes
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));
            for (i = 0; i < ca->nStates; i++) {
                save_count_data(ca->countFiles[i], ca->headerCount, ticks,
                                countData[i], ca->stateColors[i].x,
                                ca->stateColors[i].y, ca->stateColors[i].z);
                for (j = 0; j < ca->nGenes; j++) {
                    if (i == EMPTY) continue;
                    save_count_data(ca->countFiles[i*ca->nGenes+(3*ca->nStates+ca->nGenes+11)+j],
                                    ca->headerCount, ticks,
                                    countData[i*ca->nGenes+(ca->nGenes+ca->nStates)+j],
                                    ca->stateColors[i].x, ca->stateColors[i].y,
                                    ca->stateColors[i].z);
               }
            }
            for (i = 0; i < ca->nGenes; i++)
                save_count_data(ca->countFiles[i+11], ca->headerCount, ticks,
                                countData[i+7], ca->geneColors[i].x,
                                ca->geneColors[i].y, ca->geneColors[i].z);

            numCells = ca->gridSize * ca->gridSize;
            if (countData[EMPTY] == numCells
             || countData[CSC] + countData[TC] + countData[EMPTY] == numCells)
                *windowsShouldClose = true;
            CudaSafeCall(cudaFree(countData)); countData = NULL;
        }

        for (i = 0; i < ca->maxNCarcin+2; i++)
            CudaSafeCall(cudaStreamSynchronize(streams[i]));

        if (!paused && ca->nCarcin > 0)
            for (i = 0; i < ca->maxNCarcin; i++)
                if (ca->carcinogens[i]) ca->pdes[i].t++;

        if (ca->save && !paused
         && ticks % ca->framerate == 0)
            save_image(outputBitmap, dim, ca->blockSize, NULL, ticks,
                       ca->devId2);
    }

    for (i = 0; i < ca->maxNCarcin+2; i++)
        CudaSafeCall(cudaStreamDestroy(streams[i]));
}

void anim_gpu_genes(uchar4* outputBitmap, unsigned dim, CA *ca,
                    unsigned ticks, bool display, bool paused)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_genes<<< blocks, threads >>>(
            outputBitmap, ca->newGrid, ca->gridSize, ca->cellSize, dim,
            ca->NN->nOut, ca->geneColors
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    if (ca->save && ticks <= ca->maxT && !paused
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[0], ticks, ca->devId2);
}

void anim_gpu_carcin(uchar4* outputBitmap, unsigned dim, CA *ca,
                     unsigned carcinIdx, unsigned ticks,
                     bool display, bool paused)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_carcin<<< blocks, threads >>>(
            outputBitmap, &ca->pdes[carcinIdx],
            ca->gridSize, ca->cellSize, dim
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    if (ca->save && ticks <= ca->maxT && !paused
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[carcinIdx+1], ticks, ca->devId2);

}

void anim_gpu_cell(uchar4* outputBitmap, unsigned dim, CA *ca,
                   unsigned cellIdx, unsigned ticks, bool display)
{
    dim3 blocks(NBLOCKS(dim, ca->blockSize), NBLOCKS(dim, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_cell_data<<< blocks, threads >>>(
            outputBitmap, ca->newGrid, cellIdx, dim, ca->NN->nOut,
            ca->stateColors
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }
}

void anim_gpu_timer_and_saver(CA *ca, bool start, unsigned ticks, bool paused,
                              bool windowsShouldClose)
{
    const unsigned videoFramerate = 24;
    unsigned carcinIdx;

    if (start && !windowsShouldClose) {
        ca->startStep = clock();
        printf("starting %d\n", ticks);
    } else if (!start) {
        ca->endStep = clock();
        printf("%d took %f seconds to complete.\n", ticks,
               (double) (ca->endStep - ca->startStep) / CLOCKS_PER_SEC);
    }
    if (ticks == 0 && !paused) ca->start = clock();

    if (!start && ca->save && (windowsShouldClose
    || (!paused && ticks == ca->maxT))) {
        #pragma omp parallel sections num_threads(3)
        {
            #pragma omp section
            {
                printf("Saving video %s.\n", ca->outNames[0]);
                save_video(NULL, ca->outNames[0], videoFramerate);
                printf("Finished video %s.\n", ca->outNames[0]);
            }
            #pragma omp section
            {
                printf("Saving video %s.\n", ca->outNames[1]);
                save_video(ca->prefixes[0], ca->outNames[1],
                           videoFramerate);
                printf("Finished video %s.\n", ca->outNames[1]);
            }
            #pragma omp section
            {
                for (carcinIdx = 0; carcinIdx < ca->maxNCarcin; carcinIdx++) {
                    if (!ca->carcinogens[carcinIdx]) continue;
                    printf("Saving video %s.\n", ca->outNames[carcinIdx+2]);
                    save_video(ca->prefixes[carcinIdx+1],
                               ca->outNames[carcinIdx+2], videoFramerate);
                    printf("Finished video %s.\n", ca->outNames[carcinIdx+2]);
                }
            }
        }
    }

    if (!start && (ticks == ca->maxT || windowsShouldClose)) {
        ca->end = clock();
        printf("It took %f seconds to run the %d time steps.\n",
               (double) (ca->end - ca->start) / CLOCKS_PER_SEC, ticks);
    }
}

#endif // __ANIM_FUNCTIONS_H__