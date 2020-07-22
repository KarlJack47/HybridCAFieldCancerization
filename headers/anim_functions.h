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
    char answer[9] = { '\0' }, fname[50] = { '\0' };
    double *phenoSum, denom;
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
                    ca->activeCarcin[carcinIdx] = activate;
                    if (activate) ca->nCarcin++;
                    else {
                        ca->carcins[carcinIdx].t = 0;
                        ca->carcins[carcinIdx].nCycles = 0;
                    }
                }
                printf("Do you want to enter another carcinogen index? (yes/no) ");
                fflush(stdout);
                scanf("%3s", answer);
            } while (strcmp("no", answer) != 0);
        } else if (ca->nCarcin > 0) {
            deactivated = 0;
            for (i = 0; i < ca->nCarcin; i++)
                if (i < ca->maxNCarcin) {
                    ca->activeCarcin[i] = activate;
                    if (!activate) {
                        deactivated++;
                        ca->carcins[i].t = 0;
                        ca->carcins[i].nCycles = 0;
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
                scanf("%d", &ca->carcins[carcinIdx].maxTInflux);
                if (ca->carcins[carcinIdx].maxTInflux != -1)
                    ca->carcins[carcinIdx].maxTInflux *= ca->carcins[carcinIdx].exposureTime;
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while (strcmp("no", answer) != 0);
    } else if (keys[5]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the number of time steps no influx occurs: ");
                scanf("%d", &ca->carcins[carcinIdx].maxTNoInflux);
                if (ca->carcins[carcinIdx].maxTNoInflux != -1)
                    ca->carcins[carcinIdx].maxTNoInflux *= ca->carcins[carcinIdx].exposureTime;
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while (strcmp("no", answer) != 0);
    } else if (keys[6]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the exposure time period (hours): ");
                scanf("%lf", &ca->carcins[carcinIdx].exposureTime);
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while (strcmp("no", answer) != 0);
    } else if (keys[7]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the sensitivity function index: ");
                scanf("%u", &ca->carcins[carcinIdx].funcIdx);
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while (strcmp("no", answer) != 0);
    } else if (keys[8]) {
        do {
            printf("Enter a carcinogen index (0-%d): ", ca->maxNCarcin-1);
            scanf("%d", &carcinIdx);
            if (carcinIdx < ca->maxNCarcin) {
                printf("Enter the carcinogen function type (0-2): ");
                scanf("%u", &ca->carcins[carcinIdx].type);
            }
            printf("Do you want to enter another carcinogen index? (yes/no) ");
            fflush(stdout);
            scanf("%3s", answer);
        } while (strcmp("no", answer) != 0);
    }

    if (ticks <= ca->maxT) {
        if (ticks == 0) {
            if (display || ca->save) {
                display_ca<<< blocks, threads, 0, streams[0] >>>(
                    outputBitmap, ca->newGrid, ca->gridSize,
                    ca->cellSize, dim, ca->stateColors
                );
                CudaCheckError();
            }
            if (!paused)
                save_cell_data_to_file(ca, ticks, blocks, threads, &streams[1]);
            CudaSafeCall(cudaStreamSynchronize(streams[0]));
            CudaSafeCall(cudaStreamSynchronize(streams[1]));
        } else if (!paused) {
            mutate_grid<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->gridSize, ca->NN,
                ca->carcins, ca->activeCarcin, ticks
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            if (ca->nCarcin != 0)
                for (i = 0; i < ca->maxNCarcin; i++)
                    if (ca->activeCarcin[i])
                        ca->carcins[i].time_step(ca->blockSize, &streams[i+2]);

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

            update_states<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->gridSize, ca->nGenes, ticks
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
                    CudaSafeCall(cudaStreamSynchronize(streams[0]));

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

            *ca->nLineage = 0; *ca->nLineageCells = 0;
            memset(ca->cellLineage, 0, ca->gridSize*ca->gridSize*sizeof(unsigned));
            memset(ca->stateInLineage, 0,
                   (ca->nStates-1)*ca->gridSize*ca->gridSize*sizeof(bool));
            update_cell_lineage<<< blocks, threads, 0, streams[1] >>>(
                ca->newGrid, ca->cellLineage, ca->stateInLineage,
                ca->nLineage, ca->nLineageCells, ca->gridSize
            );
            CudaCheckError();

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
                CudaSafeCall(cudaStreamSynchronize(streams[1]));
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
            CudaSafeCall(cudaMallocManaged((void**)&phenoSum,
                                           4*ca->nStates*sizeof(double)));
            memset(phenoSum, 0, 4*ca->nStates*sizeof(double));
            collect_data<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, countData, phenoSum, ca->gridSize, ca->nGenes
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));
            for (i = 0; i < ca->nStates; i++) {
                save_count_data(ca->countFiles[i], ca->headerCount, ticks,
                                countData[i], ca->stateColors[i].x,
                                ca->stateColors[i].y, ca->stateColors[i].z);
                for (j = 0; j < 4; j++) {
                    if (j == 3 && (i == NC || i == MNC || i == TC)) continue;
                    sprintf(fname, "chancePheno%d_State%d.data", j, i);
                    denom = countData[i];
                    if (i == 6) {
                        denom = ca->gridSize * ca->gridSize - countData[EMPTY];
                        if (j == 3) denom -= (countData[NC] + countData[MNC]
                                              + countData[TC]);
                    }
                    save_count_data(fname, NULL, ticks,
                                    (i != 6 && countData[i] == 0) ? 0 :
                                     phenoSum[i*4+j] / denom,
                                    ca->stateColors[i].x, ca->stateColors[i].y,
                                    ca->stateColors[i].z);
                }
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
            CudaSafeCall(cudaFree(phenoSum)); phenoSum = NULL;
        }

        for (i = 0; i < ca->maxNCarcin+2; i++)
            CudaSafeCall(cudaStreamSynchronize(streams[i]));

        if (!paused) printf("Number of cell Lineages: %u\n", *ca->nLineage);
        if (!paused) printf("Number of cells in lineages: %u\n", *ca->nLineageCells);

        if (!paused && ca->nCarcin > 0)
            for (i = 0; i < ca->maxNCarcin; i++)
                if (ca->activeCarcin[i]) ca->carcins[i].t++;

        if (ca->save && !paused
         && ticks % ca->framerate == 0)
            save_image(outputBitmap, dim, ca->blockSize, NULL, ticks,
                       ca->devId2);
    }

    for (i = 0; i < ca->maxNCarcin+2; i++)
        CudaSafeCall(cudaStreamDestroy(streams[i]));
}

void anim_gpu_lineage(uchar4* outputBitmap, unsigned dim, CA *ca, unsigned idx,
                      unsigned stateIdx, unsigned ticks, bool display, bool paused)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);
    unsigned i;
    char fname[15] = { '\0' };

    if (idx == 2) {
        if (!paused)
            printf("Top Ten Cell Lineage Sizes for %u: [", stateIdx);
        for (i = 0; i < 10; i++) {
            ca->maxLineages[i] = 0;
            max_lineage<<< blocks, threads >>>(
                ca->newGrid, ca->cellLineage, ca->maxLineages,
                i, stateIdx, ca->gridSize
            );
            CudaCheckError();
            CudaSafeCall(cudaDeviceSynchronize());
            if (!paused) printf(" %u", ca->maxLineages[i]);
        }
        if (!paused) printf(" ]\n");
    }

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        if (idx == 0)
            display_genes<<< blocks, threads >>>(
                outputBitmap, ca->newGrid, ca->gridSize, ca->cellSize, dim,
                ca->NN->nOut, ca->geneColors
            );
        else if (idx == 2)
            display_maxLineages<<< blocks, threads >>>(
                outputBitmap, ca->newGrid, ca->cellLineage, ca->stateInLineage,
                ca->maxLineages, stateIdx, ca->gridSize, ca->cellSize, dim,
                ca->lineageColors
            );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    if (idx == 1 && !paused) {
        memset(ca->percentageCounts, 0, 10*sizeof(unsigned));
        display_heatmap<<< blocks, threads >>>(
            outputBitmap, ca->newGrid, ca->cellLineage,
            ca->nLineageCells, ca->percentageCounts,
            ca->gridSize, ca->cellSize, dim, ca->heatmap
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
        for (i = 0; i < 10; i++) {
            sprintf(fname, "%d-%d%%.data", i == 0 ? i*10 : i * 10 + 1, (i + 1) * 10);
            save_count_data(fname, NULL, ticks, ca->percentageCounts[i],
                            ca->heatmap[i].x, ca->heatmap[i].y, ca->heatmap[i].z);
        }
    }

    if (ca->save && ticks <= ca->maxT && !paused
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[idx == 2 ? idx+stateIdx : idx],
                   ticks, ca->devId2);
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
            outputBitmap, &ca->carcins[carcinIdx],
            ca->gridSize, ca->cellSize, dim
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    if (ca->save && ticks <= ca->maxT && !paused
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[carcinIdx+9], ticks, ca->devId2);

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
    unsigned videoFramerate = 24;
    long int startPoint = (int) (ticks - videoFramerate);
    unsigned carcinIdx, i;
    char *vidListName= NULL; struct stat buffer;

    if (start && !windowsShouldClose) {
        ca->startStep = clock();
        printf("starting %d\n", ticks);
    } else if (!start) {
        ca->endStep = clock();
        printf("%d took %f seconds to complete.\n", ticks,
               (double) (ca->endStep - ca->startStep) / CLOCKS_PER_SEC);
    }
    if (ticks == 0 && !paused) ca->start = clock();

    if (!start && ca->save && (windowsShouldClose || (!paused && ((ticks != 0
     && startPoint % videoFramerate == 0) || ticks == ca->maxT)))) {
        if (ticks < videoFramerate) {
            startPoint = 0;
            videoFramerate = 1;
        }
        if (ticks == ca->maxT && ticks > videoFramerate)
            startPoint = (int) (ticks - ticks % videoFramerate);
        #pragma omp parallel sections num_threads(5)
        {
            #pragma omp section
            {
                fflush(stdout);
                printf("Saving video %s.\n", ca->outNames[0]);
                save_video(NULL, ca->outNames[0], startPoint, videoFramerate);
                fflush(stdout);
                printf("Finished video %s.\n", ca->outNames[0]);
            }
            #pragma omp section
            {
                fflush(stdout);
                printf("Saving video %s.\n", ca->outNames[1]);
                save_video(ca->prefixes[0], ca->outNames[1], startPoint,
                           videoFramerate);
                fflush(stdout);
                printf("Finished video %s.\n", ca->outNames[1]);
            }
            #pragma omp section
            {
                fflush(stdout);
                printf("Saving video %s.\n", ca->outNames[2]);
                save_video(ca->prefixes[1], ca->outNames[2], startPoint,
                           videoFramerate);
                fflush(stdout);
                printf("Finished video %s.\n", ca->outNames[2]);
            }
            #pragma omp section
            {
                for (i = 0; i < ca->nStates; i++) {
                    fflush(stdout);
                    printf("Saving video %s.\n", ca->outNames[i+3]);
                    save_video(ca->prefixes[i+2], ca->outNames[i+3], startPoint,
                               videoFramerate);
                    fflush(stdout);
                    printf("Finished video %s.\n", ca->outNames[i+3]);
                }
            }
            #pragma omp section
            {
                for (carcinIdx = 0; carcinIdx < ca->maxNCarcin; carcinIdx++) {
                    if (!ca->activeCarcin[carcinIdx]) continue;
                    fflush(stdout);
                    printf("Saving video %s.\n", ca->outNames[carcinIdx+4]);
                    save_video(ca->prefixes[carcinIdx+3], ca->outNames[carcinIdx+4],
                               startPoint, videoFramerate);
                    fflush(stdout);
                    printf("Finished video %s.\n", ca->outNames[carcinIdx+4]);
                }
            }
        }
    }

    if (!start && (ticks == ca->maxT || windowsShouldClose)) {
        ca->end = clock();
        printf("It took %f seconds to run the %d time steps.\n",
               (double) (ca->end - ca->start) / CLOCKS_PER_SEC, ticks);
        if (!ca->save) return;
        for (i = 0; i < ca->maxNCarcin+10; i++) {
            vidListName = (char*)calloc((i == ca->maxNCarcin+9 ? 0
                                         : strlen(ca->prefixes[i])) + 11, 1);
            if (i != ca->maxNCarcin+9) strcat(vidListName, ca->prefixes[i]);
            strcat(vidListName, "videos.txt");
            if (stat(vidListName, &buffer) == -1) continue;
            if (remove(vidListName) != 0)
                fprintf(stderr, "Error removing %s.\n", vidListName);
            free(vidListName); vidListName = NULL;
        }
    }
}

#endif // __ANIM_FUNCTIONS_H__