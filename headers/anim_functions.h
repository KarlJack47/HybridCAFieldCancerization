#ifndef __ANIM_FUNCTIONS_H__
#define __ANIM_FUNCTIONS_H__

void anim_gpu_ca(uchar4* outputBitmap, unsigned dim, CA *ca,
                 unsigned ticks, bool display, bool paused, bool excise,
                 bool windowsShouldClose, cudaStream_t stream)
{
    unsigned i, *count;
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);
    cudaStream_t streams[3+ca->nCarcin];

    for (i = 0; i < ca->nCarcin+3; i++)
        CudaSafeCall(cudaStreamCreate(&streams[i]));

    if (ticks <= ca->maxT) {
        if (ticks == 0 && (display || ca->save)) {
            display_ca<<< blocks, threads, 0, stream >>>(
                outputBitmap, ca->newGrid, ca->gridSize,
                ca->cellSize, dim, ca->stateColors
            );
            CudaCheckError();
        } else if (!paused) {
            mutate_grid<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->gridSize, ca->NN, ca->pdes, ticks
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            for (i = 0; i < ca->nCarcin; i++)
                ca->pdes[i].time_step(ca->cellCycleLen * ticks,
                                      ca->cellVolume, ca->blockSize,
                                      streams[i+3]);

            cells_gpu_to_gpu_copy<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            cudaMallocManaged((void**)&count, 12*sizeof(unsigned));
            memset(count, 0, 12*sizeof(unsigned));
            rule<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize,
                ca->nGenes, ticks, count, false
            );
            CudaCheckError();
            rule<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize,
                ca->nGenes, ticks, count, true
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            check_CSC_or_TC_formed<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize, ticks,
                ca->cscFormed, ca->tcFormed, ca->exciseCount, ca->timeTCDead
            );
            CudaCheckError();

            update_states<<< blocks, threads, 0, streams[1] >>>(
                ca->newGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            CudaSafeCall(cudaStreamSynchronize(streams[0]));
            reset_rule_params<<< blocks, threads, 0, streams[2] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[1]));

            if (ca->tcFormed[ca->exciseCount]) ca->timeTCAlive++;
            else ca->timeTCDead++;

            if (ca->exciseCount <= ca->maxExcise
             && ca->tcFormed[ca->exciseCount]) {
                if (excise || ca->timeTCAlive == ca->maxTTCAlive+1) {
                    tumour_excision<<< blocks, threads, 0, streams[0] >>>(
                        ca->newGrid, ca->gridSize, ca->nGenes
                    );
                    CudaCheckError();
                    CudaSafeCall(cudaStreamSynchronize(streams[0]));

                    printf("Tumour excision was performed at time step %d.\n",
                           ticks);
                    if (ca->exciseCount == ca->maxExcise) {
                        printf("The maximum number of tumour");
                        printf(" excisions has been performed.\n");
                    }
                    ca->exciseCount++;
                    ca->timeTCAlive = 0;
                    ca->timeTCDead = 1;
                }
            }

            cells_gpu_to_gpu_copy<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            if ((display || ca->save)
             && ticks % ca->framerate == 0) {
                display_ca<<< blocks, threads, 0, stream >>>(
                    outputBitmap, ca->newGrid, ca->gridSize,
                    ca->cellSize, dim, ca->stateColors);
                CudaCheckError();
            }

            for (i = 0; i < ca->nCarcin+3; i++)
                CudaSafeCall(cudaStreamSynchronize(streams[i]));

            printf("Empty: %d, Prolif: %d, Quies: %d, Apop: %d, Diff: %d, ",
                   count[0], count[1], count[2], count[3], count[4]);
            printf("Nothing: %d, without empty neigh: %d, ",
                   count[5], count[6]);
            printf("TC: %d, TC no action: %d, TC impossible action: %d, TC success: %d, TC died: %d\n",
                   count[7], count[8], count[9], count[10], count[11]);
            CudaSafeCall(cudaFree(count)); count = NULL;
        }

        CudaSafeCall(cudaStreamSynchronize(stream));
        if (ca->save && (!paused || (paused && windowsShouldClose))
         && ticks % ca->framerate == 0)
            save_image(outputBitmap, dim, ca->blockSize, NULL, ticks,
                       ca->maxT, ca->devId2);
    }

    for (i = 0; i < ca->nCarcin+3; i++)
        CudaSafeCall(cudaStreamDestroy(streams[i]));
}

void anim_gpu_genes(uchar4* outputBitmap, unsigned dim, CA *ca,
                    unsigned ticks, bool display, bool paused,
                    bool windowsShouldClose, cudaStream_t stream)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_genes<<< blocks, threads, 0, stream >>>(
            outputBitmap, ca->newGrid, ca->gridSize, ca->cellSize, dim,
            ca->NN->nOut, ca->geneColors
        );
        CudaCheckError();
    }

    CudaSafeCall(cudaStreamSynchronize(stream));
    if (ca->save && ((ticks <= ca->maxT && !paused)
                  || (paused && windowsShouldClose))
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[0], ticks, ca->maxT, ca->devId2);
}

void anim_gpu_carcin(uchar4* outputBitmap, unsigned dim, CA *ca,
                     unsigned carcinIdx, unsigned ticks,
                     bool display, bool paused, bool windowsShouldClose,
                     cudaStream_t stream)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_carcin<<< blocks, threads, 0, stream >>>(
            outputBitmap, &ca->pdes[carcinIdx],
            ca->gridSize, ca->cellSize, dim
        );
        CudaCheckError();
    }

    CudaSafeCall(cudaStreamSynchronize(stream));
    if (ca->save && ((ticks <= ca->maxT && !paused)
                  || (paused && windowsShouldClose))
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[carcinIdx+1], ticks, ca->maxT, ca->devId2);

}

void anim_gpu_cell(uchar4* outputBitmap, unsigned dim, CA *ca,
                   unsigned cellIdx, unsigned ticks, bool display,
                   cudaStream_t stream)
{
    dim3 blocks(NBLOCKS(dim, ca->blockSize), NBLOCKS(dim, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_cell_data<<< blocks, threads, 0, stream >>>(
            outputBitmap, ca->newGrid, cellIdx, dim, ca->NN->nOut,
            ca->stateColors
        );
        CudaCheckError();
    }
}

void anim_gpu_timer_and_saver(CA *ca, bool start, unsigned ticks,
                              bool paused, bool windowsShouldClose)
{
    const unsigned videoFramerate = 24;
    unsigned carcinIdx;

    if (start) {
        ca->startStep = clock();
        printf("starting %d\n", ticks);
    } else {
        ca->endStep = clock();
        printf("%d took %f seconds to complete.\n", ticks,
               (double) (ca->endStep - ca->startStep) / CLOCKS_PER_SEC);
    }
    if (ticks == 0) ca->start = clock();

    if (!start & ca->save && ((ticks == ca->maxT && !paused)
     || (paused && windowsShouldClose))) {
        #pragma omp parallel sections num_threads(3)
        {
            #pragma omp section
            {
                printf("Saving video %s.\n", ca->outNames[0]);
                save_video(NULL, ca->outNames[0], videoFramerate, ca->maxT);
                printf("Finished video %s.\n", ca->outNames[0]);
            }
            #pragma omp section
            {
                printf("Saving video %s.\n", ca->outNames[1]);
                save_video(ca->prefixes[0], ca->outNames[1],
                           videoFramerate, ca->maxT);
                printf("Finished video %s.\n", ca->outNames[1]);
            }
            #pragma omp section
            {
                for (carcinIdx = 0; carcinIdx < ca->NN->nIn-1; carcinIdx++) {
                    printf("Saving video %s.\n", ca->outNames[carcinIdx+2]);
                    save_video(ca->prefixes[carcinIdx+1], ca->outNames[carcinIdx+2],
                               videoFramerate, ca->maxT);
                    printf("Finished video %s.\n", ca->outNames[carcinIdx+2]);
                }
            }
        }
    }

    if (ticks == ca->maxT && !start) {
        ca->end = clock();
        printf("It took %f seconds to run the %d time steps.\n",
               (double) (ca->end - ca->start) / CLOCKS_PER_SEC, ca->maxT);
    }
}

#endif // __ANIM_FUNCTIONS_H__