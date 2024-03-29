#ifndef __OUTPUT_FUNCTIONS_H__
#define __OUTPUT_FUNCTIONS_H__

void print_progress(unsigned currAmount, unsigned N)
{
    char format[40] = { '\0' };
    double progress = (currAmount / (double) N) * 100.0;
    unsigned numDig = num_digits(progress), limit = numDig + 10, k;

    if (trunc(progress * 1000.0) >= 9995 && numDig == 1) limit += 1;
    else if (trunc(progress * 1000.0) >= 99995 && numDig == 2) limit += 1;

    for (k = 0; k < limit; k++) strcat(format, "\b");
    strcat(format, "%.2f/%.2f");
    printf(format, progress, 100.0);
    fflush(stdout);
}

__global__ void copy_frame(uchar4*, unsigned char*);
int save_image(uchar4 *outputBitmap, size_t size, unsigned blockSize,
               char *prefix, unsigned time, int dev, unsigned nDigitsMaxT=7)
{
    dim3 blocks(NBLOCKS(size, blockSize), NBLOCKS(size, blockSize));
    dim3 threads(blockSize, blockSize);
    unsigned dig = num_digits(time), i;
    char fname[150] = { '\0' };
    tjhandle tjInstance = NULL;
    unsigned long jpegSize = 0;
    unsigned char *jpegBuf = NULL;
    FILE *jpegFile = NULL;
    unsigned char *frame = NULL;
    cudaStream_t streams[2];
    for (i = 0; i < 2; i++) CudaSafeCall(cudaStreamCreate(&streams[i]));

    if (prefix != NULL) strcat(fname, prefix);
    for (i = 0; i < nDigitsMaxT-dig; i++) strcat(fname, "0");
    sprintf(&fname[strlen(fname)], "%d.jpeg", time);

    CudaSafeCall(cudaMallocManaged((void**)&frame,
                                   size*size*4*sizeof(unsigned char)));
    CudaSafeCall(cudaMemPrefetchAsync(frame, size*size*4*sizeof(unsigned char),
                                      dev, streams[0]));
    copy_frame<<< blocks, threads, 0, streams[1] >>>(outputBitmap, frame);
    CudaCheckError();
    for (i = 0; i < 2; i++) {
        CudaSafeCall(cudaStreamSynchronize(streams[i]));
        CudaSafeCall(cudaStreamDestroy(streams[i]));
    }

    if ((tjInstance = tjInitCompress()) == NULL) {
        THROW_TJ("initializing compressor");
        CudaSafeCall(cudaFree(frame)); frame = NULL;
        return -1;
    }
    if (tjCompress2(tjInstance, frame, size, 0, size, TJPF_RGBA,
                    &jpegBuf, &jpegSize, TJSAMP_420, 95, 0) < 0) {
        THROW_TJ("compressing image");
        tjDestroy(tjInstance); tjInstance = NULL;
        CudaSafeCall(cudaFree(frame)); frame = NULL;
		return -1;
	}
	tjDestroy(tjInstance); tjInstance = NULL;
	CudaSafeCall(cudaFree(frame)); frame = NULL;

    if ((jpegFile = fopen(fname, "wb")) == NULL) {
        THROW_UNIX("opening output file");
        tjFree(jpegBuf); jpegBuf = NULL;
		return -1;
	}
    if (fwrite(jpegBuf, jpegSize, 1, jpegFile) < 1) {
        THROW_UNIX("writing output file");
        fclose(jpegFile); jpegFile = NULL;
        tjFree(jpegBuf); jpegBuf = NULL;
        return -1;
    }

    fclose(jpegFile); jpegFile = NULL;
    tjFree(jpegBuf); jpegBuf = NULL;

    return 0;
}

void save_video(char *prefix, char *outputName, unsigned startPoint,
                unsigned framerate, unsigned nDigitsMaxT=7)
{
    unsigned i;
    char command[250] = { '\0' };
    char frames[2][(prefix ? strlen(prefix) : 0)+11] = { { '\0' }, { '\0' } };
    char *temp;
    FILE *vidList; struct stat buffer;
    char vidListName[(prefix ? strlen(prefix) : 0)+11] = { '\0' };

    for (i = 0; i < 2; i++) {
        if (prefix) strcat(frames[i], prefix);
        sprintf(&frames[i][strlen(frames[i])], "frame%d.mp4", i+1);
    }

    if (startPoint == 0) temp = outputName;
    else temp = frames[1];

    sprintf(command, "ffmpeg -y -v quiet -framerate %d -start_number %d -i ",
            framerate, startPoint);
    if (prefix) strcat(command, prefix);
    if (nDigitsMaxT == 1) strcat(command, "%%d.jpeg");
    else sprintf(&command[strlen(command)], "%%%d%dd.jpeg", 0, nDigitsMaxT);
    sprintf(&command[strlen(command)], " -c:v libx264 -pix_fmt yuv420p %s",
            temp);

    system(command);

    if (startPoint != 0) {
        if (prefix) strcat(vidListName, prefix);
        strcat(vidListName, "videos.txt");
        if (stat(vidListName, &buffer) == -1) {
            if (!(vidList = fopen(vidListName, "w")))
                fprintf(stderr, "Error opening %s\n", vidListName);
            else {
                fprintf(vidList, "file '%s'\nfile '%s'\n",
                        frames[0], frames[1]);
                fclose(vidList);
            }
        }

        if (rename(outputName, frames[0]) != 0)
            fprintf(stderr, "%s couldn't be renamed to %s\n",
                    outputName, frames[0]);

        memset(command, '\0', 250);
        strcat(command, "ffmpeg -y -v quiet -f concat -safe 0 ");
        sprintf(&command[strlen(command)], "-i %s -c copy %s",
                vidListName, outputName);
        system(command);
        if (remove(frames[0]) != 0)
            fprintf(stderr, "%s couldn't be deleted\n", frames[0]);
        if (remove(frames[1]) != 0)
            fprintf(stderr, "%s couldn't be deleted\n", frames[1]);
    }
}

int compress_and_save_data(const char *fname, const char *header,
                           char *input, size_t bytes) {
    FILE *fptr;
    char *data = input; size_t dataSize = bytes - 1;
    size_t headerSize = strlen(header);
    size_t maxDstSize = 0;
    char *compressedData = NULL; size_t compressedDataSize;
    LZ4F_preferences_t preferences = LZ4F_INIT_PREFERENCES;

    if (!(fptr = fopen(fname, "wb"))) {
        fprintf(stderr, "Error opening %s\n", fname);
        return 1;
    }

    if (header != NULL) {
        dataSize += headerSize;
        data = (char*)calloc(dataSize, 1);
        memcpy(data, header, headerSize);
        memcpy(data+headerSize, input, bytes - 1);
    }

    preferences.compressionLevel = 0;
    maxDstSize = LZ4F_compressFrameBound(dataSize, &preferences);
    if ((compressedData = (char*)calloc(maxDstSize, 1)) == NULL) {
        fprintf(stderr, "Failed to allocate memory for compressedData.\n");
        return 1;
    }
    compressedDataSize = LZ4F_compressFrame(compressedData, maxDstSize,
                                            data, dataSize, &preferences);
    if (header != NULL) { free(data); data = NULL; }
    if (compressedDataSize <= 0) {
        fprintf(stderr, "Failure compressing the data.");
        return 1;
    }

    fwrite(compressedData, 1, compressedDataSize, fptr);

    free(compressedData); compressedData = NULL;

    fclose(fptr);

    return 0;
}

__global__ void save_cell_data(Cell*,Cell*,char*,unsigned,unsigned,
                               double,double,unsigned,size_t);
void save_cell_data_to_file(CA *ca, unsigned t, dim3 blocks,
                            dim3 threads, cudaStream_t *stream)
{
    unsigned numChar = num_digits(t) + 10;
    //char *fname = (char*)calloc(numChar, 1);
    char fname[numChar] = { '\0' };

    if (!ca->saveCellData) return;

    sprintf(fname, "%d.data.lz4", t);

    save_cell_data<<< blocks, threads, 0, *stream >>>(
        ca->prevGrid, ca->newGrid, ca->cellData, ca->gridSize, ca->maxT,
        ca->cellLifeSpan, ca->cellCycleLen, ca->nGenes, ca->bytesPerCell
    );
    CudaCheckError();
    CudaSafeCall(cudaStreamSynchronize(*stream));

    compress_and_save_data(fname, ca->headerCellData, ca->cellData,
                           ca->cellDataSize);

    //free(fname); fname = NULL;
}

int save_count_data(const char *fname, const char *header, double t,
                    double count, unsigned red, unsigned green, unsigned blue)
{
    FILE *fptr;
    struct stat buffer;
    bool fileExists = (stat(fname, &buffer) == 0);

    if (!(fptr = fopen(fname, "a"))) {
        fprintf(stderr, "Error opening %s.\n", fname);
        return 1;
    }

    if (header != NULL && !fileExists) fprintf(fptr, "%s", header);

    fprintf(fptr, "%g\t%g\t%d\n", t, count, 65536 * red + 256 * green + blue);

    fclose(fptr);

    return 0;
}

void update_avg_data(unsigned i, const char *fname, unsigned ticks, double totalValue,
                     double mutatedValue, double nonMutatedValue, dim3 color)
{
    char tempName[100] = { '\0' };

    sprintf(tempName, "%s.data", fname);
    save_count_data(tempName, "# t avg color\n", ticks, totalValue,
                    color.x, color.y, color.z);
    if (i == EMPTY) {
        // Mutated Cells
        sprintf(tempName, "%s_Mutated.data", fname);
        save_count_data(tempName, "# t avg color\n", ticks,
                        mutatedValue, 0, 0, 0);
        // Non-Mutated Cells
        sprintf(tempName, "%s_Non-Mutated.data", fname);
        save_count_data(tempName, "# t avg color\n", ticks,
                        nonMutatedValue, 0, 0, 0);
    }
}

#endif // __OUTPUT_FUNCTIONS_H__
