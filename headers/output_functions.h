#ifndef __OUTPUT_FUNCTIONS_H__
#define __OUTPUT_FUNCTIONS_H__

__global__ void copy_frame(uchar4*, unsigned char*);
int save_image(uchar4 *outputBitmap, size_t size, unsigned blockSize,
               char *prefix, unsigned time, unsigned maxT, int dev)
{
    dim3 blocks(NBLOCKS(size, blockSize), NBLOCKS(size, blockSize));
    dim3 threads(blockSize, blockSize);
    unsigned digMax = num_digits(maxT), dig = num_digits(time), i;
    char fname[150] = { '\0' };
    tjhandle tjInstance = NULL;
    unsigned long jpegSize = 0;
    unsigned char *jpegBuf = NULL;
    FILE *jpegFile = NULL;
    unsigned char *frame = NULL;

    if (prefix != NULL) strcat(fname, prefix);
    for (i = 0; i < digMax-dig; i++) strcat(fname, "0");
    sprintf(&fname[strlen(fname)], "%d.jpeg", time);

    CudaSafeCall(cudaMallocManaged((void**)&frame,
                                   size*size*4*sizeof(unsigned char)));
    CudaSafeCall(cudaMemPrefetchAsync(frame, size*size*4*sizeof(unsigned char),
                                      dev, NULL));
    copy_frame<<< blocks, threads >>>(outputBitmap, frame);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

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

void save_video(char *prefix, char *outputName,
                unsigned framerate, unsigned maxT)
{
    char command[250] = { '\0' };
    unsigned numDigMaxT = num_digits(maxT);

    sprintf(command, "ffmpeg -y -v quiet -framerate %d -start_number 0 -i ",
            framerate);
    if (prefix != NULL) strcat(command, prefix);
    if (numDigMaxT == 1) strcat(command, "%%d.jpeg");
    else sprintf(&command[strlen(command)], "%%%d%dd.jpeg", 0, numDigMaxT);
    sprintf(&command[strlen(command)], " -c:v libx264 -pix_fmt yuv420p %s",
            outputName);

    system(command);
}

#endif // __OUTPUT_FUNCTIONS_H__