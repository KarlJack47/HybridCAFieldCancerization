#ifndef __ANIM_FUNCTIONS_H__
#define __ANIM_FUNCTIONS_H__

void anim_gpu_ca(uchar4* outputBitmap, DataBlock *d, unsigned int ticks) {
	set_seed();

	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks <= d->maxT) {
		if (ticks == 0) {
			display_ca<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
		} else if (!bitmap.paused) {
			curandState_t *states;
			CudaSafeCall(cudaMalloc((void**)&states, d->grid_size*d->grid_size*sizeof(curandState_t)));
			timespec seed;
        		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &seed);
			init_curand<<< d->grid_size*d->grid_size, 1 >>>(seed.tv_nsec, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			prefetch_pdes(d->dev_id_2, -1);

			mutate_grid<<< blocks, threads >>>(d->prevGrid, d->grid_size, ticks, d->pdes, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			prefetch_pdes(-1, -1);

			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->prevGrid, d->newGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			bool used_pheno[NUM_PHENO] = { false, false, false, false };
			while (used_pheno[PROLIF] == false && used_pheno[QUIES] == false && used_pheno[APOP] == false && used_pheno[DIFF] == false) {
				unsigned int pheno = rand() % NUM_PHENO;
				if (used_pheno[pheno] == false) {
					rule<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size, pheno, states);
					CudaCheckError();
					CudaSafeCall(cudaDeviceSynchronize());

					if (pheno == QUIES) {
						cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
						CudaCheckError();
						CudaSafeCall(cudaDeviceSynchronize());
					}

					used_pheno[pheno] = true;
				}
			}

			check_CSC_or_TC_formed<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size, ticks);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			if (tc_formed[excise_count] == true) time_tc_alive++;
			else time_tc_dead++;

			reset_rule_params<<< blocks, threads >>>(d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			update_states<<< blocks, threads >>>(d->newGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			if (excise_count <= MAX_EXCISE && tc_formed[excise_count] == true) {
				if (bitmap.excise == true || time_tc_alive == d->maxt_tc_alive+1) {
					tumour_excision<<< blocks, threads >>>(d->newGrid, d->grid_size);
					CudaCheckError();
					CudaSafeCall(cudaDeviceSynchronize());
					printf("Tumour excision was performed at time step %d.\n", ticks);
					if (excise_count == MAX_EXCISE) printf("The maximum number of tumour excisions has been performed.\n");
					excise_count++;
					time_tc_alive = 0;
					time_tc_dead = 1;
				}
			}

			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			for (int i = 0; i < NUM_CARCIN; i++) d->pdes[i].time_step(ticks, d->newGrid);

			CudaSafeCall(cudaFree(states));
		}

		if (ticks % d->frame_rate == 0) {
			display_ca<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());
		}
	}

	if (!bitmap.paused && d->save_frames == 1 && ticks <= d->maxT) {
		char fname[14] = { '\0' };
		int dig_max = numDigits(d->maxT);
		int dig = numDigits(ticks);
		for (int i = 0; i < dig_max-dig; i++) fname[i] = '0';
		sprintf(&fname[dig_max-dig], "%d.png", ticks);
		unsigned char *frame;
		CudaSafeCall(cudaMallocManaged((void**)&frame, d->dim*d->dim*4*sizeof(unsigned char)));
		CudaSafeCall(cudaMemPrefetchAsync(frame, d->dim*d->dim*4*sizeof(unsigned char), 1, NULL));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>( outputBitmap, frame );
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

		CudaSafeCall(cudaFree(frame));
	}

	if (d->save_frames == 1 && ((!bitmap.paused && ticks == d->maxT) || (bitmap.paused && bitmap.windowsShouldClose))) {
		int numDigMaxT = numDigits(d->maxT);
		char command[107] = { '\0' };
		strcat(command, "ffmpeg -y -v quiet -framerate 5 -start_number 0 -i ");
		if (numDigMaxT == 1) strcat(command, "%d.png");
		else sprintf(&command[strlen(command)], "%%%d%dd.png", 0, numDigMaxT);
		strcat(command, " -c:v libx264 -pix_fmt yuv420p out_ca.mp4");
		system(command);
	}
}

void anim_gpu_genes(uchar4* outputBitmap, DataBlock *d, unsigned int ticks) {
	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks % d->frame_rate == 0) {
		display_genes<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}

	if (!bitmap.paused && d->save_frames == 1 && ticks <= d->maxT) {
		char fname[25] = { '\0' };
		strcat(fname, "genes_");
		int dig_max = numDigits(d->maxT); int dig = numDigits(ticks);
		for (int i = 0; i < dig_max-dig; i++) strcat(fname, "0");
		sprintf(&fname[strlen(fname)], "%d.png", ticks);
		unsigned char *frame;
		CudaSafeCall(cudaMallocManaged((void**)&frame, d->dim*d->dim*4*sizeof(unsigned char)));
		CudaSafeCall(cudaMemPrefetchAsync(frame, d->dim*d->dim*4*sizeof(unsigned char), 1, NULL));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>(outputBitmap, frame);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

		CudaSafeCall(cudaFree(frame));
	}

	if (d->save_frames == 1 && ((!bitmap.paused && ticks == d->maxT) || (bitmap.paused && bitmap.windowsShouldClose))) {
		char command[250] = { '\0' };
		strcat(command, "ffmpeg -y -v quiet -framerate 5 -start_number 0 -i ");
		int numDigMaxT = numDigits(d->maxT);
		if (numDigMaxT == 1) strcat(command, "genes_%%d.png");
		else sprintf(&command[strlen(command)], "genes_%%%d%dd.png", 0, numDigMaxT);
		strcat(command, " -c:v libx264 -pix_fmt yuv420p ");
		strcat(command, "out_genes.mp4");
		system(command);
	}
}

void anim_gpu_carcin(uchar4* outputBitmap, DataBlock *d, unsigned int carcin_idx, unsigned int ticks) {
	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks % d->frame_rate == 0) {
		prefetch_pdes(d->dev_id_2, carcin_idx);

		display_carcin<<< blocks, threads >>>(outputBitmap, &d->pdes[carcin_idx], d->grid_size, d->cell_size, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		prefetch_pdes(-1, carcin_idx);
	}

	if (!bitmap.paused && d->save_frames == 1 && ticks <= d->maxT) {
		char fname[25] = { '\0' };
		sprintf(&fname[strlen(fname)], "carcin%d_", carcin_idx);
		int dig_max = numDigits(d->maxT); int dig = numDigits(ticks);
		for (int i = 0; i < dig_max-dig; i++) strcat(fname, "0");
		sprintf(&fname[strlen(fname)], "%d.png", ticks);
		unsigned char *frame;
		CudaSafeCall(cudaMallocManaged((void**)&frame, d->dim*d->dim*4*sizeof(unsigned char)));
		CudaSafeCall(cudaMemPrefetchAsync(frame, d->dim*d->dim*4*sizeof(unsigned char), 1, NULL));
		dim3 blocks1(d->dim/16, d->dim/16);
		dim3 threads1(16, 16);
		copy_frame<<< blocks1, threads1 >>>(outputBitmap, frame);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());

		unsigned error = lodepng_encode32_file(fname, frame, d->dim, d->dim);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

		CudaSafeCall(cudaFree(frame));
	}

	if (d->save_frames == 1 && ((!bitmap.paused && ticks == d->maxT) || (bitmap.paused && bitmap.windowsShouldClose))) {
		char command[250] = { '\0' };
		strcat(command, "ffmpeg -y -v quiet -framerate 5 -start_number 0 -i ");
		int numDigMaxT = numDigits(d->maxT);
		if (numDigMaxT == 1) sprintf(&command[strlen(command)], "carcin%d_%%d.png", carcin_idx);
		else sprintf(&command[strlen(command)], "carcin%d_%%%d%dd.png", carcin_idx, 0, numDigMaxT);
		strcat(command, " -c:v libx264 -pix_fmt yuv420p ");
		sprintf(&command[strlen(command)], "out_carcin%d.mp4", carcin_idx);
		system(command);
	}
}

void anim_gpu_cell(uchar4* outputBitmap, DataBlock *d, unsigned int cell_idx, unsigned int ticks) {
	dim3 blocks(d->dim / BLOCK_SIZE, d->dim / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks % d->frame_rate == 0) {
		display_cell_data<<< blocks, threads >>>(outputBitmap, d->newGrid, cell_idx, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}
}

void anim_gpu_timer(DataBlock *d, bool start, int ticks) {
	if (start) {
		d->start_step = clock();
		printf("starting %d\n", ticks);
	} else {
		printf("done %d\n", ticks);
		d->end_step = clock();
		printf("The time step took %f seconds to complete.\n", (double) (d->end_step - d->start_step) / CLOCKS_PER_SEC);
	}
	if (ticks == 0) d->start = clock();

	if (ticks == d->maxT && !start) {
		d->end = clock();
		printf("It took %f seconds to run the %d time steps.\n", (double) (d->end - d->start) / CLOCKS_PER_SEC, d->maxT);
	}
}

void anim_exit(DataBlock *d) {
	CudaSafeCall(cudaDeviceSynchronize());
	for (int i = 0; i < 3; i++) bitmap.hide_window(bitmap.windows[i]);
	printf("Freeing Grid Memory:   0.00/100.00");
	clock_t start = clock();
	#pragma omp parallel for collapse(2) schedule(guided)
			for (int i = 0; i < d->grid_size; i++) {
				for (int j = 0; j < d->grid_size; j++) {
					d->prevGrid[i*d->grid_size+j].free_resources();
					d->newGrid[i*d->grid_size+j].free_resources();

					print_progress(i, j, d->grid_size, d->grid_size*d->grid_size);
				}
			}
	printf("\n");
	printf("It took %f seconds to finish freeing the memory.\n", (double) (clock() - start) / CLOCKS_PER_SEC);
	CudaSafeCall(cudaFree(d->prevGrid));
	CudaSafeCall(cudaFree(d->newGrid));
	for (int i = 0; i < NUM_CARCIN; i++)
		d->pdes[i].free_resources();
	CudaSafeCall(cudaFree(d->pdes));
}

#endif // __ANIM_FUNCTIONS_H__

