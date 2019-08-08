#ifndef __ANIM_FUNCTIONS_H__
#define __ANIM_FUNCTIONS_H__

void anim_gpu_ca(uchar4* outputBitmap, DataBlock *d, unsigned int ticks) {
	set_seed();

	dim3 blocks(d->grid_size / BLOCK_SIZE, d->grid_size / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if (ticks <= d->maxT) {
		if (ticks == 0 && (bitmap.display == 1 || d->save_frames == 1)) {
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

			mutate_grid<<< blocks, threads >>>(d->prevGrid, d->grid_size, ticks, d->pdes, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

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

			check_lifespan<<< blocks, threads >>>(d->prevGrid, d->newGrid, d->grid_size, states);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			reset_rule_params<<< blocks, threads >>>(d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			cells_gpu_to_gpu_copy<<< blocks, threads >>>(d->newGrid, d->prevGrid, d->grid_size);
			CudaCheckError();
			CudaSafeCall(cudaDeviceSynchronize());

			for (int i = 0; i < NUM_CARCIN; i++) d->pdes[i].time_step(ticks);

			CudaSafeCall(cudaFree(states));

			if ((bitmap.display == 1 || d->save_frames == 1) && ticks % d->frame_rate == 0) {
				display_ca<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
				CudaCheckError();
				CudaSafeCall(cudaDeviceSynchronize());
			}
		}

		if (d->save_frames == 1 && (!bitmap.paused || (bitmap.paused && bitmap.windowsShouldClose)))
			save_image(outputBitmap, d->dim, NULL, ticks, d->maxT);
	}
}

void anim_gpu_genes(uchar4* outputBitmap, DataBlock *d, unsigned int ticks) {
	dim3 blocks(d->grid_size/BLOCK_SIZE, d->grid_size/BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if ((bitmap.display == 1 || d->save_frames == 1) && ticks % d->frame_rate == 0) {
		display_genes<<< blocks, threads >>>(outputBitmap, d->newGrid, d->grid_size, d->cell_size, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}

	if (d->save_frames == 1 && ((ticks <= d->maxT && !bitmap.paused) || (bitmap.paused && bitmap.windowsShouldClose)))
		save_image(outputBitmap, d->dim, prefixes[0], ticks, d->maxT);
}

void anim_gpu_carcin(uchar4* outputBitmap, DataBlock *d, unsigned int carcin_idx, unsigned int ticks) {
	dim3 blocks(d->grid_size/BLOCK_SIZE, d->grid_size/BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if ((bitmap.display == 1 || d->save_frames == 1) && ticks % d->frame_rate == 0) {
		display_carcin<<< blocks, threads >>>(outputBitmap, &d->pdes[carcin_idx], d->grid_size, d->cell_size, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}

	if (d->save_frames == 1 && ((ticks <= d->maxT && !bitmap.paused) || (bitmap.paused && bitmap.windowsShouldClose)))
		save_image(outputBitmap, d->dim, prefixes[carcin_idx+1], ticks, d->maxT);

}

void anim_gpu_cell(uchar4* outputBitmap, DataBlock *d, unsigned int cell_idx, unsigned int ticks) {
	dim3 blocks(d->dim/BLOCK_SIZE, d->dim/BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	if ((bitmap.display == 1 || d->save_frames == 1) && ticks % d->frame_rate == 0) {
		display_cell_data<<< blocks, threads >>>(outputBitmap, d->newGrid, cell_idx, d->dim);
		CudaCheckError();
		CudaSafeCall(cudaDeviceSynchronize());
	}
}

void anim_gpu_timer_and_saver(DataBlock *d, bool start, int ticks) {
	if (start) {
		d->start_step = clock();
		printf("starting %d\n", ticks);
	} else {
		printf("done %d\n", ticks);
		d->end_step = clock();
		printf("The time step took %f seconds to complete.\n", (double) (d->end_step - d->start_step) / CLOCKS_PER_SEC);
	}
	if (ticks == 0) d->start = clock();

	if (!start && d->save_frames == 1 && ((ticks == d->maxT && !bitmap.paused) || (bitmap.paused && bitmap.windowsShouldClose))) {
		#pragma omp parallel sections num_threads(3)
		{
			#pragma omp section
			{
				printf("Saving video %s.\n", out_names[0]);
				save_video(NULL, out_names[0], VIDEO_FRAMERATE, d->maxT);
				printf("Finished video %s.\n", out_names[0]);
			}
			#pragma omp section
			{
				printf("Saving video %s.\n", out_names[1]);
				save_video(prefixes[0], out_names[1], VIDEO_FRAMERATE, d->maxT);
				printf("Finished video %s.\n", out_names[1]);
			}
			#pragma omp section
			{
				for (int carcin_idx = 0; carcin_idx < NUM_CARCIN; carcin_idx++) {
					printf("Saving video %s.\n", out_names[carcin_idx+2]);
					save_video(prefixes[carcin_idx+1], out_names[carcin_idx+2], VIDEO_FRAMERATE, d->maxT);
					printf("Finished video %s.\n", out_names[carcin_idx+2]);
				}
			}
		}

	}

	if (ticks == d->maxT && !start) {
		d->end = clock();
		printf("It took %f seconds to run the %d time steps.\n", (double) (d->end - d->start) / CLOCKS_PER_SEC, d->maxT);
	}
}

void anim_exit(DataBlock *d) {
	CudaSafeCall(cudaDeviceSynchronize());
	for (int i = 0; i < 3; i++) bitmap.hide_window(bitmap.windows[i]);
	double start = omp_get_wtime();
	int nt = omp_get_num_procs();
	int counts[nt] = { 0 };
	printf("Grid freeing progress:   0.00/100.00");
	for (int i = 0; i < d->grid_size; i++)
		for (int j = 0; j < d->grid_size; j+=2) {
			#pragma omp parallel sections num_threads(2)
			{
				#pragma omp section
				{
					d->prevGrid[i*d->grid_size+j].free_resources();

					d->newGrid[i*d->grid_size+j].free_resources();
					counts[omp_get_thread_num()]++;
				}

				#pragma omp section
				{
					d->prevGrid[i*d->grid_size+(j+1)].free_resources();

					d->newGrid[i*d->grid_size+(j+1)].free_resources();
					counts[omp_get_thread_num()]++;
				}
			}

			int num_finished = 0;
			for (int k = 0; k < nt; k++) num_finished += counts[k];
			print_progress(num_finished, d->grid_size*d->grid_size);
		}
	printf("\n");
	printf("It took %f seconds to finish freeing the memory.\n", omp_get_wtime() - start);
	CudaSafeCall(cudaFree(d->prevGrid));
	CudaSafeCall(cudaFree(d->newGrid));
	for (int i = 0; i < NUM_CARCIN; i++)
		d->pdes[i].free_resources();
	CudaSafeCall(cudaFree(d->pdes));
}

#endif // __ANIM_FUNCTIONS_H__
