#include "common/general.h"

int main(int argc, char *argv[]) {
	int display, save_frames, T, grid_size, maxt_tc;
	if (argc == 1) {
		display = 0;
		save_frames = 1;
		T = 200;
		grid_size = 512;
		maxt_tc = -1;
	} else if (argc == 2) {
		display = atoi(argv[1]);
		save_frames = 1;
		T = 200;
		grid_size = 512;
		maxt_tc = -1;
	} else if (argc == 3) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = 200;
		grid_size = 512;
		maxt_tc = -1;
	} else if (argc == 4) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = atoi(argv[3]);
		grid_size = 512;
		maxt_tc = -1;
	} else if (argc == 5) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = atoi(argv[3]);
		grid_size = atoi(argv[4]);
		maxt_tc = -1;
	} else if (argc == 6) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = atoi(argv[3]);
		grid_size = atoi(argv[4]);
		maxt_tc = atoi(argv[5]);
	}

	clock_t start, end;
	start = clock();

	char **carcin_names = (char**)malloc(sizeof(char*));
	carcin_names[0] = (char*)malloc(8*sizeof(char));
	sprintf(carcin_names[0], "%s", "Alcohol");
	CA ca(grid_size, T, save_frames, display, maxt_tc, carcin_names);
	free(carcin_names[0]); free(carcin_names);
	ca.initialize_memory();

	double diffusion[NUM_CARCIN] = {1.266389e-5};
	double out[NUM_CARCIN] = {9.722222e-8 / (double) (grid_size * grid_size)};
	double in[NUM_CARCIN] = {2.37268e-6 / (double) (grid_size * grid_size)};
	double ic[NUM_CARCIN] = {0.5f};
	double bc[NUM_CARCIN] = {1.0f};

	double *W_x = (double*)malloc(NUM_GENES*(NUM_CARCIN+1)*sizeof(double));
	double *W_y = (double*)malloc(NUM_GENES*NUM_GENES*sizeof(double));
	double *b_y = (double*)malloc(NUM_GENES*sizeof(double));
	
	memset(W_x, 0.0f, NUM_GENES*(NUM_CARCIN+1)*sizeof(double));

	for (int i = 0; i < NUM_GENES; i++) {
		for (int j = 0; j < NUM_CARCIN+1; j++) {
			if (j != NUM_CARCIN)
				W_x[i*(NUM_CARCIN+1)+j] = carcinogen_mutation_map[j*NUM_CARCIN+i];
			else W_x[i*(NUM_CARCIN+1)+j] = 0.00001f;
		}
	}

	W_y[0] = 1.0f;
	W_y[NUM_GENES+1] = 0.1f;
	W_y[2*NUM_GENES+2] = 0.3f;
	W_y[3*NUM_GENES+3] = 0.1f;
	W_y[4*NUM_GENES+4] = 0.1f;
	W_y[5*NUM_GENES+5] = 0.1f;
	W_y[6*NUM_GENES+6] = 0.2f;
	W_y[7*NUM_GENES+7] = 0.3f;
	W_y[8*NUM_GENES+8] = 0.1f;
	W_y[9*NUM_GENES+9] = 0.3f;
	for (int i = 1; i < NUM_GENES; i++) W_y[i*NUM_GENES] = 0.01f;
	W_y[2*NUM_GENES] = 0.01f;
	W_y[NUM_GENES+2] = 0.01f;
	W_y[6*NUM_GENES+2] = 0.01f;
	W_y[3*NUM_GENES+6] = 0.01f;
	W_y[3*NUM_GENES+7] = -0.01f;
	W_y[9*NUM_GENES+7] = 0.01f;
	W_y[6*NUM_GENES+9] = 0.01f;
	W_y[7*NUM_GENES+9] = 0.01f;

	memset(b_y, 0.0f, NUM_GENES*sizeof(double));

	ca.init(diffusion, out, in, ic, bc, W_x, W_y, b_y);

	free(W_x);
	free(W_y);
	free(b_y);

	end = clock();
	printf("It took %f seconds to initialize the memory.\n", (double) (end - start) / CLOCKS_PER_SEC); 

	ca.animate(1);

	return 0;
}
