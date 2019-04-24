#include "common/general.h"
#include "ca.h"

int main(int argc, char *argv[]) {
	int grid_size; int T; int save_frames; int display;
	if (argc == 1) {
		display = 0;
		save_frames = 1;
		T = 200;
		grid_size = 512;
	} else if (argc == 2) {
		display = atoi(argv[1]);
		save_frames = 1;
		T = 200;
		grid_size = 512;
	} else if (argc == 3) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = 200;
		grid_size = 512;
	} else if (argc == 4) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = atoi(argv[3]);
		grid_size = 512;
	} else if (argc == 5) {
		display = atoi(argv[1]);
		save_frames = atoi(argv[2]);
		T = atoi(argv[3]);
		grid_size = atoi(argv[4]);
	}

	clock_t start, end;
	start = clock();

	char **carcin_names = (char**)malloc(sizeof(char*));
	carcin_names[0] = (char*)malloc(8*sizeof(char));
	sprintf(carcin_names[0], "%s", "Alcohol");
	unsigned int n_input = 2;
	unsigned int n_hidden = 10;
	unsigned int n_output = 10;
	CA ca(grid_size, T, 1, n_output, save_frames, display, carcin_names);
	free(carcin_names[0]); free(carcin_names);
	ca.initialize_memory();

	double diffusion[1] = {1.266389e-5};
	double out[1] = {9.722222e-8 / (double) (grid_size * grid_size)};
	double in[1] = {2.37268e-6 / (double) (grid_size * grid_size)};

	double *W_x = (double*)malloc(n_hidden*n_input*sizeof(double));
	double *W_y = (double*)malloc(n_hidden*n_output*sizeof(double));
	double *b_y = (double*)malloc(n_output*sizeof(double));
	
	memset(W_x, 0.0f, n_hidden*n_input*sizeof(double));

	for (int i = 0; i < n_hidden; i++) {
		for (int j = 0; j < n_input; j++) {
			if (j != n_input-1)
				W_x[i*n_input+j] = carcinogen_mutation_map[j*(n_input-1)+i];
			else W_x[i*n_input+j] = 0.00001f;
		}
	}

	W_y[0] = 1.0f;
	W_y[n_output+1] = 0.1f;
	W_y[2*n_output+2] = 0.3f;
	W_y[3*n_output+3] = 0.1f;
	W_y[4*n_output+4] = 0.1f;
	W_y[5*n_output+5] = 0.1f;
	W_y[6*n_output+6] = 0.2f;
	W_y[7*n_output+7] = 0.3f;
	W_y[8*n_output+8] = 0.1f;
	W_y[9*n_output+9] = 0.3f;
	for (int i = 1; i < n_hidden; i++) W_y[i*n_output] = 0.01f;
	W_y[2*n_output] = 0.01f;
	W_y[n_output+2] = 0.01f;
	W_y[6*n_output+2] = 0.01f;
	W_y[3*n_output+6] = 0.01f;
	W_y[3*n_output+7] = -0.01f;
	W_y[9*n_output+7] = 0.01f;
	W_y[6*n_output+9] = 0.01f;
	W_y[7*n_output+9] = 0.01f;

	memset(b_y, 0.0f, n_output*sizeof(double));

	ca.init(diffusion, out, in, W_x, W_y, b_y);

	free(W_x);
	free(W_y);
	free(b_y);

	end = clock();
	printf("It took %f seconds to initialize the memory.\n", (double) (end - start) / CLOCKS_PER_SEC); 

	ca.animate(1);

	return 0;
}
