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
	CA ca(grid_size, T, 1, 11, save_frames, display);
	ca.initialize_memory();

	double diffusion[1] = {1.266389e-5};
	double consum[1] = {9.722222e-8 / (double) (grid_size * grid_size)};
	double in[1] = {2.37268e-6 / (double) (grid_size * grid_size)};
	bool liquid[1] = {true};
	ca.init(diffusion, consum, in, liquid);

	end = clock();
	printf("It took %f seconds to initialize the memory.\n", (double) (end - start) / CLOCKS_PER_SEC); 

	ca.animate(1, carcin_names);

	free(carcin_names[0]); free(carcin_names);

	return 0;
}
