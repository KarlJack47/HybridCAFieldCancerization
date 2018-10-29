#include "common/general.h"
#include "ca.h"

int main(int argc, char *argv[]) {
	int grid_size; int T; int save_frames; int display;
	if (argc == 1) {
		display = 1;
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

	CA ca(grid_size, T, 1, 11, save_frames, display);
	ca.initialize_memory();
	float carcinogen_mutation_map[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	double diffusion[1] = {1.266389e-5};
	bool liquid[1] = {true};
	ca.init(carcinogen_mutation_map, diffusion, liquid);

	end = clock();
	printf("It took %f seconds to initialize the memory.\n", (double) (end - start) / CLOCKS_PER_SEC); 

	ca.animate(1);

	return 0;
}
