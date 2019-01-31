# HybridCAFieldCancerization

The parameters are set such that the only carcinogen is alcohol and there are 10 genes that are allowed to mutate. The parameters are tuned towards the formation of head and neck carcinomas. 

# Notes
1. This program only currently supports Ubuntu and requires one NVIDIA gpu to run.
2. The minimum grid size with the default BLOCK_SIZE (in src/common/general.h) is 16.
3. The stability of the program has only been checked up to a time step of 200 and grid size of 512.
4. If you want to consider more carcinogens you need to update the main.cu file and change the appropriate variables.
5. If you want to consider different genes you will have to edit the weight and output matrices in mutation_nn.h.

# Compile
To compile the program run the bin/compile.sh bash script in the directory you want the program.

# Running Simulations
To run 100 simulations with the default grid size of 512 and number of time steps 200 run the bin/simulations.sh script
in the directory you want the output of the program to be.
