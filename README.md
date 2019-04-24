# HybridCAFieldCancerization

The parameters are set such that the only carcinogen is alcohol and there are 10 genes that are allowed to mutate. The parameters are tuned towards the formation of head and neck carcinomas. 

# Notes
1. This program only currently supports Ubuntu and requires one NVIDIA gpu to run.
2. The minimum grid size with the default BLOCK_SIZE (in src/common/general.h) is 16.
3. The stability of the program has only been checked up to a time step of 800 and grid size of 512.
4. If you want to consider more carcinogens you need to update the main.cu file and change the appropriate variables.
5. If you want to consider different genes you will have to edit the weight and output matrices in mutation_nn.h.

# Compile
To compile the program run the bin/compile.sh bash script in the directory you want the program.

# Running Simulations
To run 100 simulations with the default grid size of 512 and number of time steps 600 run the bin/simulations.sh script
in the directory you want the output of the program to be.

# GUI Instructions
1. If the GUI is active the simulations are initially paused.
2. When any window is the currently active one pressing the space key pauses the simulation.
3. When any window is the currently active one pressing the escape key ends the simulation.
4. While the CA window is the currently active one:
   a) Press the right arrow key to step through the simulation, hold to keep it going.
      Note this functionality only really makes sense from a paused state.
   b) Press the k key to set tumour excision mode on. While this is active all tumor cells and their neighbours are killed.
      Note that the key k acts as a toggle switch so to stop tumour excision one must hit the key again.
      Also if the simulation is currently paused tumour excision won't occur to it is unpaused and will occur at the
      end of the timestep. 
5. While the Cell window is the currently active one:
   a) The directional arrows are used to control what cell is currently being viewed.
   b) If the simulation is paused pressing the s key saves the current screen as a picture.
4. While the carcinogen window is the currently active one:
   a) The right and left arrow keys are used to go through the different carcinogens.
