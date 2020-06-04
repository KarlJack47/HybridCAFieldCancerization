# HybridCAFieldCancerization

The parameters are set such that the only carcinogen is alcohol and there are 10
genes that are allowed to mutate. The parameters are tuned towards the formation
of tongue carcinomas.

# Notes
1. This program only currently supports Ubuntu and requires one NVIDIA gpu to run.
2. The default gpu block size is 16.
3. The stability of the program has only been checked up to a time step of 10966 and grid size of 1024.
4. If you want to consider more carcinogens you need to update the main.cu file and change the appropriate variables.
5. If you want to consider different genes you will have to edit the appropriate variables.

# Compile
To compile the program run the bin/compile.sh bash script in the directory you want the program.

# Running Simulations
To run 100 simulations with the default grid size of 256 and number of time steps 8766 run the bin/create_results.sh script
in the directory you want the output of the program to be.

# GUI Instructions
1. If the GUI is active the simulations are initially paused.
2. When any window is the currently active one pressing the space key pauses the simulation.
3. When any window is the currently active one pressing the escape key ends the simulation.
4. Initially there is only one window, and it initially shows the CA window. You can cycle through the different types of
   windows using the 'A' and 'D' keys.
5. To detach a window from the main windows press 'X'. To reattach the window press 'X' again when that window is active.
6. While the CA window is the currently active one:
   1. Press the right arrow key to step through the simulation, hold to keep it going.
      Note this functionality only really makes sense from a paused state.
   2. Press the 'K' key to set tumour excision mode on. While this is active all tumor cells and their neighbours are 
      killed. Note that the key 'K' acts as a toggle switch so to stop tumour excision one must hit the key again.
      Also if the simulation is currently paused tumour excision won't occur till it is unpaused and will occur at the end
      of the timestep.  
7. When the Gene Families is the current active one:
   1. Press the right arrow key to step through the simulation.
8. While the Cell window is the currently active one:
   1. The directional arrows are used to control what cell is currently being viewed.
   2. If the simulation is paused pressing the 'S' key saves the current screen as a picture.
   3. Clicking on a cell in the CA window will cause that to be the selected cell.
9. While the carcinogen window is the currently active one:
   1. The right and left arrow keys are used to go through the different carcinogens.