# Files
1. cell.h: header file that includes the Cell 
   structure definition and related functions.
2. gene_expression_nn.h: header file that includes the
   GeneExpressionNN structure definition and related functions.
3. carcinogen_pde.h: header file that includes the CarcinogenPDE
   structure definition and related functions.
4. ca.h: header file that includes the CA structure definition and
   related functions.
5. cuda_kernels.h: header file that includes all the cuda kernels used for the simulations.
6. gpu_anim.h: header file that includes the GPUAnimBitmap structure
   definition. This structure is used to create and run the CA animations.
7. anim_functions.h: header file that includes all the functions related to running and animating the simulations.
8. main.cu: allows you to create a program that allows the creation
   of a CA with an arbitary grid size (>=32) and arbitary number of
   time steps.
9. common/general.h: header file that includes all the needed libraries
   for the above header files and main.cu. It also defines some general
   functions used by the header files above.
10. common/error_check.h: defines the inline functions used for cuda error
   checking.  
11. common/glad/glad.h, common/glad.c, common/glad.o: used by GLFW to acquire the addresses of the OpenGL functions.
