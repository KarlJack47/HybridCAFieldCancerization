# Files
1. cell.h: header file that includes the Cell structure definition and related
           functions.
2. gene_expr_nn.h: header file that includes the GeneExprNN structure definition
                   and related functions.
3. carcin.h: header file that includes the Carcin structure definition
             and related functions.
4. ca.h: header file that includes the CA structure definition and related
         functions.
5. cuda_kernels.h: header file that includes all the cuda kernels.
6. gui.h: header file that includes the GUI structure definition. This structure
          is used to create and run the CA animations.
7. anim_functions.h: header file that includes all the functions related to
                     running and animating the simulations.
8. output_functions.h: header file that includes functions related to output
                       including saving images and videos.
9. rnd_gen_functions.h: header file that includes functions related to random
                        number generation on the cpu and gpu.
10. utility.h: header file that includes useful general purpose functions.
11. error_check.h: defines the inline functions used for cuda error checking.
12. general.h: header file that includes all the needed libraries for the above
               header files and src/main.cu.