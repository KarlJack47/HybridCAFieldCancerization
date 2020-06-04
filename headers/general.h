#ifndef __GENERAL_H__
#define __GENERAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "error_check.h"
#include <errno.h>
#include <turbojpeg.h>

#define THROW(action, message) printf("ERROR in line %d while %s:\n%s\n",\
                                      __LINE__, action, message)
#define THROW_TJ(action) THROW(action, tjGetErrorStr())
#define THROW_UNIX(action) THROW(action, strerror(errno))

#define NBLOCKS(n, nThreads) \
    (n % nThreads == 0 ? n / nThreads : n / nThreads + 1)

typedef enum { POS=1, NEG=-1, NONE=0 } effect;
typedef enum { CA_GRID, GENE_INFO, CELL_INFO, CARCIN_GRID } gui_window;
typedef enum { NC, MNC, SC, MSC, CSC, TC, EMPTY, ERR=-1 } ca_state;
typedef enum { PROLIF, QUIES, APOP, DIFF } cell_phenotype;
typedef enum { TP53, TP73, RB, P21, TP16, EGFR, CCDN1, MYC, PIK3CA, RAS } gene;
typedef enum { SUPPR, ONCO } gene_type;
typedef enum { NO, YES } gene_related;
typedef enum { NORTH, EAST, SOUTH, WEST,
               NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST } neigh_location;

#include "array_functions.h"
#include "rnd_gen_functions.h"
#include "utility.h"

#include "gene_expr_nn.h"
#include "cell.h"
__global__ void init_pde(double*,double,double,unsigned,double);
__global__ void pde_space_step(double*,unsigned,unsigned,unsigned,double,
                               double,double,double,double,double);
#include "carcin_pde.h"
#include "cuda_kernels.h"
#include "output_functions.h"
#include "gui.h"
struct CA;
void anim_gpu_ca(uchar4*,unsigned,CA*,unsigned,bool,bool,bool,bool,cudaStream_t);
void anim_gpu_genes(uchar4*,unsigned,CA*,unsigned,bool,bool,bool,cudaStream_t);
void anim_gpu_carcin(uchar4*,unsigned,CA*,unsigned,unsigned,bool,bool,bool,cudaStream_t);
void anim_gpu_cell(uchar4*,unsigned,CA*,unsigned,unsigned,bool,cudaStream_t);
void anim_gpu_timer_and_saver(CA*,bool,unsigned,bool,bool);
#include "ca.h"
#include "anim_functions.h"

#endif // __GENERAL_H__