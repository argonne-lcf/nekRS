#if !defined(_NEKRS_CRS_HPP_)
#define _NEKRS_CRS_HPP_

#include <elliptic.h>
#include <gslib.h>
#include <platform.hpp>

typedef enum {
  XXT = 0,
  BOX
} jl_algo_t;

typedef enum {
  BOX_XXT = 0,
  BOX_CHOLMOD,
  BOX_GPU
} box_algo_t;

typedef struct {
  jl_algo_t algo;
  gs_dom dom;
  box_algo_t asm1;
  unsigned mult;
  unsigned aggressive;
  unsigned null_space;
} jl_opts;

void jl_timer_init();

void jl_setup_aux(uint *ntot, ulong **gids, uint *nnz, uint **ia, uint **ja,
                  double **a, elliptic_t *elliptic, elliptic_t *ellipticf);

void jl_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
              const double *A, const jl_opts *opts, MPI_Comm comm);

void jl_solve(occa::memory &o_x, occa::memory &o_rhs);

void jl_solve2(occa::memory &o_x, occa::memory &o_rhs);

void jl_free();

void jl_timer_print(MPI_Comm comm);

#endif
