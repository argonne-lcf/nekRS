#if !defined(_NEKRS_CRS_HPP_)
#define _NEKRS_CRS_HPP_

#include <elliptic.h>
#include <gslib.h>
#include <platform.hpp>

typedef enum { XXT = 0, BOX } jl_algo_t;

typedef struct {
  gs_dom dom;
  jl_algo_t algo;
  /* Does the system has a null space? */
  unsigned null_space;
  /* Size of overlap */
  unsigned nw;
} jl_opts;

void jl_setup_aux(uint *ntot, ulong **gids, uint *nnz, uint **ia, uint **ja,
                  double **a, elliptic_t *elliptic, elliptic_t *ellipticf);

void jl_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
              const double *A, const jl_opts *opts, MPI_Comm comm);

void jl_solve2(occa::memory &o_x, occa::memory &o_rhs);

void jl_free();

#endif
