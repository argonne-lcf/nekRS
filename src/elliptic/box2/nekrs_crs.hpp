#if !defined(_NEKRS_CRS_HPP_)
#define _NEKRS_CRS_HPP_

#include <occa.hpp>

#include <gslib.h>

typedef enum { XXT = 0, BOX } jl_algo_t;
typedef enum { jl_float32 = 0, jl_float64 } jl_dom_t;

typedef struct {
  jl_dom_t dom;
  jl_algo_t algo;
  unsigned null_space;
  unsigned nw;
} jl_opts;

void jl_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
              const double *A, const jl_opts *opts, MPI_Comm comm);

void jl_solve2(occa::memory &o_x, occa::memory &o_rhs);

void jl_free();

#endif
