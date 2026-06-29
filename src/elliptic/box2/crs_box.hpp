#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include "crs.hpp"

struct box {
  jl_opts opts; /* User configurable options */
  void *asm1;   /* Pointer to the asm1 solver */
  void *asm2;   /* Pointer to the asm2 solver */
};

void *crs_asm1_setup(uint n, const ulong *const id, uint nnz,
                     const uint *const Ai, const uint *const Aj,
                     const double *const A, const double *const xyz_,
                     const uint nw, const struct comm *const c);
void crs_asm1_solve(occa::memory &o_x, void *solver, occa::memory &o_rhs);
void crs_asm1_free(void *solver);

struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai,
                          const uint *Aj, const double *A, const double *xyz,
                          const jl_opts *opts, const struct comm *c);
void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs);
void crs_box_free(struct box *box);

#endif
