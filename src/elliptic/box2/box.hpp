#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include "crs.hpp"

struct box {
  void *asm1;       /* Pointer to the asm1 solver */
  struct xxt *asm2; /* Pointer to the asm2 solver */
};

void *crs_asm1_setup(uint n, const ulong *const id, uint nnz,
                     const uint *const Ai, const uint *const Aj,
                     const double *const A, const double *const xyz_,
                     const uint nw, const struct comm *const c);
void crs_asm1_solve(occa::memory &o_x, void *solver, occa::memory &o_rhs);
void crs_asm1_free(void *solver);

struct xxt *crs_asm2_setup(const uint ne, const uint nd, const uint nv,
                           const double *const xyz,
                           const double *const centroid, const uint nbx,
                           const uint nby, const uint nbz,
                           const struct comm *const c);

struct box *crs_box_setup(const uint n, const ulong *const id, const uint nnz,
                          const uint *const Ai, const uint *const Aj,
                          const double *const A, const double *const xyz,
                          const double *const centroid,
                          const jl_opts *const opts,
                          const struct comm *const c);
void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs);
void crs_box_free(struct box *box);

#endif
