#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include "crs.hpp"

struct box {
  uint ne, nd, nv;       /* User input */
  uint nu, ns, cn;       /* User size, Schwarz size, compressed size */
  real *sx, *srhs, *sim; /* Schwarz/XXT work arrays on CPU */
  struct xxt *asm1;      /* Pointer to the asm1 solver */
  struct xxt *asm2;      /* Pointer to the asm2 solver */
  struct gs_data *ras;   /* RAS */
  buffer bfr;
};

void crs_overlap(uint *nei, uint nd, uint nv, slong *vids, double *xyz,
                 double *mat, sint *frontier, uint nw, sint *wids,
                 MPI_Comm comm, uint max_ne, uint dbg);

/* ASM1 */
void crs_asm1_setup(slong *const id, const sint *const frontier,
                    const double *const A, const struct comm *const c,
                    struct box *const box);
void crs_asm1_free(struct box *const box);

/* ASM2 */
struct xxt *crs_asm2_setup(const double *const xyz,
                           const double *const centroid, const double *const A,
                           const uint nbx, const uint nby, const uint nbz,
                           const struct comm *const c, struct box *const box);
void crs_asm2_free(struct box *const box);

/* BOX */
struct box *crs_box_setup(const uint n, const ulong *const id, const uint nnz,
                          const uint *const Ai, const uint *const Aj,
                          const double *const A, const double *const xyz,
                          const double *const centroid,
                          const jl_opts *const opts,
                          const struct comm *const c);
void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs);
void crs_box_free(struct box *box);

#endif
