#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include "crs.hpp"

struct box {
  jl_opts opts;          /* User configurable options */
  uint un, cn, sn;       /* User size, compressed size, Schwarz size */
  uint ncr;              /* Number of 1D dofs */
  void *sx, *srhs, *sim; /* Schwarz/XXT work arrays on CPU */
  void *asm1;            /* Pointer to the asm1 solver */
  void *asm2;            /* Pointer to the asm2 solver */
  struct gs_data *ras;   /* RAS */
  struct comm c;         /* communicators */
  buffer bfr;            /* Buffers for gslib */
};

struct box *crs_asm1_setup(uint n, const ulong *id, uint nnz, const uint *Ai,
                           const uint *Aj, const double *A, const jl_opts *opts,
                           const struct comm *comm);
void crs_asm1_solve(occa::memory &o_x, struct box *data, occa::memory &o_rhs);
void crs_asm1_free(struct box *data);

void crs_box_setup_asm1(struct box *box);

#endif
