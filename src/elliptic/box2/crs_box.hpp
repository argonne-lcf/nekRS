#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include "nekrs_crs.hpp"

struct box {
  jl_opts opts;        /* User configurable options */
  uint un, cn, sn;     /* User size, compressed size, Schwarz size */
  uint ncr;            /* Number of 1D dofs */
  void *sx, *srhs;     /* Schwarz/XXT work arrays on CPU */
  void *asm1;          /* Pointer to the asm1 solver */
  void *asm2;          /* Pointer to the asm2 solver */
  struct gs_data *gsh; /* dssum */
  struct comm c;       /* communicators */
  buffer bfr;          /* Buffers for gslib */
};

struct box *crs_box_setup2(uint n, const ulong *id, uint nnz, const uint *Ai,
                           const uint *Aj, const double *A, const jl_opts *opts,
                           const struct comm *comm);
void crs_box_solve2(occa::memory &o_x, struct box *data, occa::memory &o_rhs);
void crs_box_free2(struct box *data);

void crs_box_setup_asm1(struct box *box);

static inline gs_dom jl_dom_to_gs_dom(jl_dom_t dom) {
  switch (dom) {
  case jl_float32: return gs_float; break;
  case jl_float64: return gs_double; break;
  default: break;
  }
}

#endif
