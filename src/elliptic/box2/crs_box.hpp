#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include "nekrs_crs.hpp"

struct box;
struct box *crs_box_setup2(uint n, const ulong *id, uint nnz, const uint *Ai,
                           const uint *Aj, const double *A, const jl_opts *opts,
                           const struct comm *comm);
void crs_box_solve2(occa::memory &o_x, struct box *data, occa::memory &o_rhs);
void crs_box_free2(struct box *data);

void fetch_nbrs_v3(unsigned *nei, slong *vids, double *mat, sint *wids,
                   unsigned nv, unsigned ndim, unsigned nw, unsigned max_ne,
                   MPI_Comm comm);

static inline gs_dom jl_dom_to_gs_dom(jl_dom_t dom) {
  switch (dom) {
  case jl_float32: return gs_float; break;
  case jl_float64: return gs_double; break;
  default: break;
  }
}

#endif
