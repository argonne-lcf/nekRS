#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

#include <cassert>

#include <elliptic.h>
#include <gslib.h>
#include <platform.hpp>

#include "nekrs_crs.hpp"
#include "crs_box_timer.hpp"

struct xxt;
struct xxt *crs_xxt_setup(uint n, const ulong *id, uint nz, const uint *Ai,
                          const uint *Aj, const double *A, gs_dom dom,
                          uint null_space, const struct comm *comm);
void crs_xxt_solve(void *x, struct xxt *data, const void *b);
void crs_xxt_stats(struct xxt *data);
void crs_xxt_times(double *cholesky, double *local, double *xxt, double *qqt);
void crs_xxt_free(struct xxt *data);

struct box;
struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai,
                          const uint *Aj, const double *A, const jl_opts *opts,
                          const struct comm *comm);
void crs_box_solve(occa::memory &o_x, struct box *data, occa::memory &o_rhs);
void crs_box_free(struct box *data);

#endif
