#if !defined(_CRS_XXT_HPP_)
#define _CRS_XXT_HPP_

#include "nekrs_crs.hpp"

struct xxt;
struct xxt *crs_xxt_setup(uint n, const ulong *id, uint nz, const uint *Ai,
                          const uint *Aj, const double *A, gs_dom dom,
                          uint null_space, const struct comm *comm);
void crs_xxt_solve(void *x, struct xxt *data, const void *b);
void crs_xxt_stats(struct xxt *data);
void crs_xxt_times(double *cholesky, double *local, double *xxt, double *qqt);
void crs_xxt_free(struct xxt *data);

#endif
