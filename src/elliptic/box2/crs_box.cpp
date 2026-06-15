#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include "crs_box.hpp"
#include "crs_xxt.hpp"

static inline void allocate_work_arrays(struct box *box) {
  box->sx = tcalloc(double, 2 * box->sn);
  box->srhs = (void *)((double *)box->sx + box->sn);
}

struct box *crs_box_setup2(uint n, const ulong *id, uint nnz, const uint *Ai,
                           const uint *Aj, const double *A, const jl_opts *opts,
                           const struct comm *c) {
  struct box *box = tcalloc(struct box, 1);
  box->un = n;
  box->ncr = nnz / n;
  box->opts = *opts;

  // Setup box members.
  buffer_init(&box->bfr, 1024);
  comm_dup(&box->c, c);

  // Setup ASM1.
  crs_box_setup_asm1(box);

  // Allocate work arrays.
  allocate_work_arrays(box);

  return box;
}

void crs_box_solve2(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  o_rhs.copyTo(box->srhs, box->un);
  gs(box->srhs, box->opts.dom, gs_add, 0, box->gsh, &box->bfr);

  crs_xxt_solve(box->sx, (struct xxt *)box->asm1, box->srhs);

  gs(box->sx, box->opts.dom, gs_add, 0, box->gsh, &box->bfr);
  o_x.copyFrom(box->sx, box->un);
}

void crs_box_free2(struct box *box) {
  if (!box) return;
  free(box->sx), free(box->srhs);
  gs_free(box->gsh);
  crs_xxt_free((struct xxt *)box->asm1), box->asm1 = 0;
  comm_free(&box->c);
  buffer_free(&box->bfr);
}
