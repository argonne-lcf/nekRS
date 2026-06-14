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
  uint wsize = box->sn;
  box->sx = malloc(sizeof(double) * 2 * wsize);
  box->srhs = (void *)((double *)box->sx + wsize);
  buffer_init(&box->bfr, 1024);
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
  // Setup ASM2.
  // Allocate work arrays.
  allocate_work_arrays(box);

  return box;
}

void crs_box_solve2(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  buffer *bfr = &box->bfr;
  const gs_dom dom = jl_dom_to_gs_dom(box->opts.dom);

  o_rhs.copyTo(box->srhs, box->un);
  gs(box->srhs, dom, gs_add, 0, box->gsh, bfr);
  crs_xxt_solve(box->sx, (struct xxt *)box->asm1, box->srhs);
  gs(box->sx, dom, gs_add, 0, box->gsh, bfr);
  o_x.copyFrom(box->sx, box->un);
}

void crs_box_free2(struct box *box) {
  if (!box) return;
  crs_xxt_free((struct xxt *)box->asm1);
  gs_free(box->gsh);
  free(box->sx), free(box->srhs);
  buffer_free(&box->bfr);
  comm_free(&box->c);
}
