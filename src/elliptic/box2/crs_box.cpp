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

struct box {
  jl_opts opts; /* User configurable options */
  uint un, cn, sn,
      ncr;         /* User size, compressed size, Schwarz size and 1D dofs */
  void *sx, *srhs; /* Schwarz/XXT work arrays on CPU */
  void *asm1;      /* Pointer to the asm1 solver */
  void *asm2;      /* Pointer to the asm2 solver */
  struct gs_data *gsh; /* dssum */
  struct comm global;  /* communicators */
  buffer bfr;          /* Buffers for gslib */
};

#if 0
  // 8. Setup the Frontier array.
  for (uint e = 0; e < fe; e++) {
    if (wids[e] == nw) {
      for (unsigned v = 0; v < nv; v++)
        frontier[e * nv + v] = 1;
    } else {
      for (unsigned v = 0; v < nv; v++)
        frontier[e * nv + v] = 0;
    }
  }

  // 9. Make sure frontier values are consistent.
  struct comm lc;
  comm_split(&c, c.id, c.id, &lc);
  struct gs_data *gsh = gs_setup(vids, fe * nv, &lc, 0, gs_pairwise, 0);
  gs(frontier, gs_int, gs_min, 0, gsh, &bfr);
  gs_free(gsh), comm_free(&lc);
#endif

static inline void allocate_work_arrays(struct box *box) {
  uint wsize = box->sn;
  box->sx = malloc(sizeof(double) * 2 * wsize);
  box->srhs = (void *)((double *)box->sx + wsize);
  buffer_init(&box->bfr, 1024);
}

static inline void setup_comms(struct box *box, const struct comm *c) {
  // Setup global comm.
  comm_dup(&box->global, c);
}

static void asm1_setup2(struct box *box) {
  const double tol = 1e-12;
  struct comm *gc = &box->global;
  buffer *bfr = &box->bfr;

  const uint nw = box->opts.nw;
  const uint nv = box->ncr;
  const uint nd = (nv == 8) ? 3 : 2;
  uint ne = box->un / nv;
  const uint max_ne = 5 * ne + 200;

  slong *vtx = tcalloc(slong, nv * max_ne);
  double *va = tcalloc(double, nv * nv * max_ne);
  // Copy data to vtx and va.
  sint *wids = tcalloc(sint, max_ne);
  fetch_nbrs_v3(&ne, vtx, va, wids, nv, nd, nw, max_ne, gc->c);

  box->sn = ne * nv;
  const uint nnz = box->sn * nv;
  uint *ia = tcalloc(uint, nnz);
  uint *ja = tcalloc(uint, nnz);
  for (uint e = 0; e < ne; e++) {
    for (uint j = 0; j < nv; j++) {
      for (uint i = 0; i < nv; i++) {
        ia[e * nv * nv + j * nv + i] = e * nv + i;
        ja[e * nv * nv + j * nv + i] = e * nv + j;
      }
    }
  }

  ulong *tmp_vtx = tcalloc(ulong, ne);
  uint null_space = 1;
  for (uint i = 0; i < box->sn; i++) {
    tmp_vtx[i] = vtx[i];
    // if (front[i] == 1)  null_space = 0, tmp_vtx[i] = 0;
  }
  assert(null_space == 0);

  // FIXME: init lc;
  struct comm *lc;
  // Setup ASM1 solver.
  box->asm1 =
      (void *)crs_xxt_setup(box->sn, tmp_vtx, nnz, ia, ja, va,
                            jl_dom_to_gs_dom(box->opts.dom), null_space, lc);

  // Setup the crs_dsavg which basically average the solution of original
  // domains.
  slong *gs_vtx = tcalloc(slong, box->sn);
  for (uint i = 0; i < box->un; i++) gs_vtx[i] = tmp_vtx[i];
  for (uint i = box->un; i < box->sn; i++) gs_vtx[i] = -tmp_vtx[i];
  box->gsh = gs_setup((const slong *)gs_vtx, box->sn, gc, 0, gs_auto, 0);

  free(tmp_vtx), free(gs_vtx), free(ia), free(ja);
}

struct box *crs_box_setup2(uint n, const ulong *id, uint nnz, const uint *Ai,
                           const uint *Aj, const double *A, const jl_opts *opts,
                           const struct comm *c) {
  struct box *box = tcalloc(struct box, 1);
  box->un = n;
  box->ncr = nnz / n;
  box->opts = *opts;

  setup_comms(box, c);
  asm1_setup2(box);
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
  buffer_free(&box->bfr);
  comm_free(&box->global);
  free(box->sx), free(box->srhs);
}
