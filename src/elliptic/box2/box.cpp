#include "box.hpp"
#include "xxt.hpp"

static inline struct box *box_init(const uint ne, const uint nd, const uint nv,
                                   const uint nu, const uint ns) {
  struct box *b = tcalloc(struct box, 1);
  b->ne = ne, b->nd = nd, b->nv = nv, b->nu = nu, b->ns = ns;
  b->sx = tcalloc(real, b->ns);
  b->srhs = tcalloc(real, b->ns);
  b->sim = tcalloc(real, b->ns);
  buffer_init(&b->bfr, 1024);
  return b;
}

struct box *crs_box_setup(const uint n, const ulong *const id, const uint nnz,
                          const uint *const Ai, const uint *const Aj,
                          const double *const A, const double *const coord,
                          const double *const centroid,
                          const jl_opts *const opts,
                          const struct comm *const c) {
  (void)Ai;
  (void)Aj;

  const uint nv = nnz / n;
  const uint nd = (nv == 8) ? 3 : 2;
  const uint ne = n / nv;
  const uint nu = ne * nv;
  const uint mne = 5 * ne + 200;

  slong *id_ = tcalloc(slong, nv * mne);
  for (uint i = 0; i < nu; i++) id_[i] = id[i];

  double *A_ = tcalloc(double, nv * nv * mne);
  memcpy(A_, A, sizeof(double) * nv * nv * ne);

  double *coord_ = tcalloc(double, nd * nv * mne);
  memcpy(coord_, coord, sizeof(double) * nd * nv * ne);

  // Call fetch neighbors.
  sint *frontier = tcalloc(sint, nv * mne);
  sint *wids = tcalloc(sint, mne);
  uint se = ne;
  crs_overlap(&se, nd, nv, id_, coord_, A_, frontier, opts->nw, wids, c->c, mne,
              0);
  const uint ns = se * nv;

  struct box *box = box_init(ne, nd, nv, nu, ns);
  if (opts->algo == ASM1 || opts->algo == BOX)
    crs_asm1_setup(id_, frontier, A_, c, box);

#if 0
  if (opts->algo == ASM2 || opts->algo == BOX)
    crs_asm2_setup(coord, centroid, A, opts->nbx, opts->nby, opts->nbz, c, box);
#endif

  free(id_), free(A_), free(coord_);
  free(frontier), free(wids);
  return box;
}

void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  o_rhs.copyTo(box->srhs, box->nu);

  gs(box->srhs, gs_real, gs_add, 0, box->ras, &box->bfr);
  for (uint i = 0; i < box->nu; i++) box->srhs[i] = box->sim[i] * box->srhs[i];

  crs_xxt_solve(box->sx, (struct xxt *)box->asm1, box->srhs);

  gs(box->sx, gs_real, gs_add, 0, box->ras, &box->bfr);
  for (uint i = 0; i < box->nu; i++) box->sx[i] = box->sim[i] * box->sx[i];

  o_x.copyFrom(box->sx, box->nu);
}

void crs_box_free(struct box *box) {
  if (!box) return;
  crs_asm1_free(box);
#if 0
  crs_asm2_free(box);
#endif
  free(box->sx), free(box->srhs), free(box->sim);
  buffer_free(&box->bfr);
  free(box);
}
