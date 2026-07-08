#include "box.hpp"

struct box *crs_box_setup(const uint n, const ulong *const id, const uint nnz,
                          const uint *const Ai, const uint *const Aj,
                          const double *const A, const double *const xyz,
                          const double *const centroid,
                          const jl_opts *const opts,
                          const struct comm *const c) {
  struct box *box = tcalloc(struct box, 1);

  if (opts->algo == ASM1 || opts->algo == BOX)
    box->asm1 = crs_asm1_setup(n, id, nnz, Ai, Aj, A, xyz, opts->nw, c);

  uint nv = nnz / n;
  uint ne = n / nv;
  assert(nv == 8);
  uint nd = 3;
  if (opts->algo == ASM2 || opts->algo == BOX) {
    box->asm2 = crs_asm2_setup(ne, nd, nv, xyz, centroid, A, opts->nbx,
                               opts->nby, opts->nbz, c);
  }

  return box;
}

void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  if (box->asm1) crs_asm1_solve(o_x, (void *)box->asm1, o_rhs);
}

void crs_box_free(struct box *box) {
  if (!box) return;
  if (box->asm1) crs_asm1_free((void *)box->asm1);
  free(box);
}
