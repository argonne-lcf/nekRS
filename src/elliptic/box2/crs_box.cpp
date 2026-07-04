#include "crs_box.hpp"

struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai,
                          const uint *Aj, const double *A, const double *xyz,
                          const jl_opts *opts, const struct comm *c) {
  // TODO: nbx, nby and nbz should be user input.
  uint nbx, nby, nbz;

  // Initiailize options.
  struct box *box = tcalloc(struct box, 1);
  box->opts = *opts;

  if (opts->algo == ASM1 || opts->algo == BOX)
    box->asm1 = crs_asm1_setup(n, id, nnz, Ai, Aj, A, xyz, opts->nw, c);

  return box;
}

void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  crs_asm1_solve(o_x, (void *)box->asm1, o_rhs);
}

void crs_box_free(struct box *box) { crs_asm1_free((void *)box->asm1); }
