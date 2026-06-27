#include "crs_box.hpp"

struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai,
                          const uint *Aj, const double *A, const double *xyz,
                          const jl_opts *opts, const struct comm *comm) {
  return crs_asm1_setup(n, id, nnz, Ai, Aj, A, xyz, opts, comm);
}
