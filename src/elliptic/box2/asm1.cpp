#include "box.hpp"
#include "xxt.hpp"

void crs_asm1_setup(slong *const id, const sint *const frontier,
                    const double *const A, const struct comm *const c,
                    struct box *const box) {
  const uint ns = box->ns, nv = box->nv;
  const uint ne = ns / nv;
  const uint nnz = ns * nv;
  uint *Ai = tcalloc(uint, nnz);
  uint *Aj = tcalloc(uint, nnz);
  for (uint e = 0; e < ne; e++) {
    for (uint j = 0; j < nv; j++) {
      for (uint i = 0; i < nv; i++) {
        Ai[e * nv * nv + j * nv + i] = e * nv + i;
        Aj[e * nv * nv + j * nv + i] = e * nv + j;
      }
    }
  }

  // Setup ASM1. First, setup an isolated comm for each MPI process.
  struct comm lc;
  MPI_Comm local;
  MPI_Comm_split(c->c, c->id, c->id, &local);
  comm_init(&lc, local);
  MPI_Comm_free(&local);

  ulong *masked_ids = tcalloc(ulong, ns);
  for (uint i = 0; i < ns; i++) masked_ids[i] = id[i] * !frontier[i];
  box->asm1 = crs_xxt_setup(ns, masked_ids, nnz, Ai, Aj, A, gs_real,
                            0 /* null space */, &lc);
  free(masked_ids), free(Ai), free(Aj);
  comm_free(&lc);

  // Setup inverse multiplicity.
  const uint nu = box->nu;
  struct gs_data *gsh = gs_setup((const slong *)id, nu, c, 0, gs_auto, 0);
  for (uint i = 0; i < nu; i++) box->sim[i] = 1.0;
  gs(box->sim, gs_real, gs_add, 0, gsh, &box->bfr);
  for (uint i = 0; i < nu; i++) box->sim[i] = 1.0 / box->sim[i];
  gs_free(gsh);

  // Setup RAS.
  for (uint i = nu; i < ns; i++) id[i] = -id[i];
  box->ras = gs_setup((const slong *)id, ns, c, 0, gs_auto, 0);
}

void crs_asm1_free(struct box *const box) {
  gs_free(box->ras);
  crs_xxt_free(box->asm1);
}
