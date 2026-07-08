#include "box.hpp"
#include "xxt.hpp"

struct asm1 {
  uint un, sn, cn;       /* User size, Schwarz size,compressed size */
  real *sx, *srhs, *sim; /* Schwarz/XXT work arrays on CPU */
  void *solver;          /* Pointer to the asm1 solver */
  struct gs_data *ras;   /* RAS */
  buffer bfr;            /* Buffers for gslib */
};

static inline void allocate_work_arrays(struct asm1 *asm1) {
  asm1->sx = tcalloc(real, asm1->sn);
  asm1->srhs = tcalloc(real, asm1->sn);
  asm1->sim = tcalloc(real, asm1->sn);
}

void *crs_asm1_setup(uint n, const ulong *const id, uint nnz,
                     const uint *const Ai, const uint *const Aj,
                     const double *const A, const double *const coord,
                     const uint nw, const struct comm *const c) {
  struct asm1 *asm1 = tcalloc(struct asm1, 1);
  buffer_init(&asm1->bfr, 1024);

  const uint un = asm1->un = n;
  const uint nv = nnz / n;
  const uint nd = (nv == 8) ? 3 : 2;

  uint ne = un / nv;
  const uint mne = 5 * ne + 200;

  // Setup eids array.
  slong *eids = tcalloc(slong, mne);
  slong out[2][1], in = ne, wrk[2][1];
  comm_scan(out, c, gs_long, gs_add, &in, 1, wrk);
  for (uint e = 0; e < ne; e++) eids[e] = out[0][0] + e;

  // Setup vtx array.
  slong *vtx = tcalloc(slong, nv * mne);
  for (uint i = 0; i < un; i++) vtx[i] = id[i];

  // Setup xyz array.
  double *xyz = tcalloc(double, nd * nv * mne);
  memcpy(xyz, coord, sizeof(double) * nd * nv * ne);

  // Setup va array.
  double *va = tcalloc(double, nv * nv * mne);
  memcpy(va, A, sizeof(double) * nv * nv * ne);

  // Setup frontier and wids arrays.
  sint *frontier = tcalloc(sint, nv * mne);
  sint *wids = tcalloc(sint, mne);

  // Call fetch neighbors.
  crs_overlap(&ne, eids, nv, vtx, xyz, va, frontier, nw, wids, c->c, mne);
  const uint sn = asm1->sn = ne * nv;

  // Allocate work arrays: needs sn to be set.
  allocate_work_arrays(asm1);

  const uint snnz = sn * nv;
  uint *ia = tcalloc(uint, snnz);
  uint *ja = tcalloc(uint, snnz);
  for (uint e = 0; e < ne; e++) {
    for (uint j = 0; j < nv; j++) {
      for (uint i = 0; i < nv; i++) {
        ia[e * nv * nv + j * nv + i] = e * nv + i;
        ja[e * nv * nv + j * nv + i] = e * nv + j;
      }
    }
  }

  // Setup an isolated comm for each MPI process.
  struct comm lc;
  MPI_Comm local;
  MPI_Comm_split(c->c, c->id, c->id, &local);
  comm_init(&lc, local);
  MPI_Comm_free(&local);

  // Setup the ASM1 solver.
  ulong *sid = tcalloc(ulong, sn);
  for (uint i = 0; i < sn; i++) sid[i] = vtx[i] * !(frontier[i]);
  asm1->solver = (void *)crs_xxt_setup(sn, sid, snnz, ia, ja, va, gs_real,
                                       0 /* null space */, &lc);
  free(sid);
  comm_free(&lc);

  // Setup inverse multiplicity.
  struct gs_data *gsh = gs_setup((const slong *)vtx, un, c, 0, gs_auto, 0);
  for (uint i = 0; i < un; i++) asm1->sim[i] = 1.0;
  gs(asm1->sim, gs_real, gs_add, 0, gsh, &asm1->bfr);
  for (uint i = 0; i < un; i++) asm1->sim[i] = 1.0 / asm1->sim[i];
  gs_free(gsh);

  // Setup the crs_dsavg which basically average the solution of original
  // domains.
  for (uint i = un; i < sn; i++) vtx[i] = -vtx[i];
  asm1->ras = gs_setup((const slong *)vtx, sn, c, 0, gs_auto, 0);

  free(eids), free(vtx), free(xyz);
  free(frontier), free(wids);
  free(ia), free(ja), free(va);

  return (void *)asm1;
}

void crs_asm1_solve(occa::memory &o_x, void *solver, occa::memory &o_rhs) {
  struct asm1 *asm1 = (struct asm1 *)solver;

  o_rhs.copyTo(asm1->srhs, asm1->un);

  gs(asm1->srhs, gs_real, gs_add, 0, asm1->ras, &asm1->bfr);
  for (uint i = 0; i < asm1->un; i++)
    asm1->srhs[i] = asm1->sim[i] * asm1->srhs[i];

  crs_xxt_solve(asm1->sx, (struct xxt *)asm1->solver, asm1->srhs);

  gs(asm1->sx, gs_real, gs_add, 0, asm1->ras, &asm1->bfr);
  for (uint i = 0; i < asm1->un; i++) asm1->sx[i] = asm1->sim[i] * asm1->sx[i];

  o_x.copyFrom(asm1->sx, asm1->un);
}

void crs_asm1_free(void *solver) {
  struct asm1 *asm1 = (struct asm1 *)solver;
  if (!asm1) return;

  free(asm1->sx), free(asm1->srhs), free(asm1->sim);
  gs_free(asm1->ras);
  crs_xxt_free((struct xxt *)asm1->solver);
  buffer_free(&asm1->bfr);
  free(asm1);
}
