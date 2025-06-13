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

#include <nekInterfaceAdapter.hpp>
#include <nekrs_crs.hpp>

#include "crs_box_impl.hpp"

static void crs_box_dump(uint n, const ulong *id, uint nnz, const uint *Ai,
                         const uint *Aj, const double *A, uint null_space,
                         const struct comm *comm) {
  char file_name[BUFSIZ];
  snprintf(file_name, BUFSIZ, "crs_box_dump_%02d.txt", comm->id);

  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open file %s for writing.\n", file_name);
    fflush(stderr);
    MPI_Abort(comm->c, EXIT_FAILURE);
  }

  fprintf(fp, "%u %u %u\n", n, nnz, null_space);
  for (uint i = 0; i < n; i++)
    fprintf(fp, "%llu\n", id[i]);
  for (uint i = 0; i < nnz; i++)
    fprintf(fp, "%u %u %e\n", Ai[i], Aj[i], A[i]);

  fclose(fp);
}

static void dump_asm1(const char *name, const uint n, const ulong *const dofs,
                      const uint nnz, const uint *const Ai,
                      const uint *const Aj, const double *const Av,
                      const struct comm *const comm) {
  char file_name[BUFSIZ];
  snprintf(file_name, BUFSIZ, "%s_%02d.txt", name, comm->id);

  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open file %s for writing.\n", file_name);
    fflush(stderr);
    MPI_Abort(comm->c, EXIT_FAILURE);
  }

  fprintf(fp, "%u %u\n", n, nnz);
  for (uint i = 0; i < nnz; i++)
    fprintf(fp, "%llu %llu %e\n", dofs[Ai[i]], dofs[Aj[i]], Av[i]);

  fclose(fp);
}

static void dump_asm1_solution(const char *name, const uint n,
                               const double *const x, const uint ndim,
                               const double *const xyz,
                               const struct comm *const comm) {
  char file_name[BUFSIZ];
  snprintf(file_name, BUFSIZ, "%s_%02d.txt", name, comm->id);

  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open file %s for writing.\n", file_name);
    fflush(stderr);
    MPI_Abort(comm->c, EXIT_FAILURE);
  }

  const uint nv = (ndim == 3) ? 8 : 4;
  assert(n % nv == 0);

  for (uint e = 0; e < n / nv; e++) {
    const double *xyz_e = &xyz[e * nv * ndim];
    const double *x_e = &x[e * nv];
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[0], xyz_e[0], xyz_e[1], xyz_e[2]);
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[1], xyz_e[3], xyz_e[4], xyz_e[5]);
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[2], xyz_e[9], xyz_e[10], xyz_e[11]);
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[3], xyz_e[6], xyz_e[7], xyz_e[8]);

    if (nv == 4)
      continue;

    xyz_e = &xyz_e[3 * 4];
    x_e = &x_e[4];
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[0], xyz_e[0], xyz_e[1], xyz_e[2]);
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[1], xyz_e[3], xyz_e[4], xyz_e[5]);
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[2], xyz_e[9], xyz_e[10], xyz_e[11]);
    fprintf(fp, "%d %e %e %e %e\n", e, x_e[3], xyz_e[6], xyz_e[7], xyz_e[8]);
  }

  fclose(fp);
}

void box_debug(const int verbose, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (verbose > 0)
    vprintf(fmt, args);
  fflush(stdout);
  va_end(args);
}

static const sint *get_u2c(unsigned *cni, const unsigned n,
                           const ulong *const vtx, buffer *bfr) {
  struct vid_t {
    ulong id;
    uint idx;
    sint perm;
  };

  struct array vids;
  array_init(struct vid_t, &vids, n);

  struct vid_t vid;
  for (uint i = 0; i < n; i++) {
    vid.id = vtx[i], vid.idx = i;
    array_cat(struct vid_t, &vids, &vid, 1);
  }
  sarray_sort(struct vid_t, vids.ptr, vids.n, id, 1, bfr);

  struct vid_t *pv = (struct vid_t *)vids.ptr;
  ulong lid = 0;
  sint cn = 0;
  for (uint i = 0; i < vids.n; i++) {
    if (pv[i].id != lid)
      lid = pv[i].id, cn++;
    pv[i].perm = cn - 1;
  }
  *cni = cn;
  sarray_sort(struct vid_t, vids.ptr, vids.n, idx, 0, bfr);

  // Setup u2c -- user vector to compress vector mapping.
  pv = (struct vid_t *)vids.ptr;
  sint *const u2c = tcalloc(sint, n);
  for (uint i = 0; i < n; i++)
    u2c[i] = pv[i].perm;

  array_free(&vids);

  return u2c;
}

static void setup_asm1(struct box *box, double tol, const struct comm *comm) {
  uint ne = *(nekData.schwz_ne);
  uint nw = *(nekData.schwz_nw);
  const long long *vtx = (const long long *)nekData.schwz_vtx;
  const double *mask = (const double *)nekData.schwz_mask;
  const int *frontier = (const int *)nekData.schwz_frontier;
  const double *const xyz = nekData.schwz_xyz;
  const double *va = (const double *)nekData.schwz_amat;

  const uint ncr = box->ncr, sn = box->sn;
  const uint nnz = sn * ncr;
  uint *ia = tcalloc(uint, nnz);
  uint *ja = tcalloc(uint, nnz);
  for (uint e = 0; e < ne; e++) {
    for (uint j = 0; j < ncr; j++) {
      for (uint i = 0; i < ncr; i++) {
        ia[e * ncr * ncr + j * ncr + i] = e * ncr + i;
        ja[e * ncr * ncr + j * ncr + i] = e * ncr + j;
      }
    }
  }

  ulong *tmp_vtx = tcalloc(ulong, box->sn);
  double *const tmp_mask = tcalloc(double, box->sn);
  double mask_min = DBL_MAX;
  for (unsigned i = 0; i < box->sn; i++) {
    tmp_vtx[i] = vtx[i];
    tmp_mask[i] = mask[i];
    if (frontier[i] == 1)
      tmp_mask[i] = 0;

    if (tmp_mask[i] < 0.1)
      tmp_vtx[i] = 0;
    if (tmp_mask[i] < mask_min)
      mask_min = tmp_mask[i];
  }
  free(tmp_mask);

  uint null_space = 1;
  if (mask_min < 1e-10)
    null_space = 0;
  assert(null_space == 0);

  box->cn = 0;
  box->u2c = (int *)get_u2c(&box->cn, box->sn, tmp_vtx, &box->bfr);
  box->ss = NULL;

  struct csr *A = csr_setup(nnz, ia, ja, va, box->u2c, tol, &box->bfr);

  if (box->algo == BOX_XXT)
    box->ss = (void *)crs_xxt_setup(box->sn, tmp_vtx, nnz, ia, ja, va, null_space, &(box->local), box->dom);
  if (box->algo == BOX_CHOLMOD)
    asm1_cholmod_setup(A, null_space, box);
  if (box->algo == BOX_GPU)
    asm1_gpu_setup(A, null_space, box);

  csr_free(A), free(ia), free(ja);

  // Setup the crs_dsavg which basically average the solution of original
  // parRSB domains.
  slong *gs_vtx = tcalloc(slong, box->sn);
  for (uint i = 0; i < box->un; i++)
    gs_vtx[i] = tmp_vtx[i];
  for (uint i = box->un; i < box->sn; i++)
    gs_vtx[i] = -tmp_vtx[i];
  box->gsh = gs_setup((const slong *)gs_vtx, box->sn, comm, 0, gs_auto, 0);
  free(gs_vtx), free(tmp_vtx);

  box->inv_mul = tcalloc(double, box->sn);
  for (uint i = 0; i < box->un; i++)
    box->inv_mul[i] = 1.0;
  gs(box->inv_mul, gs_double, gs_add, 0, box->gsh, &box->bfr);
  for (uint i = 0; i < box->sn; i++)
    box->inv_mul[i] = 1.0 / box->inv_mul[i];

    // Allocate work arrays.
#define allocate_work_arrays(T)                                                \
  {                                                                            \
    box->sx = malloc(sizeof(T) * 2 * box->sn);                                 \
    box->srhs = (void *)((T *)box->sx + box->sn);                              \
  }
  BOX_DOMAIN_SWITCH(box->dom, allocate_work_arrays);
#undef allocate_work_arrays
}

struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
                          const double *A, const jl_opts *opts, const struct comm *comm) {
  struct box *box = tcalloc(struct box, 1);
  box->un = n;
  box->ncr = nnz / n;
  box->dom = opts->dom;
  box->mult = opts->mult;
  box->algo = opts->asm1;
  buffer_init(&(box->bfr), 1024);

  // Copy the global communicator.
  comm_dup(&(box->global), comm);

  // Copy the local communicator.
  MPI_Comm local;
  MPI_Comm_split(comm->c, comm->id, 1, &local);
  comm_init(&(box->local), local);
  MPI_Comm_free(&local);

  if (opts->timer)
    timer_init();

  // ASM2 setup using Fortran. We should port this to C.
  nek::box_crs_setup();

  // ASM1 setup on C side.
  box->sn = *(nekData.schwz_ne) * box->ncr;
  setup_asm1(box, 1e-12, comm);

  // Print some info.
  if (box->global.id == 0) {
    printf("%s: mult = %u, algo = %u, ne = %u, nw = %u\n", __func__,
           box->mult, box->algo, *(nekData.schwz_ne), *(nekData.schwz_nw));
    fflush(stdout);
  }

  return box;
}

void crs_box_solve(void *x, struct box *box, const void *rhs) {
  struct comm *c = &box->global;

  // Copy RHS.
  timer_tic(c);
#define copy_rhs(T)                                                            \
  {                                                                            \
    const T *rhsi = (const T *)rhs;                                            \
    T *srhs = (T *)box->srhs;                                                  \
    for (uint i = 0; i < box->un; i++)                                         \
      srhs[i] = rhsi[i];                                                       \
  }
  BOX_DOMAIN_SWITCH(box->dom, copy_rhs);
#undef copy_rhs
  timer_toc(COPY_RHS);

  // crs_dsavg1.
  timer_tic(c);
  gs(box->srhs, box->dom, gs_add, 0, box->gsh, &box->bfr);
#define avg(T)                                                                 \
  {                                                                            \
    T *srhs = (T *)box->srhs;                                                  \
    for (uint i = 0; i < box->sn; i++)                                         \
      srhs[i] = box->inv_mul[i] * srhs[i];                                     \
  }
  BOX_DOMAIN_SWITCH(box->dom, avg);
#undef avg
  timer_toc(CRS_DSAVG1);

  // ASM1.
  timer_tic(c);
  switch (box->algo) {
  case BOX_XXT:
    crs_xxt_solve((double *)box->sx, (struct xxt *)box->ss, (const double *)box->srhs);
    break;
  case BOX_CHOLMOD:
    asm1_cholmod_solve(box->sx, box, box->srhs);
    break;
  case BOX_GPU:
    asm1_gpu_solve(box->sx, box, box->srhs);
    break;
  default:
    break;
  }
  timer_toc(ASM1);

  // crs_dsavg2.
  timer_tic(c);
  gs(box->sx, box->dom, gs_add, 0, box->gsh, &box->bfr);
#define avg(T)                                                                 \
  {                                                                            \
    T *sx = (T *)box->sx;                                                      \
    for (uint i = 0; i < box->sn; i++)                                         \
      sx[i] = box->inv_mul[i] * sx[i];                                         \
  }
  BOX_DOMAIN_SWITCH(box->dom, avg);
#undef avg
  timer_toc(CRS_DSAVG1);

  // mult_rhs_update.
  if (box->mult) {
    timer_tic(c);
    // rhs = rhs - A*sx.
#define update_rhs(T)                                                          \
  {                                                                            \
    const double *A = (const double *)nekData.schwz_amat;                      \
    const T *sx = (T *)box->sx;                                                \
    T *srhs = (T *)box->srhs;                                                  \
    uint ncr = box->ncr, ue = box->un / ncr;                                   \
    for (uint e = 0; e < ue; e++) {                                            \
      for (uint c = 0; c < ncr; c++) {                                         \
        for (uint k = 0; k < ncr; k++) {                                       \
          srhs[k + ncr * e] -=                                                 \
              sx[c + ncr * e] * A[k + c * ncr + ncr * ncr * e];                \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
    BOX_DOMAIN_SWITCH(box->dom, update_rhs);
#undef update_rhs
    timer_toc(MULT_RHS_UPDATE);
  }

  // Copy to nek5000 to do the global solve.
  timer_tic(c);
#define copy_to_nek5000(T)                                                     \
  {                                                                            \
    const T *srhs = (T *)box->srhs;                                            \
    for (uint i = 0; i < box->un; i++)                                         \
      nekData.box_r[i] = srhs[i];                                              \
  }
  BOX_DOMAIN_SWITCH(box->dom, copy_to_nek5000);
#undef copy_to_nek5000
  timer_toc(COPY_TO_NEK5000);

  // Solve on nek5000.
  timer_tic(c);
  nek::box_map_vtx_to_box();
  timer_toc(MAP_VTX_TO_BOX);

  timer_tic(c);
  nek::box_crs_solve();
  timer_toc(ASM2);

  timer_tic(c);
  nek::box_map_box_to_vtx();
  timer_toc(MAP_BOX_TO_VTX);

  // Copy from nek5000.
  timer_tic(c);
#define copy_from_nek5000(T)                                                   \
  {                                                                            \
    T *sx = (T *)box->sx;                                                      \
    for (uint i = 0; i < box->un; i++)                                         \
      sx[i] += nekData.box_e[i];                                               \
  }
  BOX_DOMAIN_SWITCH(box->dom, copy_from_nek5000);
#undef copy_from_nek5000
  timer_toc(COPY_FROM_NEK5000);

  // crs_dsavg3.
  timer_tic(c);
  gs(box->sx, box->dom, gs_add, 0, box->gsh, &box->bfr);
#define avg(T)                                                                 \
  {                                                                            \
    T *sx = (T *)box->sx;                                                      \
    for (uint i = 0; i < box->un; i++)                                         \
      sx[i] = box->inv_mul[i] * sx[i];                                         \
  }
  BOX_DOMAIN_SWITCH(box->dom, avg);
#undef avg
  timer_toc(CRS_DSAVG1);

  // Copy solution.
  timer_tic(c);
#define copy_to_x(T)                                                           \
  {                                                                            \
    T *sx = (T *)box->sx, *xi = (T *)x;                                        \
    for (uint i = 0; i < box->un; i++)                                         \
      xi[i] = sx[i];                                                           \
  }
  BOX_DOMAIN_SWITCH(box->dom, copy_to_x);
#undef copy_to_x
  timer_toc(COPY_SOLUTION);

  timer_print(&box->global, 1000);
}

void crs_box_solve2(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  struct comm *c = &(box->global);

  if ((box->dom != gs_float) || (box->algo != BOX_GPU)) {
    if (c->id == 0)
      fprintf(stderr, "Wrong domain or wrong solver!\n");
    fflush(stderr);
    MPI_Abort(c->c, EXIT_FAILURE);
  }

  timer_tic(c);
  // Can move the first inv_mul.* to here.
  o_rhs.copyTo(box->srhs, box->un, 0);
  timer_toc(COPY_RHS);

  // crs_dsavg1.
  timer_tic(c);
  gs(box->srhs, box->dom, gs_add, 0, box->gsh, &box->bfr);
#define avg(T)                                                                 \
  {                                                                            \
    T *srhs = (T *)box->srhs;                                                  \
    for (uint i = 0; i < box->sn; i++)                                         \
      srhs[i] = box->inv_mul[i] * srhs[i];                                     \
  }
  BOX_DOMAIN_SWITCH(box->dom, avg);
#undef avg
  timer_toc(CRS_DSAVG1);

  // ASM1.
  timer_tic(c);
  asm1_gpu_solve(box->sx, box, box->srhs);
  timer_toc(ASM1);

  // crs_dsavg2.
  timer_tic(c);
  gs(box->sx, box->dom, gs_add, 0, box->gsh, &box->bfr);
#define avg(T)                                                                 \
  {                                                                            \
    T *sx = (T *)box->sx;                                                      \
    for (uint i = 0; i < box->sn; i++)                                         \
      sx[i] = box->inv_mul[i] * sx[i];                                         \
  }
  BOX_DOMAIN_SWITCH(box->dom, avg);
#undef avg
  timer_toc(CRS_DSAVG1);

  // mult_rhs_update:  rhs = rhs - A*sx.
  timer_tic(c);
  if (box->mult) {
#define update_rhs(T)                                                          \
  {                                                                            \
    const double *A = (const double *)nekData.schwz_amat;                      \
    const T *sx = (T *)box->sx;                                                \
    T *srhs = (T *)box->srhs;                                                  \
    uint ncr = box->ncr, ue = box->un / ncr;                                   \
    for (uint e = 0; e < ue; e++) {                                            \
      for (uint c = 0; c < ncr; c++) {                                         \
        for (uint k = 0; k < ncr; k++) {                                       \
          srhs[k + ncr * e] -=                                                 \
              sx[c + ncr * e] * A[k + c * ncr + ncr * ncr * e];                \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
    BOX_DOMAIN_SWITCH(box->dom, update_rhs);
#undef update_rhs
  }
  timer_toc(MULT_RHS_UPDATE);

  // Copy to nek5000 to do the global solve.
  timer_tic(c);
#define copy_to_nek5000(T)                                                     \
  {                                                                            \
    const T *srhs = (T *)box->srhs;                                            \
    for (uint i = 0; i < box->un; i++)                                         \
      nekData.box_r[i] = srhs[i];                                              \
  }
  BOX_DOMAIN_SWITCH(box->dom, copy_to_nek5000);
#undef copy_to_nek5000
  timer_toc(COPY_TO_NEK5000);

  // Solve on nek5000.
  timer_tic(c);
  nek::box_map_vtx_to_box();
  timer_toc(MAP_VTX_TO_BOX);

  timer_tic(c);
  nek::box_crs_solve();
  timer_toc(ASM2);

  timer_tic(c);
  nek::box_map_box_to_vtx();
  timer_toc(MAP_BOX_TO_VTX);

  // Copy from nek5000.
  timer_tic(c);
#define copy_from_nek5000(T)                                                   \
  {                                                                            \
    T *sx = (T *)box->sx;                                                      \
    for (uint i = 0; i < box->un; i++)                                         \
      sx[i] += nekData.box_e[i];                                               \
  }
  BOX_DOMAIN_SWITCH(box->dom, copy_from_nek5000);
#undef copy_from_nek5000
  timer_toc(COPY_FROM_NEK5000);

  // crs_dsavg3.
  timer_tic(c);
  gs(box->sx, box->dom, gs_add, 0, box->gsh, &box->bfr);
#define avg(T)                                                                 \
  {                                                                            \
    T *sx = (T *)box->sx;                                                      \
    for (uint i = 0; i < box->un; i++)                                         \
      sx[i] = box->inv_mul[i] * sx[i];                                         \
  }
  BOX_DOMAIN_SWITCH(box->dom, avg);
#undef avg
  timer_toc(CRS_DSAVG1);

  // Copy solution.
  timer_tic(c);
  o_x.copyFrom(box->sx, box->un, 0);
  timer_toc(COPY_SOLUTION);

  timer_print(&box->global, 1000);
}

void crs_box_free(struct box *box) {
  if (!box)
    return;

  switch (box->algo) {
  case BOX_XXT:
    crs_xxt_free((struct xxt *)box->ss);
    break;
  case BOX_CHOLMOD:
    asm1_cholmod_free(box);
    break;
  case BOX_GPU:
    asm1_gpu_free(box);
    break;
  default:
    break;
  }

  gs_free(box->gsh);
  buffer_free(&box->bfr);
  comm_free(&box->local);
  comm_free(&box->global);
  free(box->u2c);
  free(box->inv_mul);
  free(box->sx);
}
