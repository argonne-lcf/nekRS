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

#include "nekrs_crs.hpp"
#include "crs_box_impl.hpp"


#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

/* Mesh (vertex) to box Interpolation */
occa::memory o_phi_e, o_iphi_e;
occa::memory o_ub, o_uc;
/* multiplicative RHS update */
occa::memory o_A;
occa::memory o_sx, o_srhs, o_invmul;
/* setup unassembled to assembled map*/
static int num_compressed;
static occa::memory o_u2c_off, o_u2c_idx, o_cr, o_cx;

static inline int get_num_crs_dofs_1d() {
  return *(nekData.schwz_ncr);
}

static inline int get_num_box_dofs() {
  return *(nekData.box_n);
}

static void setup_u2c_map_aux(unsigned un, const sint *u2c, buffer *bfr) {
  struct map_t {
    uint u, c;
  };

  struct array map;
  array_init(struct map_t, &map, un);

  struct map_t m;
  for (uint i = 0; i < un; i++) {
    if (u2c[i] < 0) continue;
    m.u = i, m.c = u2c[i];
    array_cat(struct map_t, &map, &m, 1);
  }
  sarray_sort_2(struct map_t, map.ptr, map.n, c, 0, u, 0, bfr);

  uint gs_n = 0;
  struct map_t *pm = (struct map_t *)map.ptr;
  if (map.n > 0) {
    gs_n = 1;
    uint c = pm[0].c;
    for (uint i = 1; i < map.n; i++) {
      if (pm[i].c != c) {
        gs_n++;
        c = pm[i].c;
      }
    }
  }

  unsigned *gs_off = tcalloc(unsigned, gs_n + 1);
  unsigned *gs_idx = tcalloc(unsigned, map.n);
  uint gs_n2 = 0;
  if (map.n > 0) {
    gs_idx[0] = pm[0].u;
    for (uint i = 1; i < map.n; i++) {
      if (pm[i].c != pm[i - 1].c) {
        gs_n2++;
        gs_off[gs_n2] = i;
      }
      gs_idx[i] = pm[i].u;
    }
    gs_n2++;
  }
  assert(gs_n == gs_n2);
  num_compressed = gs_n;

  o_u2c_off = platform->device.malloc<unsigned>(gs_n + 1);
  o_u2c_off.copyFrom(gs_off);
  o_u2c_idx = platform->device.malloc<unsigned>(map.n);
  o_u2c_idx.copyFrom(gs_idx);
  o_cr = platform->device.malloc<float>(gs_n);
  o_cx = platform->device.malloc<float>(gs_n);
  free(gs_off), free(gs_idx), array_free(&map);
}

static void setup_u2c_map(struct box *box, const ulong *const vtx) {
  struct vid_t {
    ulong id;
    uint idx;
    sint perm;
  };

  struct array vids;
  uint n = box->sn;
  array_init(struct vid_t, &vids, n);

  struct vid_t vid;
  for (uint i = 0; i < n; i++) {
    vid.id = vtx[i], vid.idx = i;
    array_cat(struct vid_t, &vids, &vid, 1);
  }

  buffer *bfr = &(box->bfr);
  sarray_sort(struct vid_t, vids.ptr, vids.n, id, 1, bfr);

  struct vid_t *pv = (struct vid_t *)vids.ptr;
  ulong lid = 0;
  sint cn = 0;
  for (uint i = 0; i < vids.n; i++) {
    if (pv[i].id != lid)
      lid = pv[i].id, cn++;
    pv[i].perm = cn - 1;
  }
  box->cn = cn;
  sarray_sort(struct vid_t, vids.ptr, vids.n, idx, 0, bfr);

  pv = (struct vid_t *)vids.ptr;
  box->u2c = tcalloc(sint, n);
  for (uint i = 0; i < n; i++)
    box->u2c[i] = pv[i].perm;
  array_free(&vids);

  setup_u2c_map_aux(box->sn, box->u2c, bfr);
}

static void setup_ij(uint **ia, uint **ja, uint n, uint nnz) {
  uint ncr = nnz / n;
  uint ne = n / ncr;

  *ia = tcalloc(uint, nnz);
  *ja = tcalloc(uint, nnz);
  for (uint e = 0; e < ne; e++) {
    for (uint j = 0; j < ncr; j++) {
      for (uint i = 0; i < ncr; i++) {
        (*ia)[e * ncr * ncr + j * ncr + i] = e * ncr + i;
        (*ja)[e * ncr * ncr + j * ncr + i] = e * ncr + j;
      }
    }
  }
}

static void asm2_setup(struct box *box) {
  nek::box_crs_setup();

  uint n = *(nekData.box_n);
  uint nnz = *(nekData.box_nnz);
  const long long *gcrs = (const long long *)nekData.box_gcrs;
  const double *va = (const double *)nekData.box_a;
  uint null_space = *(nekData.box_null_space);

  uint *ia, *ja;
  setup_ij(&ia, &ja, n, nnz);
  ulong *gcrs_ul = tcalloc(ulong, n);
  for (uint i = 0; i < n; i++)
    gcrs_ul[i] = gcrs[i];
  box->asm2 = (void *)crs_xxt_setup(n, gcrs_ul, nnz, ia, ja, va, gs_float, null_space, &(box->global));

  free(ia), free(ja), free(gcrs_ul);
}

static void asm2_solve(const struct box *box) {
  crs_xxt_solve(box->sx, (struct xxt *)box->asm2, box->srhs);
}

static void asm1_setup(struct box *box, const jl_opts *opts, double tol, const struct comm *comm) {
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

  // Setup unassembled to assembled map.
  setup_u2c_map(box, tmp_vtx);

  // Create the assembled matrix in CSR format.
  struct csr *A = csr_setup(nnz, ia, ja, va, box->u2c, tol, &(box->bfr));

  // Setup ASM1 solver.
  box->ss = NULL;
  if (box->algo == BOX_XXT)
    box->ss = (void *)crs_xxt_setup(box->sn, tmp_vtx, nnz, ia, ja, va, opts->dom, opts->null_space, &(box->local));
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
}

static void setup_gpu_implementation(const struct box *box) {
  // Copy the phi_e, and iphi_e to C.
  nek::box_copy_phi_e();

  const int nelv = nekData.nelv;
  const int num_crs_dofs_1d = get_num_crs_dofs_1d();
  assert(box->un == (num_crs_dofs_1d * nelv));
  assert(box->sn >= box->un);

  // Setup device memory for vtx-to-box and box-to-vtx interpolation.
  const int m_size = num_crs_dofs_1d * box->un;
  o_phi_e = platform->device.malloc<double>(m_size);
  o_phi_e.copyFrom(nekData.box_phi_e);

  o_uc = platform->device.malloc<float>(box->un);
  o_ub = platform->device.malloc<float>(get_num_box_dofs());

  o_iphi_e = platform->device.malloc<int>(nelv);
  int *wrki = (int *)calloc(nelv, sizeof(int));
  for (int i = 0; i < nelv; i++)
    wrki[i] = nekData.box_iphi_e[i * num_crs_dofs_1d];
  o_iphi_e.copyFrom(wrki);
  free(wrki);

  /* multiplicative RHS update */
  o_sx = platform->device.malloc<float>(box->sn);
  o_srhs = platform->device.malloc<float>(box->sn);

  o_A = platform->device.malloc<float>(m_size);
  float *wrkf = (float *)calloc(m_size, sizeof(float));
  for (uint i = 0; i < m_size; i++)
    wrkf[i] = nekData.schwz_amat[i];
  o_A.copyFrom(wrkf);
  free(wrkf);

  o_invmul = platform->device.malloc<double>(box->un);
  o_invmul.copyFrom(box->inv_mul);
}

struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
                          const double *A, const jl_opts *opts, const struct comm *comm) {
  struct box *box = tcalloc(struct box, 1);
  box->un = n;
  box->ncr = (n != 0) ? (nnz / n) : 0;
  box->dom = opts->dom;
  box->mult = opts->mult;
  box->algo = opts->asm1;
  box->aggressive = opts->aggressive;

  // Allocate workspace.
  buffer_init(&(box->bfr), 1024);

  // Copy the global communicator.
  comm_dup(&(box->global), comm);

  // Copy the local communicator.
  MPI_Comm local;
  MPI_Comm_split(comm->c, comm->id, 1, &local);
  comm_init(&(box->local), local);
  MPI_Comm_free(&local);

  // ASM2 setup on C side.
  asm2_setup(box);

  // ASM1 setup on C side.
  box->sn = *(nekData.schwz_ne) * box->ncr;
  asm1_setup(box, opts, 1e-12, comm);

  // Allocate work arrays.
  uint work_array_size = MAX(box->sn, *(nekData.box_n));
#define allocate_work_arrays(T)                                                \
  {                                                                            \
    box->sx = malloc(sizeof(T) * 2 * work_array_size);                         \
    box->srhs = (void *)((T *)box->sx + work_array_size);                      \
  }
  BOX_DOMAIN_SWITCH(box->dom, allocate_work_arrays);
#undef allocate_work_arrays

  // Setup interpolation between ASM1 and ASM2 on C.
  setup_gpu_implementation(box);

  // Print some info.
  if (box->global.id == 0) {
    printf("%s: mult = %u, algo = %u, ne = %u, nw = %u\n", __func__,
           box->mult, box->algo, *(nekData.schwz_ne), *(nekData.schwz_nw));
    fflush(stdout);
  }

  return box;
}

void crs_box_solve(void *x, struct box *box, const void *rhs) {
  struct comm *c = &(box->global);

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
  gs(box->srhs, box->dom, gs_add, 0, box->gsh, &(box->bfr));
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
    crs_xxt_solve(box->sx, (struct xxt *)box->ss, box->srhs);
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
  timer_tic(c);
  if (box->mult) {
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
  }
  timer_toc(MULT_RHS_UPDATE);

  // Copy to nek5000 to do the global solve.
  // FIXME: This doesn't work anymore as we don't use box_e and box_r anymore.
#if 0
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
#endif

  // Solve on nek5000.
  // FIXME: broken due to box_map_vtx_to_box and box_map_box_to_vtx
  // not being available anymore.
#if 0
  timer_tic(c);
  nek::box_map_vtx_to_box();
  timer_toc(MAP_VTX_TO_BOX);

  timer_tic(c);
  nek::box_crs_solve();
  timer_toc(ASM2);

  timer_tic(c);
  nek::box_map_box_to_vtx();
  timer_toc(MAP_BOX_TO_VTX);
#endif

  // Copy from nek5000.
  // FIXME: This doesn't work anymore as we don't use box_e and box_r anymore.
#if 0
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
#endif

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
}

void crs_box_solve2(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  struct comm *c = &(box->global);

  if ((box->dom != gs_float) || (box->algo != BOX_GPU)) {
    if (c->id == 0)
      fprintf(stderr, "Wrong domain and/or wrong solver!\n");
    fflush(stderr);
    MPI_Abort(c->c, EXIT_FAILURE);
  }

  // Multiply by inverse multiplicity
  timer_tic(c);
  platform->boxInvMulKernel(box->un, o_rhs, o_invmul, o_srhs);
  timer_toc(INV_MUL);

  // Copy RHS to CPU.
  timer_tic(c);
  o_srhs.copyTo(box->srhs, box->un);
  timer_toc(COPY_RHS);

  // crs_dsavg1.
  timer_tic(c);
  gs(box->srhs, box->dom, gs_add, 0, box->gsh, &(box->bfr));
  timer_toc(CRS_DSAVG1);

  // Copy RHS back to GPU.
  timer_tic(c);
  o_srhs.copyFrom(box->srhs, box->sn);
  timer_toc(COPY_RHS);

  timer_tic(c);
  platform->boxUtoCKernel(num_compressed, o_u2c_off, o_u2c_idx, o_srhs, o_cr);
  timer_toc(U2C);

  // ASM1.
  timer_tic(c);
  asm1_gpu_solve(o_cx, box, o_cr);
  timer_toc(ASM1);

  timer_tic(c);
  platform->boxZeroKernel(box->sn, o_sx);
  timer_toc(ZERO);

  timer_tic(c);
  platform->boxCtoUKernel(num_compressed, o_u2c_off, o_u2c_idx, o_cx, o_sx);
  timer_toc(C2U);

  timer_tic(c);
  platform->boxInvMulKernel(box->un, o_sx, o_invmul, o_sx);
  timer_toc(INV_MUL);

  if (!(box->aggressive)) {
    timer_tic(c);
    o_sx.copyTo(box->sx, box->un);
    timer_toc(COPY_SOLUTION);

    // crs_dsavg2.
    timer_tic(c);
    gs(box->sx, box->dom, gs_add, 0, box->gsh, &(box->bfr));
    timer_toc(CRS_DSAVG1);

    timer_tic(c);
    o_sx.copyFrom(box->sx, box->un);
    timer_toc(COPY_SOLUTION);
  }

  // mult_rhs_update:  rhs = rhs - A*sx.
  if (box->mult) {
    timer_tic(c);
    int num_crs_dofs_1d = get_num_crs_dofs_1d();
    platform->boxMultRHSKernel(box->un / num_crs_dofs_1d, num_crs_dofs_1d, o_A, o_sx, o_srhs);
    timer_toc(MULT_RHS_UPDATE);
  }

  platform->boxCopyKernel(box->un, o_srhs, o_uc);

  timer_tic(c);
  platform->boxZeroKernel(get_num_box_dofs(), o_ub);
  timer_toc(ZERO);

  timer_tic(c);
  platform->mapVtxToBoxKernel(nekData.nelv, o_iphi_e, get_num_crs_dofs_1d(), o_phi_e, o_uc, o_ub);
  timer_toc(MAP_VTX_TO_BOX);

  timer_tic(c);
  o_ub.copyTo(box->srhs);
  timer_toc(COPY_RHS);

  timer_tic(c);
  asm2_solve(box);
  timer_toc(ASM2);

  timer_tic(c);
  o_ub.copyFrom(box->sx);
  timer_toc(COPY_SOLUTION);

  timer_tic(c);
  platform->mapBoxToVtxKernel(nekData.nelv, o_iphi_e, get_num_crs_dofs_1d(), o_phi_e, o_ub, o_uc);
  timer_toc(MAP_BOX_TO_VTX);

  platform->boxUpdateSolutionKernel(box->un, o_uc, o_sx);

  timer_tic(c);
  platform->boxInvMulKernel(box->un, o_sx, o_invmul, o_sx);
  timer_toc(INV_MUL);

  timer_tic(c);
  o_sx.copyTo(box->sx);
  timer_toc(COPY_SOLUTION);

  timer_tic(c);
  gs(box->sx, box->dom, gs_add, 0, box->gsh, &(box->bfr));
  timer_toc(CRS_DSAVG1);

  timer_tic(c);
  o_x.copyFrom(box->sx, box->un, 0);
  timer_toc(COPY_SOLUTION);
}

void crs_box_free(struct box *box) {
  if (!box) return;

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

  crs_xxt_free((struct xxt *)box->asm2);
  gs_free(box->gsh);
  buffer_free(&box->bfr);
  comm_free(&box->local);
  comm_free(&box->global);
  free(box->u2c);
  free(box->inv_mul);
  free(box->sx);
}

#undef MAX
