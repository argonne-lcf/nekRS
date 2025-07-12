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

#include "nekInterfaceAdapter.hpp"
#include "nekrs_crs.hpp"
#include "crs_box_impl.hpp"

static inline int get_num_crs_dofs_1d() {
  return *(nekData.schwz_ncr);
}

static inline int get_num_box_dofs() {
  return *(nekData.box_n);
}

static inline unsigned get_wrk_size(const struct box *box) {
#define MAX(a, b) ((a) > (b) ? (a) : (b))
  return MAX(box->sn, *(nekData.box_n));
#undef MAX
}

/* setup unassembled to assembled map*/
static occa::memory o_u2c_off, o_u2c_idx, o_cr, o_cx;

template <typename T>
static void u2c_setup_aux(unsigned un, const sint *u2c, buffer *bfr) {
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

  uint gs_n = 0;
  if (map.n > 0) {
    gs_n = 1;
    sarray_sort_2(struct map_t, map.ptr, map.n, c, 0, u, 0, bfr);
    struct map_t *pm = (struct map_t *)map.ptr;
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
    struct map_t *pm = (struct map_t *)map.ptr;
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

  o_u2c_off = platform->device.malloc<unsigned>(gs_n + 1);
  o_u2c_off.copyFrom(gs_off);
  o_u2c_idx = platform->device.malloc<unsigned>(map.n);
  o_u2c_idx.copyFrom(gs_idx);
  o_cr = platform->device.malloc<T>(gs_n);
  o_cx = platform->device.malloc<T>(gs_n);
  free(gs_off), free(gs_idx), array_free(&map);
}

template <typename T>
static sint *u2c_setup(struct box *box, const ulong *const vtx) {
  const uint n = box->sn;
  buffer *bfr = &(box->bfr);

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

  ulong lid = 0;
  sint cn = 0;
  sarray_sort(struct vid_t, vids.ptr, vids.n, id, 1, bfr);
  struct vid_t *pv = (struct vid_t *)vids.ptr;
  for (uint i = 0; i < vids.n; i++) {
    if (pv[i].id != lid)
      lid = pv[i].id, cn++;
    pv[i].perm = cn - 1;
  }
  box->cn = cn;

  sint *u2c = tcalloc(sint, n);
  sarray_sort(struct vid_t, vids.ptr, vids.n, idx, 0, bfr);
  pv = (struct vid_t *)vids.ptr;
  for (uint i = 0; i < n; i++)
    u2c[i] = pv[i].perm;
  array_free(&vids);

  u2c_setup_aux<T>(box->sn, u2c, bfr);
  return u2c;
}

static void asm2_setup(struct box *box) {
  const uint n = *(nekData.box_n);
  const uint nnz = *(nekData.box_nnz);
  const long long *gcrs = (const long long *)nekData.box_gcrs;
  const double *va = (const double *)nekData.box_a;
  const uint null_space = *(nekData.box_null_space);

  const uint ncr = box->ncr;
  const uint ne = n / ncr;

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

  ulong *gcrs_ul = tcalloc(ulong, n);
  for (uint i = 0; i < n; i++)
    gcrs_ul[i] = gcrs[i];

  box->asm2 = (void *)crs_xxt_setup(n, gcrs_ul, nnz, ia, ja, va, box->opts.dom, null_space, &(box->global));

  free(ia), free(ja), free(gcrs_ul);
}

static void asm2_solve(const struct box *box) {
  crs_xxt_solve(box->sx, (struct xxt *)box->asm2, box->srhs);
}

template <typename T>
static void asm1_setup(struct box *box) {
  const double tol = 1e-12;
  struct comm *local = &(box->local);
  struct comm *global = &(box->global);
  buffer *bfr = &(box->bfr);

  const uint ncr = box->ncr;
  const uint ne = *(nekData.schwz_ne);
  const uint nw = *(nekData.schwz_nw);
  const long long *vtx = (const long long *)nekData.schwz_vtx;
  const double *mask = (const double *)nekData.schwz_mask;
  const int *frontier = (const int *)nekData.schwz_frontier;
  const double *const xyz = nekData.schwz_xyz;
  const double *va = (const double *)nekData.schwz_amat;

  box->sn = ne * ncr;

  const uint nnz = box->sn * ncr;
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
  double *tmp_mask = tcalloc(double, box->sn);
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

  uint null_space = !(mask_min < 1e-10);
  assert(null_space == 0);

  // Setup unassembled to assembled map.
  sint *u2c = u2c_setup<T>(box, tmp_vtx);
  // Create the assembled matrix in CSR format.
  struct csr *A = csr_setup(nnz, ia, ja, va, u2c, tol, bfr);
  free(u2c);

  // Setup ASM1 solver.
  box->asm1 = NULL;
  switch(box->opts.asm1) {
  case BOX_XXT:
    box->asm1 = (void *)crs_xxt_setup(box->sn, tmp_vtx, nnz, ia, ja, va,
        box->opts.dom, box->opts.null_space, local);
    break;
  case BOX_CHOLMOD:
    asm1_cholmod_setup(A, null_space, box);
    break;
  case BOX_GPU:
    asm1_gpu_setup<T>(A, null_space, box);
    break;
  }

  // Setup the crs_dsavg which basically average the solution of original
  // parRSB domains.
  slong *gs_vtx = tcalloc(slong, box->sn);
  for (uint i = 0; i < box->un; i++)
    gs_vtx[i] = tmp_vtx[i];
  for (uint i = box->un; i < box->sn; i++)
    gs_vtx[i] = -tmp_vtx[i];
  box->gsh = gs_setup((const slong *)gs_vtx, box->sn, global, 0, gs_auto, 0);

  free(tmp_mask), free(tmp_vtx), free(gs_vtx);
  csr_free(A), free(ia), free(ja);
}

template <typename T>
static void cpu_setup(struct box *box) {
  unsigned wrk_size = get_wrk_size(box);
  box->sx = malloc(sizeof(T) * 2 * wrk_size);
  box->srhs = (void *)((T *)box->sx + wrk_size);
}

/* Mesh (vertex) to box Interpolation */
occa::memory o_phi_e, o_iphi_e;
occa::memory o_ub, o_uc;
/* multiplicative RHS update */
occa::memory o_A;
occa::memory o_sx, o_srhs, o_invmul;

template <typename T>
static void gpu_setup(struct box *box) {
  const uint nelv = nekData.nelv;
  const uint num_crs_dofs_1d = get_num_crs_dofs_1d();
  assert(box->un == (num_crs_dofs_1d * nelv));
  assert(box->sn >= box->un);

  /* Copy the phi_e, and iphi_e to C */
  nek::box_copy_phi_e();

  /* Setup device memory for vtx-to-box and box-to-vtx interpolation */
  const int m_size = num_crs_dofs_1d * box->un;
  o_phi_e = platform->device.malloc<T>(m_size);
  T * box_phi_e = tcalloc(T, m_size);
  for (uint i = 0; i < m_size; i++)
    box_phi_e[i] = nekData.box_phi_e[i];
  o_phi_e.copyFrom(box_phi_e);
  free(box_phi_e);

  o_iphi_e = platform->device.malloc<int>(nelv);
  int *box_iphi_e = tcalloc(int, nelv);
  for (uint i = 0; i < nelv; i++)
    box_iphi_e[i] = nekData.box_iphi_e[i * num_crs_dofs_1d];
  o_iphi_e.copyFrom(box_iphi_e);
  free(box_iphi_e);

  o_uc = platform->device.malloc<T>(box->un);
  o_ub = platform->device.malloc<T>(get_num_box_dofs());

  /* multiplicative RHS update */
  uint wrk_size = get_wrk_size(box);
  o_sx = platform->device.malloc<T>(wrk_size);
  o_srhs = platform->device.malloc<T>(wrk_size);

  o_A = platform->device.malloc<T>(m_size);
  T *amat = tcalloc(T, m_size);
  for (uint i = 0; i < m_size; i++)
    amat[i] = nekData.schwz_amat[i];
  o_A.copyFrom(amat);
  free(amat);

  o_invmul = platform->device.malloc<T>(box->un);
  T *inv_mul = tcalloc(T, box->sn);
  for (uint i = 0; i < box->un; i++)
    inv_mul[i] = 1.0;
  gs(inv_mul, box->opts.dom, gs_add, 0, box->gsh, &(box->bfr));
  for (uint i = 0; i < box->sn; i++)
    inv_mul[i] = 1.0 / inv_mul[i];
  o_invmul.copyFrom(inv_mul);
  free(inv_mul);
}

struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
                          const double *A, const jl_opts *opts, const struct comm *comm) {
  struct box *box = tcalloc(struct box, 1);
  box->un = n;
  box->ncr = nnz / n;
  box->opts = *opts;

  /* Allocate workspace */
  buffer_init(&(box->bfr), 1024);

  /* Copy the global communicator */
  comm_dup(&(box->global), comm);

  /* Copy the local communicator */
  MPI_Comm local;
  MPI_Comm_split(comm->c, comm->id, 1, &local);
  comm_init(&(box->local), local);
  MPI_Comm_free(&local);

  /* Setup box solver in Nek5000 */
  nek::box_crs_setup();

  /* ASM2 setup on C side */
  asm2_setup(box);

  /* ASM1, CPU and GPU setup on C side */
  switch(opts->dom) {
    case gs_double:
      asm1_setup<double>(box);
      cpu_setup<double>(box);
      gpu_setup<double>(box);
      break;
    case gs_float:
      asm1_setup<float>(box);
      cpu_setup<float>(box);
      gpu_setup<float>(box);
      break;
    default:
      break;
  }

  /* Print some info */
  if (box->global.id == 0) {
    printf("%s: mult = %u, algo = %u, ne = %u, nw = %u\n", __func__,
           opts->mult, opts->asm1, *(nekData.schwz_ne), *(nekData.schwz_nw));
    fflush(stdout);
  }

  return box;
}

void crs_box_solve(occa::memory &o_x, struct box *box, occa::memory &o_rhs) {
  struct comm *c = &(box->global);
  buffer *bfr = &(box->bfr);
  const gs_dom dom = box->opts.dom;
  const box_algo_t asm1 = box->opts.asm1;

  if (asm1 != BOX_GPU) MPI_Abort(c->c, EXIT_FAILURE);

  timer_tic(c);
  if (dom == gs_double) platform->boxCopyFloatToDoubleKernel(box->un, o_rhs, o_srhs);
  else platform->boxCopyKernel(box->un, o_rhs, o_srhs);
  timer_toc(COPY_D2D);

  timer_tic(c);
  platform->boxInvMul2Kernel(box->un, o_srhs, o_invmul);
  timer_toc(INV_MUL);

  timer_tic(c);
  o_srhs.copyTo(box->srhs, box->un);
  timer_toc(COPY_RHS);

  timer_tic(c);
  gs(box->srhs, dom, gs_add, 0, box->gsh, bfr);
  timer_toc(CRS_DSAVG1);

  timer_tic(c);
  o_srhs.copyFrom(box->srhs, box->sn);
  timer_toc(COPY_RHS);

  timer_tic(c);
  platform->boxUtoCKernel(box->cn, o_u2c_off, o_u2c_idx, o_srhs, o_cr);
  timer_toc(U2C);

  timer_tic(c);
  asm1_gpu_solve(o_cx, box, o_cr);
  timer_toc(ASM1);

  timer_tic(c);
  platform->boxZeroKernel(box->sn, o_sx);
  timer_toc(ZERO);

  timer_tic(c);
  platform->boxCtoUKernel(box->cn, o_u2c_off, o_u2c_idx, o_cx, o_sx);
  timer_toc(C2U);

  timer_tic(c);
  platform->boxInvMul2Kernel(box->un, o_sx, o_invmul);
  timer_toc(INV_MUL);

  if (!(box->opts.aggressive)) {
    timer_tic(c);
    o_sx.copyTo(box->sx, box->un);
    timer_toc(COPY_SOLUTION);

    timer_tic(c);
    gs(box->sx, dom, gs_add, 0, box->gsh, bfr);
    timer_toc(CRS_DSAVG1);

    timer_tic(c);
    o_sx.copyFrom(box->sx, box->un);
    timer_toc(COPY_SOLUTION);
  }

  timer_tic(c);
  if (box->opts.mult) {
    unsigned num_crs_dofs_1d = get_num_crs_dofs_1d();
    platform->boxMultRHSKernel(box->un / num_crs_dofs_1d, num_crs_dofs_1d, o_A, o_sx, o_srhs);
  }
  timer_toc(MULT_RHS_UPDATE);

  timer_tic(c);
  platform->boxCopyKernel(box->un, o_srhs, o_uc);
  timer_toc(COPY_D2D);

  timer_tic(c);
  platform->boxZeroKernel(get_num_box_dofs(), o_ub);
  timer_toc(ZERO);

  timer_tic(c);
  platform->boxMapVtxToBoxKernel(nekData.nelv, o_iphi_e, get_num_crs_dofs_1d(), o_phi_e, o_uc, o_ub);
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
  platform->boxMapBoxToVtxKernel(nekData.nelv, o_iphi_e, get_num_crs_dofs_1d(), o_phi_e, o_ub, o_uc);
  timer_toc(MAP_BOX_TO_VTX);

  platform->boxUpdateSolutionKernel(box->un, o_uc, o_sx);

  timer_tic(c);
  platform->boxInvMul2Kernel(box->un, o_sx, o_invmul);
  timer_toc(INV_MUL);

  timer_tic(c);
  o_sx.copyTo(box->sx, box->un);
  timer_toc(COPY_SOLUTION);

  timer_tic(c);
  gs(box->sx, dom, gs_add, 0, box->gsh, bfr);
  timer_toc(CRS_DSAVG1);

  timer_tic(c);
  o_sx.copyFrom(box->sx, box->un);
  timer_toc(COPY_SOLUTION);

  timer_tic(c);
  if (dom == gs_double) platform->boxCopyDoubleToFloatKernel(box->un, o_sx, o_x);
  else platform->boxCopyKernel(box->un, o_sx, o_x);
  timer_toc(COPY_D2D);
}

void crs_box_free(struct box *box) {
  if (!box) return;

  switch (box->opts.asm1) {
  case BOX_XXT:
    crs_xxt_free((struct xxt *)box->asm1);
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
  comm_free(&box->local), comm_free(&box->global);
  free(box->sx);
}
