#include "crs_box.hpp"

static int check_alloc_(void *ptr, const char *file, unsigned line) {
  if (ptr == NULL) {
    printf("check_alloc failure: %s:%d\n", file, line);
    return 1;
  }
  return 0;
}
#define check_alloc(ptr) check_alloc_(ptr, __FILE__, __LINE__)

static int gen_crs_basis(dfloat *b, int j_, dfloat *z, int Nq, int Np) {
  dfloat *zr = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *zs = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *zt = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *z0 = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *z1 = (dfloat *)calloc(Nq, sizeof(dfloat));
  if (zr == NULL || zs == NULL || zt == NULL || z0 == NULL || z1 == NULL)
    return 1;

  for (int i = 0; i < Nq; i++) {
    z0[i] = 0.5 * (1 - z[i]);
    z1[i] = 0.5 * (1 + z[i]);
  }

  memcpy(zr, z0, Nq * sizeof(dfloat));
  memcpy(zs, z0, Nq * sizeof(dfloat));
  memcpy(zt, z0, Nq * sizeof(dfloat));

  int jj = j_ + 1;
  if (jj % 2 == 0)
    memcpy(zr, z1, Nq * sizeof(dfloat));
  if (jj == 3 || jj == 4 || jj == 7 || jj == 8)
    memcpy(zs, z1, Nq * sizeof(dfloat));
  if (jj > 4)
    memcpy(zt, z1, Nq * sizeof(dfloat));

  for (int k = 0; k < Nq; k++) {
    for (int j = 0; j < Nq; j++) {
      for (int i = 0; i < Nq; i++) {
        int n = i + Nq * j + Nq * Nq * k + j_ * Np;
        b[n] = zr[i] * zs[j] * zt[k];
      }
    }
  }

  free(zr), free(zs), free(zt), free(z0), free(z1);

  return 0;
}

static int get_local_crs_galerkin(double *a, int nc, mesh_t *mf,
                                  elliptic_t *ef) {
  size_t nelt = mf->Nelements, Np = mf->Np;
  size_t size = nelt * Np;

  dfloat *b = tcalloc(dfloat, nc * Np);
  check_alloc(b);
  for (int j = 0; j < nc; j++)
    gen_crs_basis(b, j, mf->gllz, mf->Nq, mf->Np);

  dfloat *u = tcalloc(dfloat, size), *w = tcalloc(dfloat, size);
  check_alloc(u), check_alloc(w);

  occa::memory o_u = platform->device.malloc<dfloat>(size, u);
  occa::memory o_w = platform->device.malloc<dfloat>(size, w);
  occa::memory o_upf = platform->device.malloc<pfloat>(size);
  occa::memory o_wpf = platform->device.malloc<pfloat>(size);

  int i, j, k, e;
  for (j = 0; j < nc; j++) {
    for (e = 0; e < nelt; e++)
      memcpy(&u[e * Np], &b[j * Np], Np * sizeof(dfloat));

    o_u.copyFrom(u);
    platform->copyDfloatToPfloatKernel(mf->Nlocal, o_u, o_upf);
    ellipticAx(ef, mf->Nelements, mf->o_elementList, o_upf, o_wpf, pfloatString);
    platform->copyPfloatToDfloatKernel(mf->Nlocal, o_wpf, o_w);
    o_w.copyTo(w);

    for (e = 0; e < nelt; e++)
      for (i = 0; i < nc; i++) {
        a[i + j * nc + e * nc * nc] = 0.0;
        for (k = 0; k < Np; k++)
          a[i + j * nc + e * nc * nc] += b[k + i * Np] * w[k + e * Np];
      }
  }

  free(b), free(w), free(u);
  o_u.free(), o_w.free(), o_upf.free(), o_wpf.free();

  return 0;
}

static void set_mat_ij(uint *ia, uint *ja, int nc, int nelt) {
  uint i, j, e;
  for (e = 0; e < nelt; e++) {
    for (j = 0; j < nc; j++) {
      for (i = 0; i < nc; i++) {
        ia[i + j * nc + nc * nc * e] = e * nc + i;
        ja[i + j * nc + nc * nc * e] = e * nc + j;
      }
    }
  }
}

/*
 * nekRS interface to coarse solvers
 */
void jl_setup_aux(uint *ntot_, ulong **gids_, uint *nnz_, uint **ia_,
                  uint **ja_, double **a_, elliptic_t *elliptic,
                  elliptic_t *ellipticf) {
  mesh_t *mesh = elliptic->mesh, *meshf = ellipticf->mesh;
  assert(mesh->Nelements == meshf->Nelements);
  uint nelt = meshf->Nelements, nc = mesh->Np;

  // Set global ids: copy and apply the mask
  uint ntot = *ntot_ = nelt * nc;
  ulong *gids = *gids_ = tcalloc(ulong, ntot);
  check_alloc(gids);

  for (int j = 0; j < nelt * nc; j++)
    gids[j] = mesh->globalIds[j];

  if (elliptic->Nmasked) {
    dlong *mask_ids = (dlong *)calloc(elliptic->Nmasked, sizeof(dlong));
    elliptic->o_maskIds.copyTo(mask_ids, elliptic->Nmasked);
    for (int n = 0; n < elliptic->Nmasked; n++)
      gids[mask_ids[n]] = 0;
    free(mask_ids);
  }

  // Set coarse matrix
  uint nnz = *nnz_ = nc * nc * nelt;
  double *a = *a_ = tcalloc(double, nnz);
  check_alloc(a);

  get_local_crs_galerkin(a, nc, meshf, ellipticf);

  uint *ia = *ia_ = tcalloc(uint, nnz), *ja = *ja_ = tcalloc(uint, nnz);
  check_alloc(ia), check_alloc(ja);
  set_mat_ij(ia, ja, nc, nelt);
}
#undef check_alloc

struct crs {
  uint un, type;
  struct comm c;
  gs_dom dom;
  float *wrk;
  void *x, *rhs;
  void *solver;
};

static struct crs *crs = NULL;

void allocate_work_arrays(struct crs *crs) {
  size_t usize;
  switch (crs->dom) {
  case gs_double:
    usize = sizeof(double);
    break;
  case gs_float:
    usize = sizeof(float);
    break;
  default:
    fprintf(stderr, "%s: unknown gs_dom = %d.\n", __func__, crs->dom);
    MPI_Abort(crs->c.c, EXIT_FAILURE);
    break;
  }

  crs->x = calloc(usize, crs->un);
  crs->rhs = calloc(usize, crs->un);
  crs->wrk = tcalloc(float, crs->un);
}

void jl_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
              const double *A, const jl_opts *opts, MPI_Comm comm) {
  if (crs != NULL) {
    if (crs->c.id == 0)
      fprintf(stderr, "%s: coarse solver is already initialized.\n", __func__);
    fflush(stderr);
    return;
  }

  crs = tcalloc(struct crs, 1);
  crs->un = n;
  crs->dom = opts->dom;
  crs->type = opts->algo;
  allocate_work_arrays(crs);

  comm_init(&(crs->c), comm);
  struct comm *c = &(crs->c);
  switch (crs->type) {
  case XXT:
    crs->solver = (void *)crs_xxt_setup(n, id, nnz, Ai, Aj, A, opts, c);
    break;
  case BOX:
    crs->solver = (void *)crs_box_setup(n, id, nnz, Ai, Aj, A, opts, c);
    break;
  default:
    break;
  }

  if (crs->c.id == 0) {
    printf("%s: n = %u, nnz = %u, null = %u, dom = %s\n", __func__, crs->un, nnz,
        opts->null_space, (crs->dom == gs_double) ? "double" : "float");
    fflush(stdout);
  }
}

void jl_timer_init() {
  timer_init();
}

#define DOMAIN_SWITCH(dom, macro)                                              \
  {                                                                            \
    switch (dom) {                                                             \
    case gs_double:                                                            \
      macro(double);                                                           \
      break;                                                                   \
    case gs_float:                                                             \
      macro(float);                                                            \
      break;                                                                   \
    }                                                                          \
  }

void jl_solve(occa::memory &o_x, occa::memory &o_rhs) {
  timer_tic(&(crs->c));
  o_rhs.copyTo(crs->wrk, crs->un, 0);
#define copy_from_buf(T)                                                       \
  {                                                                            \
    T *rhs = (T *)crs->rhs;                                                    \
    for (uint i = 0; i < crs->un; i++)                                         \
      rhs[i] = crs->wrk[i];                                                    \
  }
  DOMAIN_SWITCH(crs->dom, copy_from_buf);
#undef copy_from_buf
  timer_toc(COPY_RHS);

  switch (crs->type) {
  case XXT:
    crs_xxt_solve(crs->x, (struct xxt *)crs->solver, crs->rhs);
    break;
  case BOX:
    crs_box_solve(crs->x, (struct box *)crs->solver, crs->rhs);
    break;
  default:
    break;
  }

  timer_tic(&(crs->c));
#define copy_to_buf(T)                                                         \
  {                                                                            \
    T *x = (T *)crs->x;                                                        \
    for (uint i = 0; i < crs->un; i++)                                         \
      crs->wrk[i] = x[i];                                                      \
  }
  DOMAIN_SWITCH(crs->dom, copy_to_buf);
#undef copy_to_buf
  o_x.copyFrom(crs->wrk, crs->un, 0);
  timer_toc(COPY_SOLUTION);
}

void jl_solve2(occa::memory &o_x, occa::memory &o_rhs) {
  crs_box_solve2(o_x, (struct box *)crs->solver, o_rhs);
}

#undef DOMAIN_SWITCH

void jl_free() {
  if (crs == NULL) return;

  switch (crs->type) {
  case XXT:
    crs_xxt_free((struct xxt *)crs->solver);
    break;
  case BOX:
    crs_box_free((struct box *)crs->solver);
    break;
  default:
    break;
  }

  comm_free(&(crs->c));
  free(crs->x), free(crs->rhs), free(crs->wrk), free(crs);
  crs = NULL;
}

void jl_timer_print(MPI_Comm comm) {
  timer_print(comm);
}
