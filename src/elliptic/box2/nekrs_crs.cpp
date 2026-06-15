#include "crs_box.hpp"
#include "crs_xxt.hpp"

struct crs {
  uint un, algo;
  struct comm c;
  gs_dom dom;
  void *x, *rhs;
  void *solver;
};

static struct crs *crs = NULL;

static void allocate_work_arrays(struct crs *crs) {
  size_t usize;
  switch (crs->dom) {
  case gs_double: usize = sizeof(double); break;
  case gs_float: usize = sizeof(float); break;
  default:
    fprintf(stderr, "%s: unknown gs_dom = %d.\n", __func__, crs->dom);
    MPI_Abort(crs->c.c, EXIT_FAILURE);
    break;
  }

  crs->x = (void *)calloc(crs->un, usize);
  crs->rhs = (void *)calloc(crs->un, usize);
}

void jl_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
              const double *A, const jl_opts *opts, MPI_Comm comm) {
  if (opts->dom != gs_float) {
    fprintf(stderr, "%s: Only gs_float is allowed!\n", __func__);
    fflush(stderr);
    MPI_Abort(comm, EXIT_FAILURE);
  }

  crs = (struct crs *)calloc(1, sizeof(struct crs));
  crs->un = n;
  crs->dom = opts->dom;
  crs->algo = opts->algo;

  struct comm *c = &crs->c;
  comm_init(c, comm);

  allocate_work_arrays(crs);

  switch (crs->algo) {
  case XXT:
    crs->solver = (void *)crs_xxt_setup(n, id, nnz, Ai, Aj, A, crs->dom,
                                        opts->null_space, c);
    break;
  case BOX:
    crs->solver = (void *)crs_box_setup2(n, id, nnz, Ai, Aj, A, opts, c);
    break;
  default: break;
  }
}

#define DOMAIN_SWITCH(dom, macro)                                              \
  {                                                                            \
    switch (dom) {                                                             \
    case gs_double: macro(double); break;                                      \
    case gs_float: macro(float); break;                                        \
    }                                                                          \
  }

static void _crs_xxt_solve(occa::memory &o_x, occa::memory &o_rhs) {
  o_rhs.copyTo(crs->rhs, crs->un);
  crs_xxt_solve(crs->x, (struct xxt *)crs->solver, crs->rhs);
  o_x.copyFrom(crs->x, crs->un);
}

void jl_solve2(occa::memory &o_x, occa::memory &o_rhs) {
  switch (crs->algo) {
  case XXT: _crs_xxt_solve(o_x, o_rhs); break;
  case BOX: crs_box_solve2(o_x, (struct box *)crs->solver, o_rhs); break;
  default: break;
  }
}

void jl_free() {
  if (crs == NULL) return;

  switch (crs->algo) {
  case XXT: crs_xxt_free((struct xxt *)crs->solver); break;
  case BOX: crs_box_free2((struct box *)crs->solver); break;
  default: break;
  }

  comm_free(&(crs->c));
  free(crs->x), free(crs->rhs);
  free(crs), crs = NULL;
}

#undef DOMAIN_SWITCH
