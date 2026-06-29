#include "crs_box.hpp"
#include "crs_xxt.hpp"

struct crs {
  uint algo;
  void *solver;
};

struct xxt_data {
  uint un;
  void *x, *rhs;
};

static struct crs *crs = NULL;
static struct xxt_data *xxt_data = NULL;

void jl_setup(uint n, const ulong *id, uint nnz, const uint *Ai, const uint *Aj,
              const double *A, const double *xyz, const jl_opts *opts,
              MPI_Comm comm) {
  if (opts->dom != gs_real) {
    fprintf(stderr, "%s: Invalid precision!\n", __func__);
    fflush(stderr);
    MPI_Abort(comm, EXIT_FAILURE);
  }

  struct comm c;
  comm_init(&c, comm);

  crs = (struct crs *)calloc(1, sizeof(struct crs));
  crs->algo = opts->algo;

  switch (crs->algo) {
  case XXT:
    crs->solver = (void *)crs_xxt_setup(n, id, nnz, Ai, Aj, A, opts->dom,
                                        opts->null_space, &c);
    xxt_data = (struct xxt_data *)calloc(1, sizeof(struct xxt_data));
    xxt_data->un = n;
    xxt_data->x = (void *)calloc(xxt_data->un, sizeof(real));
    xxt_data->rhs = (void *)calloc(xxt_data->un, sizeof(real));
    break;
  case ASM1:
  case ASM2:
  case BOX:
    crs->solver = (void *)crs_box_setup(n, id, nnz, Ai, Aj, A, xyz, opts, &c);
    break;
  default: break;
  }

  comm_free(&c);
}

void jl_solve(occa::memory &o_x, occa::memory &o_rhs) {
  switch (crs->algo) {
  case XXT:
    o_rhs.copyTo(xxt_data->rhs, xxt_data->un);
    crs_xxt_solve(xxt_data->x, (struct xxt *)crs->solver, xxt_data->rhs);
    o_x.copyFrom(xxt_data->x, xxt_data->un);
    break;
  case ASM1:
  case ASM2:
  case BOX: crs_box_solve(o_x, (struct box *)crs->solver, o_rhs); break;
  default: break;
  }
}

void jl_free() {
  if (crs == NULL) return;

  switch (crs->algo) {
  case XXT:
    crs_xxt_free((struct xxt *)crs->solver);
    free(xxt_data->rhs), free(xxt_data->x), free(xxt_data), xxt_data = NULL;
    break;
  case ASM1:
  case ASM2:
  case BOX: crs_box_free((struct box *)crs->solver); break;
  default: break;
  }

  free(crs), crs = NULL;
}
