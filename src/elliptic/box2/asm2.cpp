#include "box.hpp"
#include "xxt.hpp"

#include <cfloat>

struct asm2 {
  /* User input */
  uint nd, nv;
  uint nbx, nby, nbz;
  double tol;
  /* Structured grid */
  double mnmx[3][2];
  double dx[3];
  double *xyz;
};

#define IDX3(E, V, D) (nv * nd * E + nd * V + D)

/*----------------------------------------------------------------------
 *  domain_size
 *
 *  Global extremes of the domain.
 *--------------------------------------------------------------------*/
void domain_size(double *min, double *max, const uint ne, const uint nd,
                 const uint nv, const double *const xyz,
                 const struct comm *const c) {
  for (uint d = 0; d < nd; d++) min[d] = DBL_MAX, max[d] = DBL_MIN;

  for (uint e = 0; e < ne; e++) {
    for (uint v = 0; v < nv; v++) {
      for (uint d = 0; d < nd; d++) {
        if (xyz[IDX3(e, v, d)] < min[d]) min[d] = xyz[IDX3(e, v, d)];
        if (xyz[IDX3(e, v, d)] > max[d]) max[d] = xyz[IDX3(e, v, d)];
      }
    }
  }

  double wrk[3];
  assert(nd <= 3);
  comm_allreduce(c, gs_double, gs_min, min, 3, &wrk);
  comm_allreduce(c, gs_double, gs_max, max, 3, &wrk);
}

/*----------------------------------------------------------------------
 *  set_crs_box_dims
 *
 *  Size the structured box grid from the global domain extent and fill
 *  the box vertex coordinate table; zero the per-box matrices.
 *--------------------------------------------------------------------*/
static void set_crs_box_dims(struct asm2 *data, const double *const min,
                             const double *const max) {
  assert(data->nd == 3 && "Only 3D meshes are supported");

  const double tol = data->tol;
  double xmax = max[0], xmin = min[0];
  double ymax = max[1], ymin = min[1];
  double zmax = max[2], zmin = min[2];

  double dx = xmax - xmin;
  double x0 = xmin - tol * dx;
  double x1 = xmax + tol * dx;

  double dy = ymax - ymin;
  double y0 = ymin - tol * dy;
  double y1 = ymax + tol * dy;

  double dz = zmax - zmin;
  double z0 = zmin - tol * dz;
  double z1 = zmax + tol * dz;

  data->mnmx[0][0] = x0;
  data->mnmx[0][1] = x1;
  data->mnmx[1][0] = y0;
  data->mnmx[1][1] = y1;
  data->mnmx[2][0] = z0;
  data->mnmx[2][1] = z1;

  dx = data->dx[0] = (x1 - x0) / data->nbx;
  dy = data->dx[1] = (y1 - y0) / data->nby;
  dz = data->dx[2] = (z1 - z0) / data->nbz;

  const uint nbox = data->nbx * data->nby * data->nbz;
  const uint nd = data->nd;
  const uint nv = data->nv;
  data->xyz = tcalloc(double, nbox * nv * nd);
  uint i, j, k, l = 0, ii, kk, jj, ll;
  for (k = 0; k < data->nbz; k++) {
    for (j = 0; j < data->nby; j++) {
      for (i = 0; i < data->nbx; i++) {
        ll = 0;
        for (kk = 0; kk <= 1; kk++) {
          for (jj = 0; jj <= 1; jj++) {
            for (ii = 0; ii <= 1; ii++) {
              data->xyz[IDX3(l, ll, 0)] = x0 + dx * (i + ii);
              data->xyz[IDX3(l, ll, 1)] = y0 + dy * (j + jj);
              data->xyz[IDX3(l, ll, 2)] = z0 + dz * (k + kk);
              ll = ll + 1;
            }
          }
        }
        l = l + 1;
      }
    }
  }
}

struct xxt *crs_asm2_setup(const uint ne, const uint nd, const uint nv,
                           const double *const xyz,
                           const double *const centroid, const uint nbx,
                           const uint nby, const uint nbz,
                           const struct comm *const c) {
  struct asm2 data;
  data.nd = nd, data.nv = nv, data.nbx = nbx, data.nby = nby, data.nbz = nbz;
  data.tol = 1e-2;

  double min[3], max[3];
  domain_size(min, max, ne, nd, nv, xyz, c);
  set_crs_box_dims(&data, min, max);

  if (c->id == 0) {
    printf("%g %g box domain x %d\n", min[0], max[0], data.nbx);
    printf("%g %g box domain y %d\n", min[1], max[1], data.nby);
    printf("%g %g box domain z %d\n", min[2], max[2], data.nbz);
    printf("%g %g %g box dxdydz\n", data.dx[0], data.dx[1], data.dx[2]);
  }
}

void crs_asm2_solve(occa::memory &o_x, struct xxt *xxt, occa::memory &o_rhs) {}

void crs_asm2_free(struct xxt *xxt) {}

#undef IDX3
