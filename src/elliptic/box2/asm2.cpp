#include "box.hpp"
#include "xxt.hpp"

#include <cfloat>

struct asm2 {
  /* User input */
  uint ne, nd, nv;
  uint nbx, nby, nbz;
  double tol;
  /* Structured grid */
  double mnmx[3][2];
  double dx[3];
  ulong *lelem_to_gbox;
  uint *gbox_to_lbox;
  double *phi;
  /* Assembled coarse box operator */
  uint nbox;    /* Number of active (local) boxes on this rank */
  double *abox; /* nbox 8x8 matrices, one per active box */
};

#define IDX3(E, V, D) (nv * nd * E + nd * V + D)

/*----------------------------------------------------------------------
 *  domain_size
 *
 *  Global extremes of the domain.
 *--------------------------------------------------------------------*/
static void domain_size(struct asm2 *const data, const double *const xyz,
                        const struct comm *const c) {
  const uint ne = data->ne, nd = data->nd, nv = data->nv;
  assert(nd <= 3);

  double min[3], max[3], wrk[3];
  for (uint d = 0; d < nd; d++) min[d] = DBL_MAX, max[d] = -DBL_MAX;

  for (uint e = 0; e < ne; e++) {
    for (uint v = 0; v < nv; v++) {
      for (uint d = 0; d < nd; d++) {
        if (xyz[IDX3(e, v, d)] < min[d]) min[d] = xyz[IDX3(e, v, d)];
        if (xyz[IDX3(e, v, d)] > max[d]) max[d] = xyz[IDX3(e, v, d)];
      }
    }
  }

  comm_allreduce(c, gs_double, gs_min, min, 3, &wrk);
  comm_allreduce(c, gs_double, gs_max, max, 3, &wrk);

  for (uint d = 0; d < nd; d++) {
    data->mnmx[d][0] = min[d];
    data->mnmx[d][1] = max[d];
  }
}

/*----------------------------------------------------------------------
 *  set_crs_box_dims
 *
 *  Size the structured box grid from the global domain extent and fill
 *  the box vertex coordinate table; zero the per-box matrices.
 *--------------------------------------------------------------------*/
static void set_crs_box_dims(struct asm2 *data) {
  assert(data->nd == 3 && "Only 3D meshes are supported");

  const double xmax = data->mnmx[0][1], xmin = data->mnmx[0][0];
  const double ymax = data->mnmx[1][1], ymin = data->mnmx[1][0];
  const double zmax = data->mnmx[2][1], zmin = data->mnmx[2][0];
  const double tol = data->tol;

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
}

/*----------------------------------------------------------------------
 * find_crs_interp
 *
 * Find the coarse box to which each spectral element belongs to and
 * then find the prolongation operator from local elements to coarse
 * boxes.
 *--------------------------------------------------------------------*/
void find_crs_interp(struct asm2 *const data, const double *const xyz,
                     const double *const centroid) {
  const uint ne = data->ne;
  const uint nv = data->nv;
  const uint nd = data->nd;
  const uint nbx = data->nbx;
  const uint nby = data->nby;
  const uint nbz = data->nbz;
  const ulong nbxyz = (ulong)nbx * (ulong)nby * (ulong)nbz;

  data->lelem_to_gbox = tcalloc(ulong, ne);
  data->phi = tcalloc(double, ne * nv * nv);
  data->gbox_to_lbox = tcalloc(uint, nbxyz);

  const double x0 = data->mnmx[0][0], dx = data->dx[0];
  const double y0 = data->mnmx[1][0], dy = data->dx[1];
  const double z0 = data->mnmx[2][0], dz = data->dx[2];
  uint nbox = 0;
  for (uint e = 0; e < ne; e++) {
    double x = centroid[e * nd + 0];
    double y = centroid[e * nd + 1];
    double z = centroid[e * nd + 2];

    ulong ig = (ulong)(fabs(x - x0) / dx);
    ulong jg = (ulong)(fabs(y - y0) / dy);
    ulong kg = (ulong)(fabs(z - z0) / dz);

    if (ig >= nbx) ig = nbx - 1;
    if (jg >= nby) jg = nby - 1;
    if (kg >= nbz) kg = nbz - 1;
    ulong bg = ig + nbx * jg + nbx * nby * kg;

    data->lelem_to_gbox[e] = bg;
    if (data->gbox_to_lbox[bg] == 0) data->gbox_to_lbox[bg] = ++nbox;

    double xm = x0 + ig * dx, xM = x0 + (ig + 1) * dx;
    double ym = y0 + jg * dy, yM = y0 + (jg + 1) * dy;
    double zm = z0 + kg * dz, zM = z0 + (kg + 1) * dz;

    for (uint v = 0; v < nv; v++) {
      // calculate the reference coordinates.
      double rb = -1.0 + 2 * (xyz[IDX3(e, v, 0)] - xm) / (xM - xm);
      double sb = -1.0 + 2 * (xyz[IDX3(e, v, 1)] - ym) / (yM - ym);
      double tb = -1.0 + 2 * (xyz[IDX3(e, v, 2)] - zm) / (zM - zm);

      // evaluate the hat functions at the reference coordinates.
      double phi0_r = 0.5 * (1 - rb);
      double phi1_r = 0.5 * (1 + rb);
      double phi0_s = 0.5 * (1 - sb);
      double phi1_s = 0.5 * (1 + sb);
      double phi0_t = 0.5 * (1 - tb);
      double phi1_t = 0.5 * (1 + tb);

      data->phi[e * nv * nv + v * nv + 0] = phi0_r * phi0_s * phi0_t;
      data->phi[e * nv * nv + v * nv + 1] = phi1_r * phi0_s * phi0_t;
      data->phi[e * nv * nv + v * nv + 2] = phi0_r * phi1_s * phi0_t;
      data->phi[e * nv * nv + v * nv + 3] = phi1_r * phi1_s * phi0_t;
      data->phi[e * nv * nv + v * nv + 4] = phi0_r * phi0_s * phi1_t;
      data->phi[e * nv * nv + v * nv + 5] = phi1_r * phi0_s * phi1_t;
      data->phi[e * nv * nv + v * nv + 6] = phi0_r * phi1_s * phi1_t;
      data->phi[e * nv * nv + v * nv + 7] = phi1_r * phi1_s * phi1_t;
    }
  }

  data->nbox = nbox;
}

/*----------------------------------------------------------------------
 * assemble_box_system
 *
 * Build the structured-grid (box) coarse operator.
 *
 * Fortran (nrs_set_global_crs / box_src.f) computes, per element e:
 *
 *     Abox(:,:,ilb) += phi_e^T * A_e * phi_e
 *
 * where A_e is the 8x8 SEM Galerkin operator for element e (`asem`),
 * phi_e(iv,jb) is the value of coarse box basis function jb at SEM
 * vertex iv (already Dirichlet-masked), and ilb is the local index of
 * the box that element e belongs to.  This mirrors the `asem`/`abox`
 * accumulation loop in the Fortran.
 *
 * Storage note (C vs Fortran):
 *   - Fortran arrays asem(lcr,lcr,e), phi_e(lcr,lcr,e), abox(lcr,lcr,l)
 *     are column-major: element (i,j) lives at i + lcr*j.
 *   - Here `A` arrives in the layout used by crs_asm1_setup(): row-major
 *     8x8 blocks, A[e*nv*nv + i*nv + j] = A_e(row i, col j).
 *   - data->phi is row-major with row = SEM vertex, col = box basis:
 *     phi[e*nv*nv + iv*nv + jb] = phi_e(iv, jb), matching the Fortran
 *     phi_e(iv,jb) indexing.
 *   - We store abox row-major as well: abox[l*nv*nv + a*nv + b].
 *--------------------------------------------------------------------*/
static void assemble_box_system(struct asm2 *const data,
                                const double *const A) {
  const uint ne = data->ne;
  const uint nv = data->nv;
  const uint nbox = data->nbox;

  data->abox = tcalloc(double, (size_t)nbox * nv * nv);

  for (uint e = 0; e < ne; e++) {
    /* Local (compressed) box index for this element; gbox_to_lbox is
     * 1-based (0 == inactive), so subtract 1 to index abox. */
    const ulong bg = data->lelem_to_gbox[e];
    const uint ilb = data->gbox_to_lbox[bg] - 1;

    const double *const Ae = &A[(size_t)e * nv * nv];          /* Ae(i,j) */
    const double *const phi = &data->phi[(size_t)e * nv * nv]; /* phi(iv,jb) */
    double *const Ac = &data->abox[(size_t)ilb * nv * nv];     /* Ac(a,b) */

    /* Ac(a,b) += sum_{i,j} phi(i,a) * Ae(i,j) * phi(j,b) */
    for (uint a = 0; a < nv; a++) {
      for (uint b = 0; b < nv; b++) {
        double sum = 0.0;
        for (uint i = 0; i < nv; i++) {
          double aij_phi = 0.0;
          for (uint j = 0; j < nv; j++)
            aij_phi += Ae[i * nv + j] * phi[j * nv + b];
          sum += phi[i * nv + a] * aij_phi;
        }
        Ac[a * nv + b] += sum;
      }
    }
  }
}

/*----------------------------------------------------------------------
 * free_asm2_data
 *
 * Free the arrays allocated inside the asm2 struct.
 *--------------------------------------------------------------------*/
static void free_asm2_data(struct asm2 *const data) {
  if (data == NULL) return;
  free(data->lelem_to_gbox), data->lelem_to_gbox = NULL;
  free(data->gbox_to_lbox), data->gbox_to_lbox = NULL;
  free(data->phi), data->phi = NULL;
  free(data->abox), data->abox = NULL;
}

struct xxt *crs_asm2_setup(const uint ne, const uint nd, const uint nv,
                           const double *const xyz,
                           const double *const centroid, const double *const A,
                           const uint nbx, const uint nby, const uint nbz,
                           const struct comm *const c) {
  struct asm2 data;
  data.ne = ne, data.nd = nd, data.nv = nv;
  data.nbx = nbx, data.nby = nby, data.nbz = nbz;
  data.tol = 1e-2;

  domain_size(&data, xyz, c);

  set_crs_box_dims(&data);
  if (c->id == 0) {
    printf("%g %g box domain x %d\n", data.mnmx[0][0], data.mnmx[0][1],
           data.nbx);
    printf("%g %g box domain y %d\n", data.mnmx[1][0], data.mnmx[1][1],
           data.nby);
    printf("%g %g box domain z %d\n", data.mnmx[2][0], data.mnmx[2][1],
           data.nbz);
    printf("%g %g %g box dxdydz\n", data.dx[0], data.dx[1], data.dx[2]);
  }

  find_crs_interp(&data, xyz, centroid);

  assemble_box_system(&data, A);

  free_asm2_data(&data);

  return 0;
}

void crs_asm2_solve(occa::memory &o_x, struct xxt *xxt, occa::memory &o_rhs) {}

void crs_asm2_free(struct xxt *xxt) {}

#undef IDX3
