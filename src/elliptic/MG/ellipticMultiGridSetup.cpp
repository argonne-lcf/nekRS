/*

   The MIT License (MIT)

   Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

 */

#include "platform.hpp"
#include "elliptic.h"
#include "ellipticPrecon.h"
#include "ellipticMultiGrid.h"
#include "ellipticBuildFEM.hpp"

#include "crs.hpp"

//////////////////////////////////////////////////////////////////////////////////////
// Helper functions for jl coarse solver                                            //
//////////////////////////////////////////////////////////////////////////////////////
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
  if (jj % 2 == 0) memcpy(zr, z1, Nq * sizeof(dfloat));
  if (jj == 3 || jj == 4 || jj == 7 || jj == 8)
    memcpy(zs, z1, Nq * sizeof(dfloat));
  if (jj > 4) memcpy(zt, z1, Nq * sizeof(dfloat));

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

  dfloat *b = (dfloat *)calloc(nc * Np, sizeof(dfloat));
  for (int j = 0; j < nc; j++) gen_crs_basis(b, j, mf->gllz, mf->Nq, mf->Np);

  dfloat *u = (dfloat *)calloc(size, sizeof(dfloat));
  dfloat *w = (dfloat *)calloc(size, sizeof(dfloat));

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
    ellipticAx(ef, mf->Nelements, mf->o_elementList, o_upf, o_wpf,
               pfloatString);
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

void jl_setup_aux(uint *ntot_, ulong **gids_, uint *nnz_, uint **ia_,
                  uint **ja_, double **a_, double **xyz_, double **centroid_,
                  elliptic_t *elliptic, elliptic_t *ellipticf) {
  mesh_t *mesh = elliptic->mesh, *meshf = ellipticf->mesh;
  assert(mesh->Nelements == meshf->Nelements);
  uint nelt = meshf->Nelements, nc = mesh->Np;

  uint ntot = *ntot_ = nelt * nc;
  ulong *gids = *gids_ = (ulong *)calloc(ntot, sizeof(ulong));
  for (int j = 0; j < nelt * nc; j++) gids[j] = mesh->globalIds[j];

  if (elliptic->Nmasked) {
    dlong *mask_ids = (dlong *)calloc(elliptic->Nmasked, sizeof(dlong));
    elliptic->o_maskIds.copyTo(mask_ids, elliptic->Nmasked);
    for (int n = 0; n < elliptic->Nmasked; n++) gids[mask_ids[n]] = 0;
    free(mask_ids);
  }

  // Set coordinates.
  uint ndim = mesh->dim;
  uint nv   = (ndim == 3) ? 8 : 4;
  double *xyz = *xyz_ = (double *)calloc(ntot * ndim, sizeof(double));
  uint count = 0;
  for (uint e = 0; e < nelt; e++) {
    for (uint v = 0; v < nv; v++) {
      xyz[count++] = mesh->EX[e * nv + v];
      xyz[count++] = mesh->EY[e * nv + v];
      xyz[count++] = mesh->EZ[e * nv + v];
    }
  }

  // Set element centroids by interpolating the fine-mesh GLL coordinates to the
  // single GL point at the reference-element center (r,s,t)=(0,0,0). Because this
  // acts on the actual (curved) node coordinates, it is exact for curved elements.
  double *centroid = *centroid_ = (double *)calloc(nelt * ndim, sizeof(double));
  {
    // Build the 1D interpolation row from the Nq fine GLL nodes to the single
    // point r=0 (the reference-element center along one axis).
    //   in : meshf->N          polynomial order N (interpolant has degree N)
    //        Nq = N+1          number of source nodes (required = N+1)
    //        meshf->r          source node coordinates on [-1,1] (fine GLL nodes)
    //        1, &rc            one target point, at rc = 0
    //   out: w                 1 x Nq row; w[i] is the weight of source node i,
    //                          i.e. p(0) = sum_i w[i] * f[i] for the degree-N
    //                          polynomial p interpolating values f at the nodes.
    int Nq = meshf->Nq, Np = meshf->Np;
    dfloat rc = 0.0;
    std::vector<dfloat> w(Nq);
    InterpolationMatrix1D(meshf->N, Nq, meshf->r, 1, &rc, w.data());

    // Curved GLL coordinates of the fine mesh.
    auto [x, y, z] = meshf->xyzHost();

    // The tensor-product interpolant evaluated at the center is the triple sum
    //
    //   c = \sum_{i,j,k} w_i w_j w_k \, f_{ijk},
    //
    // where w is the 1D interpolation row to r=0 and f_{ijk} are the element's
    // node coordinates. Because the weight factorizes as w_i w_j w_k, this is
    // equivalent to three sequential 1D contractions (sum-factorization),
    //
    //   c = \sum_k w_k \Big( \sum_j w_j \big( \sum_i w_i f_{ijk} \big) \Big),
    //
    // which is cheaper when there are many output points. Here there is a single
    // output point per element, so we apply the fused triple sum directly; the
    // two forms agree up to floating-point round-off.
    for (uint e = 0; e < nelt; e++) {
      double cx = 0, cy = 0, cz = 0;
      for (int k = 0; k < Nq; k++) {
        for (int j = 0; j < Nq; j++) {
          for (int i = 0; i < Nq; i++) {
            const double wijk = (double)w[i] * w[j] * w[k];
            const dlong id = e * Np + i + j * Nq + k * Nq * Nq;
            cx += wijk * x[id];
            cy += wijk * y[id];
            cz += wijk * z[id];
          }
        }
      }
      centroid[e * ndim + 0] = cx;
      centroid[e * ndim + 1] = cy;
      centroid[e * ndim + 2] = cz;
    }
  }

  // Set coarse matrix
  uint nnz = *nnz_ = nc * nc * nelt;
  double *a = *a_ = (double *)calloc(nnz, sizeof(double));
  get_local_crs_galerkin(a, nc, meshf, ellipticf);

  uint *ia = *ia_ = (uint *)calloc(nnz, sizeof(uint));
  uint *ja = *ja_ = (uint *)calloc(nnz, sizeof(uint));
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


//////////////////////////////////////////////////////////////////////////////////////
// ellipticMulriGridSetup                                                           //
//////////////////////////////////////////////////////////////////////////////////////
void ellipticMultiGridSetup(elliptic_t *elliptic_)
{
  if (platform->comm.mpiRank == 0) {
    printf("building MG preconditioner ... \n");
  }
  fflush(stdout);

  elliptic_->precon = new precon_t();
  const auto precon = elliptic_->precon;

  // setup new object from fine grid but with constant coeff
  elliptic_t *elliptic = ellipticBuildMultigridLevelFine(elliptic_);
  setupAide options = elliptic_->options;
  mesh_t *mesh = elliptic->mesh;

  // read all the nodes files and load them in a dummy mesh array
  std::vector<mesh_t *> meshLevels(mesh->N + 1);
  for (int n = 1; n < mesh->N + 1; n++) {
    meshLevels[n] = new mesh_t();
    meshLevels[n]->Nverts = mesh->Nverts;
    meshLevels[n]->Nfaces = mesh->Nfaces;
    meshLevels[n]->Nfields = mesh->Nfields; // TW: ahem

    switch (elliptic->elementType) {
    case HEXAHEDRA:
      meshLoadReferenceNodesHex3D(meshLevels[n], n, 1);
      break;
    }
  }

  // set the number of MG levels and their degree
  int numMGLevels = elliptic->nLevels;
  std::vector<int> levelDegree(numMGLevels);
  for (int i = 0; i < numMGLevels; ++i) {
    levelDegree[i] = elliptic->levels[i];
  }

  int Nmax = levelDegree[0];
  int Nmin = levelDegree[numMGLevels - 1];

  precon->MGSolver = new MGSolver_t(platform->device.occaDevice(), platform->comm.mpiComm, options);
  MGSolver_t::multigridLevel **levels = precon->MGSolver->levels;

  oogs_mode oogsMode = OOGS_AUTO;

  auto autoOverlap = [&](elliptic_t *elliptic) {
    if (!options.compareArgs("MULTIGRID SMOOTHER", "CHEBYSHEV")) {
      return;
    }

    auto o_p = platform->deviceMemoryPool.reserve<pfloat>(mesh->Nlocal);
    auto o_Ap = platform->deviceMemoryPool.reserve<pfloat>(mesh->Nlocal);

    auto timeOperator = [&]() {
      const int Nsamples = 10;
      ellipticOperator(elliptic, o_p, o_Ap, pfloatString);

      platform->device.finish();
      MPI_Barrier(platform->comm.mpiComm);
      const double start = MPI_Wtime();

      for (int test = 0; test < Nsamples; ++test) {
        ellipticOperator(elliptic, o_p, o_Ap, pfloatString);
      }

      platform->device.finish();
      double elapsed = (MPI_Wtime() - start) / Nsamples;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm);

      return elapsed;
    };

    if (platform->options.compareArgs("ENABLE GS COMM OVERLAP", "TRUE")) {
      auto nonOverlappedTime = timeOperator();
      auto callback = [&]() {
        ellipticAx(elliptic,
                   elliptic->mesh->NlocalGatherElements,
                   elliptic->mesh->o_localGatherElementList,
                   o_p,
                   o_Ap,
                   pfloatString);
      };

      elliptic->oogsAx = oogs::setup(elliptic->ogs, 1, 0, ogsPfloat, callback, oogsMode);

      auto overlappedTime = timeOperator();
      if (overlappedTime > nonOverlappedTime) {
        elliptic->oogsAx = elliptic->oogs;
      }

      if (platform->comm.mpiRank == 0) {
        printf("testing overlap in ellipticOperator: %.2es %.2es ", nonOverlappedTime, overlappedTime);
        if (elliptic->oogsAx != elliptic->oogs) {
          printf("(overlap enabled)");
        }

        printf("\n");
      }
    }
  };

  // set up the finest level 0
  if (Nmax > Nmin) {
    if (platform->comm.mpiRank == 0) {
      printf("============= BUILDING pMG%d ==================\n", Nmax);
    }

    elliptic->oogs = oogs::setup(elliptic->ogs, 1, 0, ogsPfloat, NULL, oogsMode);
    elliptic->oogsAx = elliptic->oogs;

    levels[0] = new pMGLevel(elliptic, Nmax, options, platform->comm.mpiComm);
    precon->MGSolver->numLevels++;

    autoOverlap(elliptic);
  }

  // build intermediate MGLevels
  for (int n = 1; n < numMGLevels - 1; n++) {
    int Nc = levelDegree[n];
    int Nf = levelDegree[n - 1];
    elliptic_t *ellipticFine = ((pMGLevel *)levels[n - 1])->elliptic;
    if (platform->comm.mpiRank == 0) {
      printf("============= BUILDING pMG%d ==================\n", Nc);
    }

    elliptic_t *ellipticC = ellipticBuildMultigridLevel(ellipticFine, Nc, Nf);

    ellipticC->oogs = oogs::setup(ellipticC->ogs, 1, 0, ogsPfloat, NULL, oogsMode);
    ellipticC->oogsAx = ellipticC->oogs;

    levels[n] = new pMGLevel(elliptic,
                             meshLevels.data(),
                             ellipticFine,
                             ellipticC,
                             Nf,
                             Nc,
                             options,
                             platform->comm.mpiComm);
    precon->MGSolver->numLevels++;

    autoOverlap(ellipticC);
  }

  // set up coarse level numMGLevels - 1
  elliptic_t *ellipticCoarse;
  if (platform->comm.mpiRank == 0) {
    printf("============= BUILDING COARSE pMG%d ==================\n", Nmin);
  }

  if (Nmax > Nmin) {
    int Nc = levelDegree[numMGLevels - 1];
    int Nf = levelDegree[numMGLevels - 2];
    elliptic_t *ellipticFine = ((pMGLevel *)levels[numMGLevels - 2])->elliptic;

    ellipticCoarse = ellipticBuildMultigridLevel(ellipticFine, Nc, Nf);

    ellipticCoarse->oogs = oogs::setup(ellipticCoarse->ogs, 1, 0, ogsPfloat, NULL, oogsMode);
    ellipticCoarse->oogsAx = ellipticCoarse->oogs;

    levels[numMGLevels - 1] = new pMGLevel(elliptic,
                                           meshLevels.data(),
                                           ellipticFine,
                                           ellipticCoarse,
                                           Nf,
                                           Nc,
                                           options,
                                           platform->comm.mpiComm,
                                           true);

    if (options.compareArgs("MULTIGRID COARSE SOLVE", "FALSE") ||
        options.compareArgs("MULTIGRID COARSE SOLVE AND SMOOTH", "TRUE")) {
      autoOverlap(ellipticCoarse);
    }
  } else {
    ellipticCoarse = elliptic;
    levels[numMGLevels - 1] = new pMGLevel(ellipticCoarse, Nmin, options, platform->comm.mpiComm, true);
  }
  precon->MGSolver->baseLevel = precon->MGSolver->numLevels;
  precon->MGSolver->numLevels++;

  if (options.compareArgs("MULTIGRID COARSE SOLVE", "TRUE")) {
    if (options.compareArgs("MULTIGRID SEMFEM", "TRUE")) {
      precon->SEMFEMSolver = new SEMFEMSolver_t(ellipticCoarse);
      if (options.compareArgs("MULTIGRID COARSE SOLVE AND SMOOTH", "TRUE")) {
        auto baseLevel = (pMGLevel *)levels[numMGLevels - 1];

        precon->MGSolver->coarseLevel->solvePtr =
            [elliptic,
             baseLevel](MGSolver_t::coarseLevel_t *coarseLevel, occa::memory &o_rhs,
                 occa::memory &o_x) {
              auto &o_res = baseLevel->o_res;
              baseLevel->smooth(o_rhs, o_x, true);
              baseLevel->residual(o_rhs, o_x, o_res);

              auto o_tmp = platform->deviceMemoryPool.reserve<pfloat>(o_x.size());
              elliptic->precon->SEMFEMSolver->run(o_res, o_tmp);

              platform->linAlg->paxpby(o_x.size(), 1.0, o_tmp, 1.0, o_x);
              baseLevel->smooth(o_rhs, o_x, false);
            };
      } else {
        precon->MGSolver->coarseLevel->solvePtr =
            [elliptic](MGSolver_t::coarseLevel_t *, occa::memory &o_rhs,
                occa::memory &o_x) {
              elliptic->precon->SEMFEMSolver->run(o_rhs, o_x);
            };
      }
    } else {
      hlong *coarseGlobalStarts = (hlong *)calloc(platform->comm.mpiCommSize + 1,
          sizeof(hlong));

      int xxt = options.compareArgs("COARSE SOLVER", "XXT");
      int asm1 = options.compareArgs("COARSE SOLVER", "ASM1");
      if (xxt || asm1) {
        uint n, nnz;
        ulong *gids;
        uint *ia, *ja;
        double *va, *xyz, *centroid;
        jl_setup_aux(&n, &gids, &nnz, &ia, &ja, &va, &xyz, &centroid,
            ellipticCoarse, elliptic);

        jl_opts opts;
        opts.algo =  xxt ? XXT : ASM1;
        opts.null_space = elliptic->nullspace;
        opts.dom = gs_float;
        opts.nw = 1;
        jl_setup(n, gids, nnz, ia, ja, va, xyz, centroid, &opts,
            platform->comm.mpiComm);

        int rank = platform->comm.mpiRank;
        coarseGlobalStarts[rank] = 0;
        coarseGlobalStarts[rank + 1] = n;

        precon->MGSolver->coarseLevel->setupSolver(coarseGlobalStarts, 0, 0, 0, 0,
            elliptic->nullspace);

        free(gids), free(ia), free(ja), free(va), free(xyz), free(centroid);
      } else {
        nonZero_t *coarseA;
        dlong nnzCoarseA;

        if (options.compareArgs("GALERKIN COARSE OPERATOR", "TRUE")) {
          ellipticBuildFEMGalerkinHex3D(ellipticCoarse, elliptic, &coarseA,
              &nnzCoarseA, coarseGlobalStarts);
        } else {
          ellipticBuildFEM(ellipticCoarse, &coarseA, &nnzCoarseA, coarseGlobalStarts);
        }

        hlong *Rows = (hlong *)calloc(nnzCoarseA, sizeof(hlong));
        hlong *Cols = (hlong *)calloc(nnzCoarseA, sizeof(hlong));
        dfloat *Vals = (dfloat *)calloc(nnzCoarseA, sizeof(dfloat));

        for (dlong i = 0; i < nnzCoarseA; i++) {
          Rows[i] = coarseA[i].row;
          Cols[i] = coarseA[i].col;
          Vals[i] = coarseA[i].val;

          nekrsCheck(Rows[i] < 0 || Cols[i] < 0 || std::isnan(Vals[i]),
                     MPI_COMM_SELF,
                     EXIT_FAILURE,
                     "invalid {row %lld, col %lld , val %g}\n",
                     Rows[i],
                     Cols[i],
                     Vals[i]);
        }
        free(coarseA);

        precon->MGSolver->coarseLevel
            ->setupSolver(coarseGlobalStarts, nnzCoarseA, Rows, Cols, Vals, elliptic->nullspace);

        free(coarseGlobalStarts);
        free(Rows);
        free(Cols);
        free(Vals);
      }

      MGSolver_t::coarseLevel_t *coarseLevel = precon->MGSolver->coarseLevel;
      coarseLevel->ogs = ellipticCoarse->ogs;

      coarseLevel->o_weight = ellipticCoarse->o_invDegree;
      coarseLevel->weight = (pfloat *)calloc(ellipticCoarse->mesh->Nlocal, sizeof(pfloat));
      coarseLevel->o_weight.copyTo(coarseLevel->weight, ellipticCoarse->mesh->Nlocal);

      coarseLevel->h_Gx = platform->device.mallocHost<pfloat>(coarseLevel->ogs->Ngather);
      coarseLevel->Gx = (pfloat *)coarseLevel->h_Gx.ptr();
      coarseLevel->o_Gx = platform->device.malloc<pfloat>(coarseLevel->ogs->Ngather);

      coarseLevel->h_Sx = platform->device.mallocHost<pfloat>(ellipticCoarse->mesh->Nlocal);
      coarseLevel->Sx = (pfloat *)coarseLevel->h_Sx.ptr();
      coarseLevel->o_Sx = platform->device.malloc<pfloat>(ellipticCoarse->mesh->Nlocal);

      if (options.compareArgs("MULTIGRID COARSE SOLVE AND SMOOTH", "TRUE")) {
        auto baseLevel = (pMGLevel *)levels[numMGLevels - 1];

        precon->MGSolver->coarseLevel->solvePtr =
            [baseLevel](MGSolver_t::coarseLevel_t *coarseLevel, occa::memory &o_rhs, occa::memory &o_x) {
              occa::memory o_res = baseLevel->o_res;
              baseLevel->smooth(o_rhs, o_x, true);
              baseLevel->residual(o_rhs, o_x, o_res);

              auto o_tmp = platform->deviceMemoryPool.reserve<pfloat>(baseLevel->Nrows);
              coarseLevel->solve(o_res, o_tmp);

              platform->linAlg->paxpby(baseLevel->Nrows, 1.0, o_tmp, 1.0, o_x);
              baseLevel->smooth(o_rhs, o_x, false);
            };
      }
    }
  } else {
    auto baseLevel = (pMGLevel *)levels[numMGLevels - 1];
    precon->MGSolver->coarseLevel->solvePtr =
        [baseLevel](MGSolver_t::coarseLevel_t *, occa::memory &o_rhs, occa::memory &o_x) {
          baseLevel->smooth(o_rhs, o_x, true);
        };
  }

  if (platform->comm.mpiRank == 0) {
    printf("-----------------------------------------------------------------------\n");
    printf("level|    Type    |                 |     Smoother                    |\n");
    printf("     |            |                 |                                 |\n");
    printf("-----------------------------------------------------------------------\n");
  }

  for (int lev = 0; lev < precon->MGSolver->numLevels; lev++) {
    if (platform->comm.mpiRank == 0) {
      printf(" %3d ", lev);
    }
    levels[lev]->Report();
  }

  if (platform->comm.mpiRank == 0) {
    printf("-----------------------------------------------------------------------\n");
  }

  fflush(stdout);
}
