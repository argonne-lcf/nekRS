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

template <typename T>
static T extractQualifierValue(const std::string &txt, const std::string &key, T defaultVal)
{
  std::regex pattern(lowerCase(key) + "=([0-9\\.eE+-]+)");
  std::smatch match;
  const auto solver = lowerCase(txt);

  if (std::regex_search(solver, match, pattern)) {
    if constexpr (std::is_same<T, int>::value) {
      return std::stoi(match[1]);
    } else if constexpr (std::is_same<T, float>::value) {
      return std::stof(match[1]);
    } else if constexpr (std::is_same<T, double>::value) {
      return std::stod(match[1]);
    }
  }

  return defaultVal;
}

void ellipticCoarseFEMGridSetup(elliptic_t *elliptic, bool update)
{
  auto precon = elliptic->precon;
  auto options = elliptic->options;

  MGSolver_t::multigridLevel **levels = precon->MGSolver->levels;
  auto ellipticCoarse = dynamic_cast<pMGLevel *>(levels[elliptic->levels.size() - 1])->elliptic;

  std::vector<hlong> coarseGlobalStarts(platform->comm.mpiCommSize() + 1);
  dlong nnzCoarseA = 0;
  nonZero_t *coarseA;

  if (options.compareArgs("GALERKIN COARSE OPERATOR", "TRUE") ||
      platform->options.compareArgs("GALERKIN COARSE OPERATOR", "TRUE")) {
    ellipticBuildFEMGalerkinHex3D(ellipticCoarse, elliptic, &coarseA, &nnzCoarseA, coarseGlobalStarts.data());
  } else {
    ellipticBuildFEM(ellipticCoarse, &coarseA, &nnzCoarseA, coarseGlobalStarts.data());
  }

  std::vector<hlong> Rows(nnzCoarseA);
  std::vector<hlong> Cols(nnzCoarseA);
  std::vector<dfloat> Vals(nnzCoarseA);

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

  auto &coarseLevel = precon->MGSolver->coarseLevel;

  if (update) {
    coarseLevel->updateMatrix(nnzCoarseA, Rows.data(), Cols.data(), Vals.data());
  } else {
    coarseLevel->setupSolver(coarseGlobalStarts.data(),
                             nnzCoarseA,
                             Rows.data(),
                             Cols.data(),
                             Vals.data(),
                             ellipticCoarse->o_invDegree,
                             ellipticCoarse->ogs,
                             ellipticCoarse->nullspace);
  }
}

void ellipticMultiGridSetup(elliptic_t *elliptic_)
{
  if (platform->comm.mpiRank() == 0) {
    printf("building MG preconditioner ... \n");
  }
  fflush(stdout);

  elliptic_->precon = new precon_t();
  const auto& precon = elliptic_->precon;

  auto& options = elliptic_->options;

  auto elliptic = ellipticBuildMultigridLevelFine(elliptic_);
  auto& mesh = elliptic->mesh;

  std::vector<mesh_t *> meshLevels(mesh->N + 1);
  for (int n = 1; n < mesh->N + 1; n++) {
    meshLevels[n] = new mesh_t();
    meshLevels[n]->Nverts = mesh->Nverts;
    meshLevels[n]->Nfaces = mesh->Nfaces;
    meshLevels[n]->Nfields = mesh->Nfields;

    switch (elliptic->elementType) {
    case HEXAHEDRA:
      meshLoadReferenceNodesHex3D(meshLevels[n], n, 1);
      break;
    }
  }

  // set the number of MG levels and their degree
  std::vector<int> levelDegree(elliptic->levels.size());
  for (int i = 0; i < elliptic->levels.size(); ++i) {
    levelDegree[i] = elliptic->levels[i];
  }

  const auto Nmax = levelDegree[0];
  const auto Nmin = levelDegree[elliptic->levels.size() - 1];

  precon->MGSolver =
      new MGSolver_t(elliptic->timerName, platform->device.occaDevice(), platform->comm.mpiComm(), options);
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
      ellipticOperator(elliptic, o_p, o_Ap);

      platform->device.finish();
      MPI_Barrier(platform->comm.mpiComm());
      const double start = MPI_Wtime();

      for (int test = 0; test < Nsamples; ++test) {
        ellipticOperator(elliptic, o_p, o_Ap);
      }

      platform->device.finish();
      double elapsed = (MPI_Wtime() - start) / Nsamples;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm());

      return elapsed;
    };

    if (platform->options.compareArgs("ENABLE GS COMM OVERLAP", "TRUE")) {
      auto nonOverlappedTime = timeOperator();
      auto callback = [&]() {
        ellipticAx(elliptic,
                   elliptic->mesh->NlocalGatherElements,
                   elliptic->mesh->o_localGatherElementList,
                   o_p,
                   o_Ap);
      };

      elliptic->oogsAx = oogs::setup(elliptic->ogs, 1, 0, ogsPfloat, callback, oogsMode);

      auto overlappedTime = timeOperator();
      if (overlappedTime > nonOverlappedTime) {
        elliptic->oogsAx = elliptic->oogs;
      }

      if (platform->comm.mpiRank() == 0) {
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
    if (platform->comm.mpiRank() == 0) {
      printf("============= BUILDING pMG%d ==================\n", Nmax);
    }

    elliptic->oogs = oogs::setup(elliptic->ogs, 1, 0, ogsPfloat, NULL, oogsMode);
    elliptic->oogsAx = elliptic->oogs;

    levels[0] = new pMGLevel(elliptic, Nmax, options, platform->comm.mpiComm());
    precon->MGSolver->numLevels++;

    autoOverlap(elliptic);
  }

  // build intermediate MGLevels
  for (int n = 1; n < elliptic->levels.size() - 1; n++) {
    const auto Nc = levelDegree[n];
    const auto Nf = levelDegree[n - 1];
    auto fine = ((pMGLevel *)levels[n - 1])->elliptic;
    if (platform->comm.mpiRank() == 0) {
      printf("============= BUILDING pMG%d ==================\n", Nc);
    }

    auto lvl = ellipticBuildMultigridLevel(fine, Nc, Nf);
    lvl->oogs = oogs::setup(lvl->ogs, 1, 0, ogsPfloat, NULL, oogsMode);
    lvl->oogsAx = lvl->oogs;

    levels[n] =
        new pMGLevel(elliptic, meshLevels.data(), fine, lvl, Nf, Nc, options, platform->comm.mpiComm());

    precon->MGSolver->numLevels++;
    autoOverlap(lvl);
  }

  // set up coarse level elliptic->levels.size() - 1
  auto ellipticCoarse = [&]() {
    if (platform->comm.mpiRank() == 0) {
      printf("============= BUILDING COARSE pMG%d ==================\n", Nmin);
    }

    elliptic_t *crs;
    if (Nmax > Nmin) {
      const auto Nc = levelDegree[elliptic->levels.size() - 1];
      const auto Nf = levelDegree[elliptic->levels.size() - 2];
      auto fine = ((pMGLevel *)levels[elliptic->levels.size() - 2])->elliptic;

      crs = ellipticBuildMultigridLevel(fine, Nc, Nf);

      crs->oogs = oogs::setup(crs->ogs, 1, 0, ogsPfloat, NULL, oogsMode);
      crs->oogsAx = crs->oogs;

      levels[elliptic->levels.size() - 1] = new pMGLevel(elliptic,
                                             meshLevels.data(),
                                             fine,
                                             crs,
                                             Nf,
                                             Nc,
                                             options,
                                             platform->comm.mpiComm(),
                                             true);

      if (options.compareArgs("MULTIGRID COARSE SOLVER", "SMOOTHER") ||
          options.compareArgs("MULTIGRID COARSE SOLVER", "CG") ||
          options.compareArgs("MULTIGRID COARSE SOLVER", "GMRES") ||
          options.compareArgs("PRECONDITIONER", "SEMFEM")) {
        autoOverlap(crs);
      }
    } else {
      crs = elliptic;
      levels[elliptic->levels.size() - 1] = new pMGLevel(crs, Nmin, options, platform->comm.mpiComm(), true);
    }
    precon->MGSolver->baseLevel = precon->MGSolver->numLevels;
    precon->MGSolver->numLevels++;

    return crs;
  }();

  if (options.compareArgs("PRECONDITIONER", "SEMFEM")) { // smoothed SEMFEM
    precon->SEMFEMSolver = new SEMFEMSolver_t(ellipticCoarse);
    auto baseLevel = (pMGLevel *)levels[elliptic->levels.size() - 1];
    precon->MGSolver->coarseLevel->solvePtr = [elliptic, baseLevel](occa::memory &o_rhs,
                                                                    occa::memory &o_x) {
      auto &o_res = baseLevel->o_res;
      baseLevel->smooth(o_rhs, o_x, true);
      baseLevel->residual(o_rhs, o_x, o_res);

      auto o_tmp = platform->deviceMemoryPool.reserve<pfloat>(o_x.size());
      elliptic->precon->SEMFEMSolver->run(o_res, o_tmp);

      platform->linAlg->axpby<pfloat>(o_x.size(), 1.0, o_tmp, 1.0, o_x);
      baseLevel->smooth(o_rhs, o_x, false);
    };
  } else if (options.compareArgs("MULTIGRID COARSE GRID DISCRETIZATION", "SEMFEM")) {
    precon->SEMFEMSolver = new SEMFEMSolver_t(ellipticCoarse);
    auto baseLevel = (pMGLevel *)levels[elliptic->levels.size() - 1];
    precon->MGSolver->coarseLevel->solvePtr =
        [elliptic](occa::memory &o_rhs, occa::memory &o_x) {
          elliptic->precon->SEMFEMSolver->run(o_rhs, o_x);
        };
  } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "JPCG")) {
    auto baseLevel = (pMGLevel *)levels[elliptic->levels.size() - 1];
    auto Ax = [baseLevel](const occa::memory &o_p, occa::memory &o_Ap) { baseLevel->Ax(o_p, o_Ap); };

    baseLevel->elliptic->KSP = linearSolverFactory<pfloat>::create(options.getArgs("MULTIGRID COARSE SOLVER"),
                                                                   "crs::" + baseLevel->elliptic->name,
                                                                   baseLevel->elliptic->mesh->Nlocal,
                                                                   baseLevel->elliptic->Nfields,
                                                                   baseLevel->elliptic->fieldOffset,
                                                                   baseLevel->elliptic->o_invDegree,
                                                                   elliptic->nullspace,
                                                                   Ax);

    if (baseLevel->elliptic->options.compareArgs("MULTIGRID COARSE SOLVER", "COMBINED")) {
      ellipticUpdateJacobi(baseLevel->elliptic, baseLevel->elliptic->KSP->o_invDiagA);
    }

    const auto maxNiter =
        extractQualifierValue(baseLevel->options.getArgs("MULTIGRID COARSE SOLVER"), "maxIter", 20);
    const auto tol =
        extractQualifierValue(baseLevel->options.getArgs("MULTIGRID COARSE SOLVER"), "residualTol", 1e-4);

    precon->MGSolver->coarseLevel->solvePtr =
        [maxNiter, tol, baseLevel](occa::memory &o_rhs,
                                   occa::memory &o_x) {
          auto &elliptic = baseLevel->elliptic;

          if (elliptic->options.compareArgs("ELLIPTIC PRECO COEFF FIELD", "TRUE")) {
            ellipticUpdateJacobi(elliptic, elliptic->KSP->o_invDiagA);
          }
          auto &o_res = o_rhs;
          elliptic->KSP->solve(tol, maxNiter, o_res, o_x);
        };
  } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "BOOMERAMG")) {
    ellipticCoarseFEMGridSetup(elliptic);
    if (options.compareArgs("MULTIGRID COARSE SOLVER", "SMOOTHER")) {
      auto baseLevel = (pMGLevel *)levels[elliptic->levels.size() - 1];
      auto& coarseLevel = precon->MGSolver->coarseLevel;

      precon->MGSolver->coarseLevel->solvePtr =
          [baseLevel, &coarseLevel](occa::memory &o_rhs, occa::memory &o_x) {
            auto &o_res = baseLevel->o_res;
            baseLevel->smooth(o_rhs, o_x, true);
            baseLevel->residual(o_rhs, o_x, o_res);

            auto o_tmp = platform->deviceMemoryPool.reserve<pfloat>(baseLevel->Nrows);
            coarseLevel->solve(o_res, o_tmp);

            platform->linAlg->axpby<pfloat>(baseLevel->Nrows, 1.0, o_tmp, 1.0, o_x);
            baseLevel->smooth(o_rhs, o_x, false);
          };
    }
  } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "SMOOTHER")) {
    auto baseLevel = (pMGLevel *)levels[elliptic->levels.size() - 1];
    precon->MGSolver->coarseLevel->solvePtr =
        [baseLevel](occa::memory &o_rhs, occa::memory &o_x) {
          baseLevel->smooth(o_rhs, o_x, true);
        };
  } else {
    nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "unknown MULTIGRID COARSE SOLVER!");
  }

  if (platform->comm.mpiRank() == 0) {
    printf("-----------------------------------------------------------------------\n");
    printf("level|    Type    |                 |     Smoother                    |\n");
    printf("     |            |                 |                                 |\n");
    printf("-----------------------------------------------------------------------\n");
  }

  for (int lev = 0; lev < precon->MGSolver->numLevels; lev++) {
    if (platform->comm.mpiRank() == 0) {
      printf(" %3d ", lev);
    }
    levels[lev]->Report();
  }

  if (platform->comm.mpiRank() == 0) {
    printf("-----------------------------------------------------------------------\n");
  }

  fflush(stdout);
}
