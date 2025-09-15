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

#include "elliptic.h"
#include "ellipticMultiGrid.h"
#include "platform.hpp"
#include "linAlg.hpp"
#include "ellipticParseMultigridSchedule.hpp"
#include "randomVector.hpp"

namespace
{

ChebyshevSmootherType convertSmootherType(SmootherType s)
{
  switch (s) {
  case SmootherType::ASM:
    return ChebyshevSmootherType::ASM;
  case SmootherType::RAS:
    return ChebyshevSmootherType::RAS;
  case SmootherType::JACOBI:
    return ChebyshevSmootherType::JACOBI;
  default:
    nekrsAbort(platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "Invalid configuration hit in convertSmootherType!");
  }

  return ChebyshevSmootherType::ASM;
}

} // namespace

pMGLevel::pMGLevel(elliptic_t *ellipticBase, int Nc, setupAide options_, MPI_Comm comm_, bool _isCoarse)
    : multigridLevel(ellipticBase->mesh->Nelements * ellipticBase->mesh->Np,
                     (ellipticBase->mesh->Nelements) * ellipticBase->mesh->Np,
                     comm_)
{
  isCoarse = _isCoarse;

  elliptic = ellipticBase;
  mesh = elliptic->mesh;
  options = options_;
  degree = Nc;

  this->setupSmoother(ellipticBase);
}

// build a level and connect it to the previous one
pMGLevel::pMGLevel(elliptic_t *ellipticBase, // finest level
                   mesh_t **meshLevels,
                   elliptic_t *ellipticFine,   // previous level
                   elliptic_t *ellipticCoarse, // current level
                   int Nf,
                   int Nc,
                   setupAide options_,
                   MPI_Comm comm_,
                   bool _isCoarse)
    : multigridLevel(ellipticCoarse->mesh->Nelements * ellipticCoarse->mesh->Np,
                     ellipticCoarse->mesh->Np * (ellipticCoarse->mesh->Nelements),
                     comm_)
{

  isCoarse = _isCoarse;
  elliptic = ellipticCoarse;
  mesh = elliptic->mesh;
  options = options_;
  degree = Nc;

  NpF = ellipticFine->mesh->Np;
  o_invDegreeFine = ellipticFine->o_invDegree;

  /* build coarsening and prologation operators to connect levels */
  this->buildCoarsenerQuadHex(meshLevels, Nf, Nc);

  if (!isCoarse || options.compareArgs("MULTIGRID COARSE SOLVER", "SMOOTHER")) {
    this->setupSmoother(ellipticBase);
  }
}

void pMGLevel::updateSetupSmootherChebyshev()
{
  // estimate the max eigenvalue of S*A
  this->maxEig = this->maxEigSmoothAx();

  dfloat minMultiplier = 0.1;
  options.getArgs("MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minMultiplier);

  dfloat maxMultiplier = 1.1;
  options.getArgs("MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxMultiplier);

  lambda0 = minMultiplier * this->maxEig;
  lambda1 = maxMultiplier * this->maxEig;

  // default degree
  if (isCoarse) {
    if (options.compareArgs("MULTIGRID COARSE SOLVER", "SMOOTHER")) {
      UpLegChebyshevDegree = 3;
      DownLegChebyshevDegree = 3;
    } else {
      UpLegChebyshevDegree = 3;
      DownLegChebyshevDegree = 3;
    }
  } else {
    UpLegChebyshevDegree = 3;
    DownLegChebyshevDegree = 3;
  }
  options.getArgs("MULTIGRID CHEBYSHEV DEGREE", UpLegChebyshevDegree);
  options.getArgs("MULTIGRID CHEBYSHEV DEGREE", DownLegChebyshevDegree);
}

void pMGLevel::setupSmoother(elliptic_t *ellipticBase)
{

  const bool useASM = options.compareArgs("MULTIGRID SMOOTHER", "ASM");
  const bool useRAS = options.compareArgs("MULTIGRID SMOOTHER", "RAS");
  const bool useJacobi = options.compareArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI");

  if (useASM || useRAS) {
    smootherType = useASM ? SmootherType::ASM : SmootherType::RAS;
    setupSmootherSchwarz(ellipticBase);
  } else if (useJacobi) {
    smootherType = SmootherType::JACOBI;
    o_invDiagA = platform->device.malloc<pfloat>(mesh->Nlocal);
    ellipticUpdateJacobi(elliptic, o_invDiagA); // required to compute eigenvalues
  } else {
    nekrsAbort(platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "No supported pMGLevel smoother found!");
  }

  if (options.compareArgs("MULTIGRID SMOOTHER", "CHEBYSHEV")) {
    chebySmootherType = convertSmootherType(smootherType);
    smootherType = SmootherType::CHEBYSHEV;
    updateSetupSmootherChebyshev();
  }

  std::string schedule = options.getArgs("MULTIGRID SCHEDULE");
  if (!schedule.empty()) {
    auto [scheduleMap, errorString] =
        ellipticParseMultigridSchedule(schedule, options, DownLegChebyshevDegree);
    if (scheduleMap[{degree, true}] > -1) {
      UpLegChebyshevDegree = scheduleMap[{degree, true}];
    }
    if (scheduleMap[{degree, false}] > -1) {
      DownLegChebyshevDegree = scheduleMap[{degree, false}];
    }
  }

  if (options.compareArgs("MULTIGRID SMOOTHER", "FOURTHOPT")) {
    UpLegBetas = optimalCoeffs(UpLegChebyshevDegree);
    DownLegBetas = optimalCoeffs(DownLegChebyshevDegree);
    smootherType = SmootherType::OPT_FOURTH_CHEBYSHEV;
  } else if (options.compareArgs("MULTIGRID SMOOTHER", "FOURTH")) {
    // same as above, but beta_i = 1 for all i
    UpLegBetas = std::vector<pfloat>(UpLegChebyshevDegree, 1.0);
    DownLegBetas = std::vector<pfloat>(DownLegChebyshevDegree, 1.0);
    smootherType = SmootherType::FOURTH_CHEBYSHEV;
  }
}

void pMGLevel::Report()
{

  std::string smootherString;
  {
    if (smootherType == SmootherType::CHEBYSHEV) {
      smootherString += "1st Kind Chebyshev+";
    }
    if (smootherType == SmootherType::FOURTH_CHEBYSHEV) {
      smootherString += "4th Kind Chebyshev+";
    }
    if (smootherType == SmootherType::OPT_FOURTH_CHEBYSHEV) {
      smootherString += "Opt. 4th Kind Chebyshev+";
    }
    if (smootherType == SmootherType::ASM || chebySmootherType == ChebyshevSmootherType::ASM) {
      smootherString += "ASM";
    }
    if (smootherType == SmootherType::RAS || chebySmootherType == ChebyshevSmootherType::RAS) {
      smootherString += "RAS";
    }
    if (smootherType == SmootherType::JACOBI || chebySmootherType == ChebyshevSmootherType::JACOBI) {
      smootherString += "Jacobi";
    }
    if (options.compareArgs("MULTIGRID SMOOTHER", "CHEBYSHEV")) {
      smootherString +=
          "(" + std::to_string(UpLegChebyshevDegree) + "," + std::to_string(DownLegChebyshevDegree) + ")";
    }
  }

  if (platform->comm.mpiRank() == 0) {
    if (isCoarse) {
      std::string spaces = "";
      if (options.compareArgs("MULTIGRID COARSE SOLVER", "SMOOTHER") ||
          options.compareArgs("PRECONDITIONER", "SEMFEM")) {
        printf("|    pMG     |   Matrix-free   | %s\n", smootherString.c_str());
        printf("     |            |     p = %2d      |\n", degree);
        spaces = "     ";
      }

      if (options.compareArgs("PRECONDITIONER", "SEMFEM") ||
          options.compareArgs("MULTIGRID COARSE GRID DISCRETIZATION", "SMEFEM")) {
        printf("%s|    AMG     |   SEMFEM Matrix | \n", spaces.c_str());
      } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "BOOMERAMG")) {
        printf("%s|    AMG     |   FEM Matrix    | \n", spaces.c_str());
      } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "CG")) {
        printf("|    pMG     |   Matrix-free   | %s\n", "Krylov");
        printf("     |            |     p = %2d      |\n", degree);
      }
    } else {
      printf("|    pMG     |   Matrix-free   | %s\n", smootherString.c_str());
      printf("     |            |     p = %2d      |\n", degree);
    }
  }
}

void pMGLevel::buildCoarsenerQuadHex(mesh_t **meshLevels, int Nf, int Nc)
{

  const int Nfq = Nf + 1;
  const int Ncq = Nc + 1;
  dfloat *cToFInterp = (dfloat *)calloc(Nfq * Ncq, sizeof(dfloat));
  InterpolationMatrix1D(Nc, Ncq, meshLevels[Nc]->r, Nfq, meshLevels[Nf]->r, cToFInterp);

  pfloat *R = (pfloat *)calloc(Nfq * Ncq, sizeof(pfloat));
  // transpose
  for (int i = 0; i < Ncq; i++) {
    for (int j = 0; j < Nfq; j++) {
      R[i * Nfq + j] = cToFInterp[j * Ncq + i];
    }
  }

  o_R = platform->device.malloc<pfloat>(Nfq * Ncq);
  o_R.copyFrom(R);

  free(R);
  free(cToFInterp);
}

static void eigenValue(const int Nrows, double *A, double *WR, double *WI)
{
  int NB = 256;
  char JOBVL = 'V';
  char JOBVR = 'V';
  int N = Nrows;
  int LDA = Nrows;
  int LWORK = (NB + 2) * N;

  auto WORK = new double[LWORK];
  auto VL = new double[Nrows * Nrows];
  auto VR = new double[Nrows * Nrows];

  bool invalid = false;
  for (int i = 0; i < Nrows * Nrows; i++) {
    invalid |= std::isnan(A[i]) || std::isinf(A[i]);
  }
  nekrsCheck(invalid, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "invalid matrix entries!");

  int INFO = -999;
  dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDA, VR, &LDA, WORK, &LWORK, &INFO);

  nekrsCheck(INFO != 0, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "dgeev failed");

  delete[] VL;
  delete[] VR;
  delete[] WORK;
}

dfloat pMGLevel::maxEigSmoothAx()
{
  MPI_Barrier(platform->comm.mpiComm());
  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank() == 0) {
    printf("estimating maxEigenvalue ... ");
  }
  fflush(stdout);

  const dlong M = Ncols;

  hlong Nlocal = (hlong)Nrows;
  hlong Nglobal = 0;
  MPI_Allreduce(&Nlocal, &Nglobal, 1, MPI_HLONG, MPI_SUM, platform->comm.mpiComm());

  auto o_invDegree = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);
  o_invDegree.copyFrom(elliptic->ogs->invDegree);
  const auto k = (unsigned int)std::min(pMGLevel::Narnoldi, Nglobal);

  std::vector<double> H(k * k, 0.0);

  auto Vx = randomVector<dfloat>(M, 0.0, 1.0, true); // deterministic random numbers

  std::vector<occa::memory> o_V(k + 1);
  for (int i = 0; i <= k; i++) {
    o_V[i] = platform->deviceMemoryPool.reserve<dfloat>(M);
  }

  auto o_Vx = platform->deviceMemoryPool.reserve<dfloat>(M);
  auto o_VxPfloat = platform->deviceMemoryPool.reserve<pfloat>(M);

  auto o_AVx = platform->deviceMemoryPool.reserve<dfloat>(M);
  auto o_AVxPfloat = platform->deviceMemoryPool.reserve<pfloat>(M);

  if (options.compareArgs("DISCRETIZATION", "CONTINUOUS")) {
    ogsGatherScatter(Vx.data(), ogsDfloat, ogsAdd, mesh->ogs);

    if (elliptic->Nmasked > 0) {
      dlong *maskIds = (dlong *)calloc(elliptic->Nmasked, sizeof(dlong));
      elliptic->o_maskIds.copyTo(maskIds, elliptic->Nmasked);
      for (dlong i = 0; i < elliptic->Nmasked; i++) {
        Vx[maskIds[i]] = 0.;
      }
      free(maskIds);
    }
  }

  o_Vx.copyFrom(Vx.data(), M);
  platform->linAlg->fill(Nlocal, 0.0, o_V[0]);

  dfloat norm_v0 = platform->linAlg->weightedInnerProdMany(Nlocal,
                                                           elliptic->Nfields,
                                                           elliptic->fieldOffset,
                                                           o_invDegree,
                                                           o_Vx,
                                                           o_Vx,
                                                           platform->comm.mpiComm());
  nekrsCheck(norm_v0 <= 0, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "invalid v0 norm!");
  norm_v0 = sqrt(norm_v0);

  // normalize
  platform->linAlg
      ->axpbyMany(Nlocal, elliptic->Nfields, elliptic->fieldOffset, 1. / norm_v0, o_Vx, 0.0, o_V[0]);

  // Arnoldi
  for (int j = 0; j < k; j++) {
    // v[j+1] = invD*(A*v[j])
    {
      platform->copyDfloatToPfloatKernel(M, o_V[j], o_VxPfloat);
      ellipticOperator(elliptic, o_VxPfloat, o_AVxPfloat);
      this->smoother(o_AVxPfloat, o_VxPfloat, true);
      platform->copyPfloatToDfloatKernel(M, o_VxPfloat, o_V[j + 1]);
    }

    // modified Gram-Schmidth
    for (int i = 0; i <= j; i++) {
      // H(i,j) = v[i]'*A*v[j]
      const auto hij = platform->linAlg->weightedInnerProdMany(Nlocal,
                                                               elliptic->Nfields,
                                                               elliptic->fieldOffset,
                                                               o_invDegree,
                                                               o_V[i],
                                                               o_V[j + 1],
                                                               platform->comm.mpiComm());

      // v[j+1] = v[j+1] - hij*v[i]
      platform->linAlg
          ->axpbyMany(Nlocal, elliptic->Nfields, elliptic->fieldOffset, -hij, o_V[i], 1.0, o_V[j + 1]);

      H[i + j * k] = (double)hij;
    }

    if (j + 1 < k) {
      // v[j+1] = v[j+1]/||v[j+1]||
      auto norm_vj = platform->linAlg->weightedInnerProdMany(Nlocal,
                                                             elliptic->Nfields,
                                                             elliptic->fieldOffset,
                                                             o_invDegree,
                                                             o_V[j + 1],
                                                             o_V[j + 1],
                                                             platform->comm.mpiComm());

      nekrsCheck(norm_vj <= 0, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "invalid vj norm!");
      norm_vj = sqrt(norm_vj);

      platform->linAlg->scaleMany(Nlocal, elliptic->Nfields, elliptic->fieldOffset, 1 / norm_vj, o_V[j + 1]);
      H[j + 1 + j * k] = (double)norm_vj;
    }
  }

  std::vector<double> WR(k, 0.0);
  std::vector<double> WI(k, 0.0);

  eigenValue(k, H.data(), WR.data(), WI.data());

  double rho = 0.;

  for (int i = 0; i < k; i++) {
    double rho_i = sqrt(WR[i] * WR[i] + WI[i] * WI[i]);

    if (rho < rho_i) {
      rho = rho_i;
    }
  }

  MPI_Barrier(platform->comm.mpiComm());
  if (platform->comm.mpiRank() == 0) {
    printf("value=%g done (%gs)\n", rho, MPI_Wtime() - tStart);
  }
  fflush(stdout);

  return rho;
}
