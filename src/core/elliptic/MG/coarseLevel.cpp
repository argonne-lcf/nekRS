/*

The MIT License (MIT)


Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus, Rajesh Gandham

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

#include "limits.h"
#include "stdio.h"
#include "timer.hpp"

#include "AMGX.hpp"

#include "platform.hpp"
#include "linAlg.hpp"
#include "MGSolver.hpp"

static occa::kernel vectorDotStarKernel;

MGSolver_t::coarseLevel_t::coarseLevel_t(const std::string &name_, setupAide options_, MPI_Comm comm_)
{
  name = name_;
  options = options_;
  comm = comm_;
  solveOnHost = false;
  solvePtr = [this](occa::memory &rhs, occa::memory &x) {
      this->solve(rhs, x);
  };
}

void MGSolver_t::coarseLevel_t::updateMatrix(
    dlong nnz,     //--
    hlong *Ai,     //-- Local A matrix data (globally indexed, COO storage, row sorted)
    hlong *Aj,     //--
    dfloat *Avals) //--
{
  std::string crsSolver;
  options.getArgs("MULTIGRID COARSE SOLVER", crsSolver);

  if (crsSolver.find("BOOMERAMG") != std::string::npos) {
    auto boomerAMG = (hypreWrapper::boomerAMG_t *)this->boomerAMG;

    // convert dfloat to double
    std::vector<double> Av(nnz);
    for (int i = 0; i < Av.size(); i++) {
      Av[i] = Avals[i];
    }

    boomerAMG->setMatrix(nnz, Ai, Aj, Av.data());
    boomerAMG->setup();
  } else {
    nekrsAbort(platform->comm.mpiComm(),
               EXIT_FAILURE,
               "MULTIGRID COARSE SOLVER <%s> is not supported!\n",
               crsSolver.c_str());
  }
}

void MGSolver_t::coarseLevel_t::setupSolver(
    hlong *globalRowStarts,
    dlong nnz,     //--
    hlong *Ai,     //-- Local A matrix data (globally indexed, COO storage, row sorted)
    hlong *Aj,     //--
    dfloat *Avals, //--
    const occa::memory& o_weight_,
    ogs_t *ogs_,
    bool nullSpace)
{
  ogs = ogs_;
  o_weight = o_weight_;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Barrier(comm);
  double startTime = MPI_Wtime();
  if (rank == 0) {
    printf("setup FEM solver ...");
  }
  fflush(stdout);

  N = (dlong)(globalRowStarts[rank + 1] - globalRowStarts[rank]);

  const int verbose = (platform->verbose()) ? 1 : 0;
  const bool useDevice = options.compareArgs("MULTIGRID COARSE SOLVER LOCATION", "DEVICE");
  const int useFP32 = options.compareArgs("MULTIGRID COARSE SOLVER PRECISION", "FP32");

  const std::string kernelName = "vectorDotStar";
  if (!vectorDotStarKernel.isInitialized()) {
    vectorDotStarKernel = platform->kernelRequests.load(kernelName);
  }

  o_xBuffer = platform->device.malloc<pfloat>(N);
  h_xBuffer = platform->device.mallocHost<pfloat>(N);

  h_Gx = platform->device.mallocHost<pfloat>(ogs->Ngather);
  o_Gx = platform->device.malloc<pfloat>(h_Gx.size());

  h_Sx = platform->device.mallocHost<pfloat>(ogs->N);
  o_Sx = platform->device.malloc<pfloat>(h_Sx.size());

  h_weight = platform->device.mallocHost<pfloat>(o_Sx.size());
  o_weight.copyTo(h_weight, h_weight.size());

  // convert dfloat to double
  std::vector<double> Av(nnz);
  for (int i = 0; i < Av.size(); i++) {
    Av[i] = Avals[i];
  }

  if (options.compareArgs("MULTIGRID COARSE SOLVER", "BOOMERAMG")) {

    double settings[hypreWrapperDevice::NPARAM + 1];
    settings[0] = 1;  /* custom settings              */
    settings[1] = 10; /* coarsening                   */
    if (useDevice) {
      settings[1] = 8; /*  HMIS currently not supported on device */
    }
    settings[2] = 6;    /* interpolation                */
    settings[3] = 1;    /* number of cycles             */
    settings[4] = 16;   /* smoother for crs level       */
    settings[5] = 3;    /* number of coarse sweeps      */
    settings[6] = 16;   /* smoother                     */
    settings[7] = 1;    /* number of sweeps             */
    settings[8] = 0.25; /* strong threshold             */
    settings[9] = 0.05; /* non galerkin tol             */
    settings[10] = 0;   /* aggressive coarsening levels */
    settings[11] = 1;   /* chebyRelaxOrder */
    settings[12] = 0.3; /* chebyRelaxOrder */

    options.getArgs("BOOMERAMG COARSEN TYPE", settings[1]);
    options.getArgs("BOOMERAMG INTERPOLATION TYPE", settings[2]);
    options.getArgs("BOOMERAMG COARSE SMOOTHER TYPE", settings[4]);
    options.getArgs("BOOMERAMG SMOOTHER TYPE", settings[6]);
    options.getArgs("BOOMERAMG SMOOTHER SWEEPS", settings[7]);
    options.getArgs("BOOMERAMG ITERATIONS", settings[3]);
    options.getArgs("BOOMERAMG STRONG THRESHOLD", settings[8]);
    options.getArgs("BOOMERAMG NONGALERKIN TOLERANCE", settings[9]);
    options.getArgs("BOOMERAMG AGGRESSIVE COARSENING LEVELS", settings[10]);
    options.getArgs("BOOMERAMG CHEBYSHEV RELAX ORDER", settings[11]);
    options.getArgs("BOOMERAMG CHEBYSHEV FRACTION", settings[12]);

    if (useDevice) {
      boomerAMG = new hypreWrapperDevice::boomerAMG_t(N,
                                                      nnz,
                                                      Ai,
                                                      Aj,
                                                      Av.data(),
                                                      (int)nullSpace,
                                                      comm,
                                                      platform->device.occaDevice(),
                                                      useFP32,
                                                      settings,
                                                      verbose);
    } else {
      const int Nthreads = 1;
      boomerAMG = new hypreWrapper::boomerAMG_t(N,
                                                nnz,
                                                Ai,
                                                Aj,
                                                Av.data(),
                                                (int)nullSpace,
                                                comm,
                                                Nthreads,
                                                useFP32,
                                                settings,
                                                verbose);
    }
  } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "AMGX")) {
    std::string configFile;
    platform->options.getArgs("AMGX CONFIG FILE", configFile);
    char *cfg = NULL;
    if (configFile.size()) {
      cfg = (char *)configFile.c_str();
    }
    AMGX = new AMGX_t(N,
                      nnz,
                      Ai,
                      Aj,
                      Av.data(),
                      (int)nullSpace,
                      comm,
                      platform->device.id(),
                      useFP32,
                      std::stoi(getenv("NEKRS_GPU_MPI")),
                      cfg);
  } else {
    std::string amgSolver;
    options.getArgs("MULTIGRID COARSE SOLVER", amgSolver);
    nekrsAbort(platform->comm.mpiComm(),
               EXIT_FAILURE,
               "MULTIGRID COARSE SOLVER <%s> is not supported!\n",
               amgSolver.c_str());
  }

  MPI_Barrier(comm);
  if (rank == 0) {
    printf("done (%gs)\n", MPI_Wtime() - startTime);
  }
}

MGSolver_t::coarseLevel_t::~coarseLevel_t()
{
  const auto useDevice = options.compareArgs("MULTIGRID COARSE SOLVER LOCATION", "DEVICE");
  if (boomerAMG) {
    if (useDevice) {
      delete (hypreWrapperDevice::boomerAMG_t *)this->boomerAMG;
    } else {
      delete (hypreWrapper::boomerAMG_t *)this->boomerAMG;
    }
  }
  if (AMGX) {
    delete AMGX;
  }
}

void MGSolver_t::coarseLevel_t::solve(occa::memory &o_rhs, occa::memory &o_x)
{
  if (solveOnHost) {
    platform->timer.hostTic(name + " coarseSolve", true);

    // masked E->T
    auto rhsPtr = o_rhs.ptr<pfloat>();
    auto weightPtr = h_weight.ptr<pfloat>();
    auto SxPtr = h_Sx.ptr<pfloat>();
    for (int i = 0; i < ogs->N; i++) {
      SxPtr[i] = rhsPtr[i] * weightPtr[i];
    }

    ogsGather(h_Gx.ptr<pfloat>(), h_Sx.ptr<pfloat>(), ogsPfloat, ogsAdd, ogs);

    auto xBufferPtr = h_xBuffer.ptr<pfloat>();
    for (int i = 0; i < N; i++) {
      xBufferPtr[i] = 0;
    }
    auto boomerAMG = (hypreWrapper::boomerAMG_t *)this->boomerAMG;
    boomerAMG->solve(h_Gx.ptr<pfloat>(), h_xBuffer.ptr<pfloat>());

    // masked T->E
    ogsScatter(o_x.ptr<pfloat>(), h_xBuffer.ptr<pfloat>(), ogsPfloat, ogsAdd, ogs);
    platform->timer.hostToc(name + " coarseSolve");
  } else {
    platform->timer.tic(name + " coarseSolve", true);
    const bool useDevice = options.compareArgs("MULTIGRID COARSE SOLVER LOCATION", "DEVICE");

    // masked E->T
    vectorDotStarKernel(ogs->N, static_cast<pfloat>(1.0), static_cast<pfloat>(0.0), o_weight, o_rhs, o_Sx);
    ogsGather(o_Gx, o_Sx, ogsPfloat, ogsAdd, ogs);
    if (!useDevice) {
      o_Gx.copyTo(h_Gx.ptr<pfloat>(), N);
    }

    platform->linAlg->fill<pfloat>(N, 0.0, o_xBuffer);
    if (!useDevice) {
      o_xBuffer.copyTo(h_xBuffer, N);
    }
    if (options.compareArgs("MULTIGRID COARSE SOLVER", "BOOMERAMG")) {
      if (useDevice) {
        auto boomerAMG = (hypreWrapperDevice::boomerAMG_t *)this->boomerAMG;
        boomerAMG->solve(o_Gx, o_xBuffer);
      } else {
        auto boomerAMG = (hypreWrapper::boomerAMG_t *)this->boomerAMG;
        boomerAMG->solve(h_Gx.ptr<pfloat>(), h_xBuffer.ptr<pfloat>());
      }
    } else if (options.compareArgs("MULTIGRID COARSE SOLVER", "AMGX")) {
      AMGX->solve(o_Gx.ptr(), o_xBuffer.ptr());
    }

    // masked T->E
    if (useDevice) {
      ogsScatter(o_x, o_xBuffer, ogsPfloat, ogsAdd, ogs);
    } else {
      o_Gx.copyFrom(h_xBuffer, N);
      ogsScatter(o_x, o_Gx, ogsPfloat, ogsAdd, ogs);
    }
    platform->timer.toc(name + " coarseSolve");
  }
}
