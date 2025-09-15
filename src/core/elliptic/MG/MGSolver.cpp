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

#include "MGSolver.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

MGSolver_t::MGSolver_t(const std::string &name_, occa::device device_, MPI_Comm comm_, setupAide options_)
{
  name = name_;
  device = device_;
  comm = comm_;
  options = options_;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  nekrsCheck(!options.compareArgs("MGSOLVER CYCLE", "VCYCLE"),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "Unknown multigrid cycle type!");

  levels = (MGSolver_t::multigridLevel **)calloc(MAX_LEVELS, sizeof(MGSolver_t::multigridLevel *));

  coarseLevel = new coarseLevel_t(name, options, comm);

  numLevels = 0;

  ctype = VCYCLE;
  additive = (options.compareArgs("MGSOLVER CYCLE", "ADDITIVE")) ? true : false;

  overlapCrsGridSolve = false;
  if (options.compareArgs("MGSOLVER CYCLE", "OVERLAPCRS")) {
    if (size > 1) {
      const auto env = std::getenv("NEKRS_MPI_THREAD_MULTIPLE");
      nekrsCheck(!(env && std::string(env) == "1"),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s\n",
                 "overlapping coarse solve requires NEKRS_MPI_THREAD_MULTIPLE=1!");

      int provided;
      MPI_Query_thread(&provided);
      if (provided != MPI_THREAD_MULTIPLE) {
        nekrsAbort(platform->comm.mpiComm(),
                   EXIT_FAILURE,
                   "%s\n",
                   "overlapping coarse solve requires MPI_THREAD_MULTIPLE support!");
      }
    }
    overlapCrsGridSolve = true;
    coarseLevel->solveOnHost = true;

    if (rank == 0) {
      printf("overlapping coarse grid solve enabled\n");
    }
  }
}

MGSolver_t::~MGSolver_t()
{
  for (int n = 0; n < numLevels; n++) {
    delete levels[n];
  }

  free(levels);

  if (coarseLevel) {
    delete coarseLevel;
  }
}

void MGSolver_t::Run(occa::memory o_rhsFine, occa::memory o_xFine)
{
  levels[0]->o_x = o_xFine;
  levels[0]->o_rhs = o_rhsFine;

  runVcycle();

  levels[0]->o_x = nullptr;
  levels[0]->o_rhs = nullptr;
}

void MGSolver_t::Report() {}

void MGSolver_t::runVcycle()
{
  // precompute coarse rhs for all levels
  if (additive || this->overlapCrsGridSolve) {
    for (int k = 0; k < numLevels - 1; ++k) {
      auto &level = levels[k];
      auto &o_rhs = level->o_rhs;
      auto &o_wrk = level->o_res;
      auto &levelC = levels[k + 1];
      auto &o_rhsC = levelC->o_rhs;

      o_wrk.copyFrom(o_rhs, level->Nrows);
      levelC->coarsen(o_wrk, o_rhsC);
    }
  }

  occa::memory o_xCoarse, o_rhsCoarse;
  if (this->overlapCrsGridSolve) {
    o_rhsCoarse = platform->memoryPool.reserve<pfloat>(levels[baseLevel]->o_rhs.size());
    o_rhsCoarse.copyFrom(levels[baseLevel]->o_rhs);
    o_xCoarse = platform->memoryPool.reserve<pfloat>(levels[baseLevel]->o_x.size());
    auto xCoarsePtr = o_xCoarse.ptr<pfloat>();
    for (int i = 0; i < o_xCoarse.size(); ++i) {
      xCoarsePtr[i] = 0;
    }
  } else {
    o_rhsCoarse = levels[baseLevel]->o_rhs;
    o_xCoarse = levels[baseLevel]->o_x;
    platform->linAlg->fill<pfloat>(o_xCoarse.size(), 0.0, o_xCoarse);
  }

  o_rhsCoarse.getDevice().finish();

  // overlap the downward V-cycle phase with the coarse-level solve on the host
  // parallel execution of downward additive V-cycle (on device) is not utilized
  const auto nThreads = this->overlapCrsGridSolve ? 2 : 1;
#pragma omp parallel proc_bind(close) num_threads(nThreads)
  {
#pragma omp single
    {
#pragma omp task
      {
        for (int k = 0; k < numLevels - 1; ++k) {
          auto &level = levels[k];
          auto &o_rhs = levels[k]->o_rhs;
          auto &o_x = levels[k]->o_x;

          auto &levelC = levels[k + 1];
          auto &o_rhsC = levelC->o_rhs;
          auto &o_xC = levelC->o_x;

          level->smooth(o_rhs, o_x, true);
          if (!additive) {
            level->residual(o_rhs, o_x, level->o_res);
            levelC->coarsen(level->o_res, o_rhsC);
          }
        }
      }

#pragma omp task
      {
        coarseLevel->solvePtr(o_rhsCoarse, o_xCoarse);
      }
    }
  }

  if (this->overlapCrsGridSolve) {
    levels[baseLevel]->o_x.copyFrom(o_xCoarse);
  }

  // upward V-cycle
  for (int k = numLevels - 2; k >= 0; --k) {
    auto &level = levels[k];
    auto &o_rhs = levels[k]->o_rhs;
    auto &o_x = levels[k]->o_x;

    auto &levelC = levels[k + 1];
    auto &o_rhsC = levelC->o_rhs;
    auto &o_xC = levelC->o_x;

    levelC->prolongate(o_xC, o_x);
    if (!additive) {
      level->smooth(o_rhs, o_x, false);
    }
  }
}

void MGSolver_t::allocateWorkStorage()
{
  for (int k = 0; k < numLevels; k++) {
    levels[k]->o_res = platform->deviceMemoryPool.reserve<pfloat>(levels[k]->Ncols);
    // allocate coarse levels only
    if (k) {
      levels[k]->o_x = platform->deviceMemoryPool.reserve<pfloat>(levels[k]->Ncols);
      levels[k]->o_rhs = platform->deviceMemoryPool.reserve<pfloat>(levels[k]->Nrows);
    }
  }
}

void MGSolver_t::freeWorkStorage()
{
  for (int k = 0; k < numLevels; k++) {
    levels[k]->o_res.free();
    if (k) {
      levels[k]->o_x.free();
      levels[k]->o_rhs.free();
    }
  }
}
