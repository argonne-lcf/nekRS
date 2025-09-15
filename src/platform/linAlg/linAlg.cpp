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

#include "linAlg.hpp"
#include "platform.hpp"
#include "re2Reader.hpp"
#include <numeric>
#include <variant>

linAlg_t *linAlg_t::singleton = nullptr;

void linAlg_t::runTimers()
{
  int nelgt, nelgv;
  const std::string meshFile = platform->options.getArgs("MESH FILE");
  re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm());
  const int nel = nelgv / platform->comm.mpiCommSize();

  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  const auto Nrep = 20;

  auto run = [&](int fields) {
    const auto Nlocal = nel * (N + 1) * (N + 1) * (N + 1);
    const auto offset = Nlocal;
    auto o_weight = platform->device.malloc<dfloat>(Nlocal);
    auto o_r = platform->device.malloc<dfloat>(fields * Nlocal);
    auto o_z = platform->device.malloc<dfloat>(fields * Nlocal);

    // warm-up
    weightedInnerProdMany(Nlocal, fields, offset, o_weight, o_r, o_z, platform->comm.mpiComm());

    std::vector<double> elapsed;
    for (int i = 0; i < Nrep; i++) {
      MPI_Barrier(platform->comm.mpiComm());
      const auto tStart = MPI_Wtime();

      weightedInnerProdMany(Nlocal, fields, offset, o_weight, o_r, o_z, platform->comm.mpiComm());

      elapsed.push_back((MPI_Wtime() - tStart));
    }

    double elapsedMax = *std::max_element(elapsed.begin(), elapsed.end());
    double elapsedMin = *std::min_element(elapsed.begin(), elapsed.end());
    double elapsedAvg = std::accumulate(elapsed.begin(), elapsed.end(), 0.0);

    MPI_Allreduce(MPI_IN_PLACE, &elapsedMax, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm());
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMin, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm());
    MPI_Allreduce(MPI_IN_PLACE, &elapsedAvg, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("wdotp nFields=%02d min/avg/max: %.3es %.3es %.3es  ",
             fields,
             elapsedMin,
             elapsedAvg / Nrep,
             elapsedMax);
    }

    if (platform->comm.mpiCommSize() > 1) {
      platform->device.finish();
      MPI_Barrier(platform->comm.mpiComm());
      const auto tStart = MPI_Wtime();
      for (int i = 0; i < Nrep; i++) {
        weightedInnerProdMany(Nlocal, fields, offset, o_weight, o_r, o_z, MPI_COMM_SELF);
      }
      platform->device.finish();
      const auto elapsed = (MPI_Wtime() - tStart) / Nrep;
      auto elapsedMax = 0.0;
      MPI_Allreduce(&elapsed, &elapsedMax, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm());
      if (platform->comm.mpiRank() == 0) {
        printf("(avg local: %.3es / %.3eGB/s)\n",
               elapsedMax,
               (1 + 2 * fields) * Nlocal * sizeof(dfloat) / elapsedMax / 1e9);
      }
    } else {
      if (platform->comm.mpiRank() == 0) {
        printf("\n");
      }
    }
  };

  for (int i : {1, 3}) {
    run(i);
  }

  if (platform->comm.mpiRank() == 0) {
    std::cout << std::endl;
  }
}

linAlg_t *linAlg_t::getInstance()
{
  if (!singleton) {
    singleton = new linAlg_t();
  }
  return singleton;
}

linAlg_t::linAlg_t()
{
  blocksize = BLOCKSIZE;
  serial = platform->serial();
  comm = platform->comm.mpiComm();
  timer = 0;

  const auto tStart = MPI_Wtime();
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "initializing linAlg ...\n";
  }

  runTimers();

  if (platform->options.compareArgs("ENABLE LINALG TIMER", "TRUE")) {
    timer = 1;
  }

  MPI_Barrier(platform->comm.mpiComm());
  if (platform->comm.mpiRank() == 0) {
    printf("done (%g)\n", MPI_Wtime() - tStart);
  }
}

void linAlg_t::enableTimer()
{
  timer = 1;
}

void linAlg_t::disableTimer()
{
  timer = 0;
}

linAlg_t::~linAlg_t() {}

void linAlg_t::crossProduct(const dlong N,
                            const dlong fieldOffset,
                            const occa::memory &o_x,
                            const occa::memory &o_y,
                            occa::memory &o_z)
{
  nekrsCheck(o_x.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");
  nekrsCheck(o_y.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");
  nekrsCheck(o_z.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "crossProduct", N, fieldOffset, o_x, o_y, o_z);
}

void linAlg_t::dotProduct(const dlong N,
                          const dlong fieldOffset,
                          const occa::memory &o_x,
                          const occa::memory &o_y,
                          occa::memory &o_z)
{
  nekrsCheck(o_x.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");
  nekrsCheck(o_y.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");
  nekrsCheck(o_z.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "dotProduct", N, fieldOffset, o_x, o_y, o_z);
}

void linAlg_t::dotProduct(const dlong N,
                           const dlong fieldOffset,
                           const occa::memory &o_x,
                           const std::array<dfloat, 3> y,
                           occa::memory &o_z)
{
  nekrsCheck(o_x.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");
  nekrsCheck(o_z.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_x too small to store a vector field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "dotConstProduct", N, fieldOffset, o_x, y[0], y[1], y[2], o_z);
}

void linAlg_t::unitVector(const dlong N, const dlong fieldOffset, occa::memory &o_v)
{
  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "unitVector", N, fieldOffset, o_v);
}

void linAlg_t::entrywiseMag(const dlong N,
                            const dlong Nfields,
                            const dlong fieldOffset,
                            const occa::memory &o_a,
                            occa::memory &o_b)
{
  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "entrywiseMag", N, Nfields, fieldOffset, o_a, o_b);
}

void linAlg_t::magSqrVector(const dlong N,
                            const dlong fieldOffset,
                            const occa::memory &o_u,
                            occa::memory &o_mag)
{
  nekrsCheck(o_u.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_u too small to store a vector field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "magSqrVector", N, fieldOffset, o_u, o_mag);
}

void linAlg_t::magVector(const dlong N, const dlong fieldOffset, const occa::memory &o_u, occa::memory &o_mag)
{
  nekrsCheck(o_u.length() < (3 * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_u too small to store a vector field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "magVector", N, fieldOffset, o_u, o_mag);
}

void linAlg_t::magSqrTensor(const dlong N,
                            const dlong fieldOffset,
                            const occa::memory &o_tensor,
                            occa::memory &o_mag)
{
  nekrsCheck(o_tensor.length() < 9 * fieldOffset,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_tensor too small to store a symmetric tensor field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "magSqrTensor", N, fieldOffset, o_tensor, o_mag);
}

void linAlg_t::magSqrSymTensor(const dlong N,
                               const dlong fieldOffset,
                               const occa::memory &o_tensor,
                               occa::memory &o_mag)
{
  nekrsCheck(o_tensor.length() < 6 * fieldOffset,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_tensor too small to store a symmetric tensor field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "magSqrSymTensor", N, fieldOffset, o_tensor, o_mag);
}

void linAlg_t::magSqrSymTensorDiag(const dlong N,
                                   const dlong fieldOffset,
                                   const occa::memory &o_tensor,
                                   occa::memory &o_mag)
{
  nekrsCheck(o_tensor.length() < 6 * fieldOffset,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "o_tensor too small to store a symmetric tensor field!\n");

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "magSqrSymTensorDiag", N, fieldOffset, o_tensor, o_mag);
}

void linAlg_t::linearCombination(const dlong N,
                                 const dlong Nfields,
                                 const dlong fieldOffset,
                                 const occa::memory &o_coeff,
                                 const occa::memory &o_x,
                                 occa::memory &o_y)
{
  nekrsCheck(o_coeff.length() < Nfields,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "o_c too small for %d fields!\n",
             Nfields);
  nekrsCheck(o_x.length() < (Nfields * fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "o_x too small to store %d fields!\n",
             Nfields);

  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "linearCombination",
                     N,
                     Nfields,
                     fieldOffset,
                     o_coeff,
                     o_x,
                     o_y);
}

dfloat linAlg_t::maxRelativeError(const dlong N,
                                  const int Nfields,
                                  const dlong fieldOffset,
                                  const dfloat absTol,
                                  const occa::memory &o_u,
                                  const occa::memory &o_uRef,
                                  MPI_Comm comm)
{
  auto o_err = platform->deviceMemoryPool.reserve<dfloat>(std::max(Nfields * fieldOffset, N));
  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "relativeError",
                     N,
                     Nfields,
                     fieldOffset,
                     absTol,
                     o_u,
                     o_uRef,
                     o_err);
  return this->amaxMany(N, Nfields, fieldOffset, o_err, comm);
}

dfloat linAlg_t::maxAbsoluteError(const dlong N,
                                  const int Nfields,
                                  const dlong fieldOffset,
                                  const dfloat absTol,
                                  const occa::memory &o_u,
                                  const occa::memory &o_uRef,
                                  MPI_Comm comm)
{
  auto o_err = platform->deviceMemoryPool.reserve<dfloat>(std::max(Nfields * fieldOffset, N));
  linAlgLaunchKernel(getKnlPrefix<dfloat>() + "absoluteError",
                     N,
                     Nfields,
                     fieldOffset,
                     absTol,
                     o_u,
                     o_uRef,
                     o_err);
  return this->amaxMany(N, Nfields, fieldOffset, o_err, comm);
}
