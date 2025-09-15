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

#ifndef LINALG_HPP
#define LINALG_HPP

#include "platform.hpp"


#define linAlgLaunchKernel(name, ...)                                                                        \
do {                                                                                                         \
  static occa::kernel kernel;                                                                                \
  if (!kernel.isInitialized()) kernel = platform->kernelRequests.load(name);                                 \
  kernel(__VA_ARGS__);                                                                                       \
} while (0)

#define USE_WEIGHTED_INNER_PROD_MULTI_DEVICE 0

class linAlg_t
{
private:
  occa::properties kernelInfo;
  MPI_Comm comm;
  int blocksize;
  bool serial;

  int timer = 0;

  void runTimers();

  ~linAlg_t();
  linAlg_t();
  static linAlg_t *singleton;

  template <typename T = dfloat> static std::string getKnlPrefix()
  {
    const auto supportedDataType = std::is_same<T, float>::value || std::is_same<T, double>::value;
    nekrsCheck(!supportedDataType, MPI_COMM_SELF, EXIT_FAILURE, "%s", "unsupported data type on input!\n");  

    return std::string("linAlg::") + ((std::is_same<T, float>::value) ? std::string("f_") : std::string("d_"));
  }

  template <typename T = dfloat> occa::memory getScratch(const size_t n, bool host = false)
  {
    if (host) {
      static occa::memory o_mem;
      if (o_mem.size() < n) {
        if (o_mem.isInitialized()) o_mem.free();
        o_mem = platform->device.mallocHost<T>(n);
      } 
      return o_mem.slice(0, n);
    } else {
      static occa::memory o_mem;
      if (o_mem.size() < n) {
        if (o_mem.isInitialized()) o_mem.free();
        o_mem = platform->device.malloc<T>(n);
      }
      return o_mem.slice(0, n);
    }
  }

public:
  static linAlg_t *getInstance();

  void enableTimer();
  void disableTimer();

#include "linAlg.tpp"
  void dotProduct(const dlong N,
                  const dlong fieldOffset,
                  const occa::memory &o_x,
                  const std::array<dfloat, 3> y,
                  occa::memory &o_z);

  void dotProduct(const dlong N,
                  const dlong fieldOffset,
                  const occa::memory &o_x,
                  const occa::memory &o_y,
                  occa::memory &o_z);

  // z = x \cross y
  void crossProduct(const dlong N,
                    const dlong fieldOffset,
                    const occa::memory &o_x,
                    const occa::memory &o_y,
                    occa::memory &o_z);

  void unitVector(const dlong N, const dlong fieldOffset, occa::memory &o_v);

  // o_b[n] = \sqrt{\sum_{i=0}^{Nfields-1} o_a[n+i*fieldOffset]^2}
  void entrywiseMag(const dlong N,
                    const dlong Nfields,
                    const dlong fieldOffset,
                    const occa::memory &o_a,
                    occa::memory &o_b);

  void magSqrVector(const dlong N, const dlong fieldOffset, const occa::memory &o_u, occa::memory &o_mag);

  void magVector(const dlong N, const dlong fieldOffset, const occa::memory &o_u, occa::memory &o_mag);

  void
  magSqrSymTensor(const dlong N, const dlong fieldOffset, const occa::memory &o_tensor, occa::memory &o_mag);

  void magSqrSymTensorDiag(const dlong N,
                           const dlong fieldOffset,
                           const occa::memory &o_tensor,
                           occa::memory &o_mag);

  void
  magSqrTensor(const dlong N, const dlong fieldOffset, const occa::memory &o_tensor, occa::memory &o_mag);

  // o_y[n] = x_{Nfields} * coeff_{Nfields} + \sum_{i=0}^{Nfields-1} coeff_i * x_i
  void linearCombination(const dlong N,
                         const dlong Nfields,
                         const dlong fieldOffset,
                         const occa::memory &o_coeff,
                         const occa::memory &o_x,
                         occa::memory &o_y);

  dfloat maxRelativeError(const dlong N,
                          const int Nfields,
                          const dlong fieldOffset,
                          const dfloat absTol,
                          const occa::memory &o_u,
                          const occa::memory &o_uRef,
                          MPI_Comm comm);

  dfloat maxAbsoluteError(const dlong N,
                          const int Nfields,
                          const dlong fieldOffset,
                          const dfloat absTol,
                          const occa::memory &o_u,
                          const occa::memory &o_uRef,
                          MPI_Comm comm);

  // matrix is in row major ordering
  std::vector<dfloat> matrixInverse(const int N, const std::vector<dfloat> &A);
  std::vector<dfloat> matrixPseudoInverse(const int N, const std::vector<dfloat> &A);
  std::vector<dfloat> matrixTranspose(const int N, const std::vector<dfloat> &A);
};

#endif
