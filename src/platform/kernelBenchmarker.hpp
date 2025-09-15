#include "platform.hpp"
#include "mesh.h"
#include <utility>
#include <functional>

std::pair<occa::kernel, double>
benchmarkKernel(std::function<occa::kernel(int kernelVariant)> kernelBuilder,
                std::function<void(occa::kernel &)> kernelRunner,
                std::function<void(int kernelVariant, double tKernel, int Ntests)> printCallback,
                const std::vector<int> &kernelVariants,
                int Ntests);

std::pair<occa::kernel, double>
benchmarkKernel(std::function<occa::kernel(int kernelVariant)> kernelBuilder,
                std::function<void(occa::kernel &)> kernelRunner,
                std::function<void(int kernelVariant, double tKernel, int Ntests)> printCallback,
                const std::vector<int> &kernelVariants,
                double targetTime);

template <typename T>
T maxRelErr(const std::vector<T>& uRef, const std::vector<T>& u, MPI_Comm comm, T absTol = 0)
{
  double err = 0;

  if (absTol > 0) {
    for (int i = 0; i < uRef.size(); ++i) {
      if (std::abs(uRef[i]) > absTol) {
        err = std::max(err, (double) std::abs((uRef[i] - u[i])/uRef[i]));
      }
    }
  } else {
    for (int i = 0; i < uRef.size(); ++i) {
      err = std::max(err, (double) std::abs((uRef[i] - u[i])/uRef[i]));
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, comm);
  return static_cast<T>(err);
}

template <typename T>
T maxAbsErr(const std::vector<T>& uRef, const std::vector<T>& u, MPI_Comm comm, T absTol = 0)
{
  double err = 0;

  if (absTol > 0) {
    for (int i = 0; i < uRef.size(); ++i) {
      if (std::abs(uRef[i]) > absTol) {
        err = std::max(err, (double) std::abs(uRef[i] - u[i]));
      }
    }
  } else {
    for (int i = 0; i < uRef.size(); ++i) {
      err = std::max(err, (double) std::abs(uRef[i] - u[i]));
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, comm);
  return static_cast<T>(err);
}

template <typename T>
T range(const std::vector<T>& u, T absTol)
{
  T minAbsValue = std::numeric_limits<T>::max();
  for (const auto& val : u) {
      if (std::abs(val) < absTol) continue;
      if (std::abs(val) < minAbsValue) minAbsValue = std::abs(val);
  }
  const auto maxValue = *std::max_element(u.begin(), u.end());
  return std::abs(maxValue) / minAbsValue;
}
