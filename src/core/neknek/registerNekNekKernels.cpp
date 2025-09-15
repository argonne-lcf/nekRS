#include <registerKernels.hpp>
#include "solver.hpp"
#include "bdryBase.hpp"

void registerNekNekKernels()
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerNekNekKernels" << std::endl;
  }

  dlong N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

  const std::string prefix = "neknek::";

  std::string kernelName = "copyNekNekPoints";
  std::string fileName = oklpath + "/core/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "pack";
  fileName = oklpath + "/core/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  auto surfaceFluxKernelInfo = platform->kernelInfo;
  surfaceFluxKernelInfo += meshKernelProperties(N);
  platform->app->bc->addKernelConstants(surfaceFluxKernelInfo);
  kernelName = "computeFlux";
  fileName = oklpath + "/core/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, surfaceFluxKernelInfo);

  kernelName = "fixSurfaceFlux";
  fileName = oklpath + "/core/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, surfaceFluxKernelInfo);

  auto extrapolateBoundaryInfo = platform->kernelInfo;
  extrapolateBoundaryInfo["includes"].asArray();
  extrapolateBoundaryInfo["includes"] += oklpath + "/core/neknek/timeInterpWeights.okl.hpp";
  extrapolateBoundaryInfo["defines/p_NVfields"] = 3;
  kernelName = "extrapolateBoundary";
  fileName = oklpath + "/core/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, extrapolateBoundaryInfo);
}
