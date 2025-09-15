#include "registerKernels.hpp"

void registerPostProcessingKernels()
{
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;

  auto kernelInfo = platform->kernelInfo + meshKernelProperties(N);
  const std::string section = "nrs-";

  kernelInfo["includes"].asArray();

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");
  std::string kernelName, fileName;

  kernelName = "aeroForces";
  fileName = oklpath + "/app/nrs/postProcessing/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfo);

  kernelName = "Qcriterion";
  fileName = oklpath + "/app/nrs/postProcessing/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfo);

  kernelName = "viscousShearStress";
  fileName = oklpath + "/app/nrs/postProcessing/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfo);
}
