#include <compileKernels.hpp>
#include "LVector.hpp"

void registerCoreKernels()
{
  oogs::registerKernels();

  if (!platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->linAlg = linAlg_t::getInstance();

    std::string kernelName;
    kernelName = "core-copyDfloatToPfloat";
    platform->copyDfloatToPfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyPfloatToDfloat";
    platform->copyPfloatToDfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyDfloatToDouble";
    platform->copyDfloatToDoubleKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyDfloatToFloat";
    platform->copyDfloatToFloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyDoubleToDfloat";
    platform->copyDoubleToDfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyFloatToDfloat";
    platform->copyFloatToDfloatKernel = platform->kernelRequests.load(kernelName);

    platform->mapVtxToBoxKernel = platform->kernelRequests.load("core-mapVtxToBox");
    platform->mapBoxToVtxKernel = platform->kernelRequests.load("core-mapBoxToVtx");
    platform->boxZeroKernel = platform->kernelRequests.load("core-boxZero");
    platform->boxMultRHSKernel = platform->kernelRequests.load("core-boxMultRHS");
    platform->boxInvMulKernel = platform->kernelRequests.load("core-boxInvMul");
    platform->boxUtoCKernel = platform->kernelRequests.load("core-boxUtoC");
    platform->boxCtoUKernel = platform->kernelRequests.load("core-boxCtoU");

    return;
  }

  std::string kernelName;
  std::string fileName;

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");
  const std::string section = "core-";
  const std::string extension = platform->serial ? ".c" : ".okl";

  kernelName = "nStagesSum3";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  auto meshProps = platform->kernelInfo;
  meshProps += meshKernelProperties(N);
  const std::string suffix = "Hex3D";

  kernelName = "gradientVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "wGradientVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "wDivergenceVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "divergenceVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "curl" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "filterRT" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "vectorFilterRT" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "tensorProduct1D" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "relativeMassHighestMode";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, meshProps);

  kernelName = "relativeMassAveragedMode";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, meshProps);

  kernelName = "computeMaxVisc";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, meshProps);

  kernelName = "interpolateP1";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, meshProps);

  {
    auto prop = meshProps;
    prop["includes"].asArray();
    std::string derivDataFile = oklpath + "/mesh/constantGLLDifferentiationMatrices.h";
    prop["includes"] += derivDataFile.c_str();
    prop["defines/p_inputAdd"] = 0;
    kernelName = "weakLaplacian" + suffix;
    fileName = oklpath + "/core/" + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, prop);
  }

  // register platform kernels
  {

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + "/core/" + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/pfloat"] = "double";
      prop["defines/dummy"] = 1; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDfloatToDouble", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + "/core/" + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/pfloat"] = "float";
      prop["defines/dummy"] = 2; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDfloatToFloat", fileName, prop);
    }
 
    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + "/core/" + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/dfloat"] = "double";
      prop["defines/pfloat"] = dfloatString;
      prop["defines/dummy"] = 3; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDoubleToDfloat", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + "/core/" + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/dfloat"] = "float";
      prop["defines/pfloat"] = dfloatString;
      prop["defines/dummy"] = 4; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyFloatToDfloat", fileName, prop);
    }

    auto prop = platform->kernelInfo;
    kernelName = "copyDfloatToPfloat";
    fileName = oklpath + "/core/" + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);

    kernelName = "copyPfloatToDfloat";
    fileName = oklpath + "/core/" + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);

    prop["defines/p_NC"] = 8;

    fileName = oklpath + "/core/mapVtxToBox" + extension;
    platform->kernelRequests.add(section + "mapVtxToBox", fileName, prop);

    fileName = oklpath + "/core/mapBoxToVtx" + extension;
    platform->kernelRequests.add(section + "mapBoxToVtx", fileName, prop);

    fileName = oklpath + "/core/boxZero" + extension;
    platform->kernelRequests.add(section + "boxZero", fileName, prop);

    fileName = oklpath + "/core/boxMultRHS" + extension;
    platform->kernelRequests.add(section + "boxMultRHS", fileName, prop);

    fileName = oklpath + "/core/boxInvMul" + extension;
    platform->kernelRequests.add(section + "boxInvMul", fileName, prop);

    fileName = oklpath + "/core/boxUtoC" + extension;
    platform->kernelRequests.add(section + "boxUtoC", fileName, prop);

    fileName = oklpath + "/core/boxCtoU" + extension;
    platform->kernelRequests.add(section + "boxCtoU", fileName, prop);
  }

  registerLinAlgKernels();
  LVector_t<dfloat>::registerKernels();
  LVector_t<pfloat>::registerKernels();
}
