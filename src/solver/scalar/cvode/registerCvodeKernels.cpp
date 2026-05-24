#include "registerKernels.hpp"

void registerCvodeKernels(occa::properties kernelInfoBC)
{
  int Nscalars = 0;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalars);

  int NscalarsCvode = 0;

  if (Nscalars) {
    for (int is = 0; is < Nscalars; is++) {
      const auto sid = scalarDigitStr(is);

      if (platform->options.compareArgs("SCALAR" + sid + " SOLVER", "CVODE")) {
        NscalarsCvode++;
      }
    }
  }

  if (!NscalarsCvode) {
    return;
  }

  std::string kernelName;
  std::string fileName;

  const std::string suffix = "Hex3D";
  const std::string prefix = "cvode_t::";
  const std::string extension = platform->serial() ? ".c" : ".okl";

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

  kernelName = "extrapolate";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  {
    auto prop = kernelInfoBC;
    prop["defines/p_lhs"] = 0;
    kernelName = "neumannBC" + suffix;
    fileName = oklpath + "/solver/scalar/" + kernelName + ".okl";
    platform->kernelRequests.add(prefix + kernelName, fileName, prop);
  }

  kernelName = "errorWeight";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "mapToMaskedPoint";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "extrapolateDirichlet";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "axpby";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "axmyz";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "linearCombination";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  kernelName = "innerProdMulti";
  fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";
  platform->kernelRequests.add(prefix + kernelName, fileName, platform->kernelInfo);

  {
    auto prop = platform->kernelInfo;
    kernelName = "fusedAddRhoDiv";
    fileName = oklpath + "/solver/scalar/cvode/" + kernelName + ".okl";

    prop["defines/p_addPointSource"] = 0;
    platform->kernelRequests.add(prefix + "rhoDiv", fileName, prop);

    prop["defines/p_addPointSource"] = 1;
    platform->kernelRequests.add(prefix + kernelName, fileName, prop);
  }

  {
    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);

    auto prop = platform->kernelInfo;
    prop["includes"].asArray();
    prop += meshKernelProperties(N);

    std::string derivDataFile = oklpath + "/core/mesh/constantGLLDifferentiationMatrices.h";

    prop["includes"] += derivDataFile.c_str();
    prop["defines/p_weightInputAdd"] = 1;
    kernelName = "weakLaplacianHex3D";
    fileName = oklpath + "/core/" + kernelName + ".okl";

    platform->kernelRequests.add(prefix + kernelName, fileName, prop);
  }

  {
    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);

    int cubN;
    platform->options.getArgs("OVERINTEGRATION POLYNOMIAL DEGREE ", cubN);
    const auto cubNq = cubN + 1;

    auto prop = platform->kernelInfo;
    prop["includes"].asArray();
    prop += meshKernelProperties(N);

    std::string diffDataFile = oklpath + "/core/mesh/constantDifferentiationMatrices.h";
    std::string interpDataFile = oklpath + "/core/mesh/constantInterpolationMatrices.h";
    std::string diffInterpDataFile = oklpath + "/core/mesh/constantDifferentiationInterpolationMatrices.h";

    prop["includes"] += diffDataFile.c_str();
    prop["includes"] += interpDataFile.c_str();
    prop["includes"] += diffInterpDataFile.c_str();
    prop["defines/p_add"] = 1;

    if (platform->options.compareArgs("ADVECTION TYPE", "OVERINTEGRATION")) {
      prop["defines/p_cubNq"] = cubN + 1;
      prop["defines/p_cubNp"] = std::pow(cubNq, 3);

      kernelName = "strongAdvectionCubatureVolumeScalar" + suffix;
      fileName = oklpath + "/core/" + kernelName + ".okl";
      platform->kernelRequests.add(prefix + kernelName, fileName, prop);
    } else {
      kernelName = "strongAdvectionVolumeScalar" + suffix;
      fileName = oklpath + "/core/" + kernelName + ".okl";
      platform->kernelRequests.add(prefix + kernelName, fileName, prop);
    }
  }
}
