#include <registerKernels.hpp>

void registerScalarKernels(occa::properties kernelInfoBC)
{
  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";
  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;
  constexpr int Nfaces{6};

  constexpr int NVfields{3};
  kernelInfo["defines/p_NVfields"] = NVfields;

  std::string fileName, kernelName;
  const std::string suffix = "Hex3D";
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/scalar/");
  const std::string section = "scalar_t::";
  occa::properties meshProps = kernelInfo;
  meshProps += meshKernelProperties(N);

  {
    kernelName = "advectMeshVelocityHex3D";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "neumannBC" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

    kernelName = "dirichletBC";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

    {
      occa::properties prop = kernelInfo;

      int Nsubsteps = 0;
      platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

      int nBDF = 0;
      int nEXT = 0;
      platform->options.getArgs("BDF ORDER", nBDF);
      platform->options.getArgs("EXT ORDER", nEXT);
      if (Nsubsteps) {
        nEXT = nBDF;
      }

      const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

      prop["defines/p_MovingMesh"] = movingMesh;
      prop["defines/p_nEXT"] = nEXT;
      prop["defines/p_nBDF"] = nBDF;

      if (Nsubsteps) {
        prop["defines/p_SUBCYCLING"] = 1;
      } else {
        prop["defines/p_SUBCYCLING"] = 0;
      }

      prop["defines/p_ADVECTION"] = 0;
      if (platform->options.compareArgs("EQUATION TYPE", "NAVIERSTOKES") && !Nsubsteps) {
        prop["defines/p_ADVECTION"] = 1;
      }

      kernelName = "sumMakef";
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + kernelName, fileName, prop);
    }
  }

  registerCvodeKernels();
}
