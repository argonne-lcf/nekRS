#include "registerKernels.hpp"
#include "LVector.hpp"
#include "benchmarkAdvsub.hpp"

void registerCoreKernels(occa::properties kernelInfoBC)
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerCoreKernels" << std::endl;
  }

  oogs::registerKernels();
  registerLinAlgKernels();
  registerLinearSolverKernels();
  registerMeshKernels(kernelInfoBC);
  registerPointInterpolationKernels();
  registerNekNekKernels();

  LVector_t<dfloat>::registerKernels();
  LVector_t<pfloat>::registerKernels();

  const std::string section = "core-";
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/");
  const std::string extension = platform->serial() ? ".c" : ".okl";
  const std::string suffix = "Hex3D";

  std::string kernelName;
  std::string fileName;

  // platform copy kernels
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/pfloat"] = "double";
      prop["defines/dummy"] = 1; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDfloatToDouble", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/pfloat"] = "float";
      prop["defines/dummy"] = 2; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDfloatToFloat", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/dfloat"] = "double";
      prop["defines/pfloat"] = dfloatString;
      prop["defines/dummy"] = 3; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDoubleToDfloat", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/dfloat"] = "float";
      prop["defines/pfloat"] = dfloatString;
      prop["defines/dummy"] = 4; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyFloatToDfloat", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/dfloat"] = "float";
      prop["defines/pfloat"] = "double";
      prop["defines/dummy"] = 5; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyFloatToDouble", fileName, prop);
    }

    {
      kernelName = "copyDfloatToPfloat";
      fileName = oklpath + kernelName + extension;
      auto prop = platform->kernelInfo;
      prop["defines/dfloat"] = "double";
      prop["defines/pfloat"] = "float";
      prop["defines/dummy"] = 6; // just to make it different from copyDfloatToDouble to avoid collison
      platform->kernelRequests.add(section + "copyDoubleToFloat", fileName, prop);
    }

    auto prop = platform->kernelInfo;
    kernelName = "copyDfloatToPfloat";
    fileName = oklpath + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);

    kernelName = "copyPfloatToDfloat";
    fileName = oklpath + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);
  } else {
    std::string kernelName;
    kernelName = section + "copyDfloatToPfloat";
    platform->copyDfloatToPfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyPfloatToDfloat";
    platform->copyPfloatToDfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyDfloatToDouble";
    platform->copyDfloatToDoubleKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyDfloatToFloat";
    platform->copyDfloatToFloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyDoubleToDfloat";
    platform->copyDoubleToDfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyFloatToDfloat";
    platform->copyFloatToDfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyFloatToDouble";
    platform->copyFloatToDoubleKernel = platform->kernelRequests.load(kernelName);

    kernelName = section + "copyDoubleToFloat";
    platform->copyDoubleToFloatKernel = platform->kernelRequests.load(kernelName);
  }

  const auto meshProps = [&]() {
    auto props = platform->kernelInfo;
    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);
    props += meshKernelProperties(N);
    return props;
  }();

  // register advection kernels
  {
    const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);
    const int Nq = N + 1;

    int Nsubsteps = 0;
    platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

    int nBDF = 0;
    int nEXT = 0;
    platform->options.getArgs("BDF ORDER", nBDF);
    platform->options.getArgs("EXT ORDER", nEXT);
    if (Nsubsteps) {
      nEXT = nBDF;
    }

    int cubN;
    platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
    const int cubNq = cubN + 1;
    const int cubNp = cubNq * cubNq * cubNq;

    auto prop = meshProps;

    std::string diffDataFile = oklpath + "mesh/constantDifferentiationMatrices.h";
    std::string interpDataFile = oklpath + "mesh/constantInterpolationMatrices.h";
    std::string diffInterpDataFile = oklpath + "mesh/constantDifferentiationInterpolationMatrices.h";

    prop["includes"] += diffDataFile.c_str();
    prop["includes"] += interpDataFile.c_str();
    prop["includes"] += diffInterpDataFile.c_str();

    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
        prop["defines/p_cubNq"] = cubNq;
        prop["defines/p_cubNp"] = cubNp;

        kernelName = "strongAdvectionCubatureVolume" + suffix;
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, prop);

        kernelName = "strongAdvectionCubatureVolumeScalar" + suffix;
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, prop);
      } else {
        kernelName = "strongAdvectionVolume" + suffix;
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, prop);

        kernelName = "strongAdvectionVolumeScalar" + suffix;
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, prop);
      }
    }

    if (Nsubsteps) {
      constexpr int nVFields{3};

      if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
        kernelName = "subCycleRK";
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

        {
          auto p = platform->kernelInfo;
          p["defines/p_MovingMesh"] = movingMesh;

          kernelName = "subCycleInitU0";
          fileName = oklpath + kernelName + ".okl";
          platform->kernelRequests.add(section + kernelName, fileName, p);
        }

        prop["defines/p_MovingMesh"] = movingMesh;
        prop["defines/p_nEXT"] = nEXT;
        prop["defines/p_cubNq"] = cubNq;
        prop["defines/p_cubNp"] = cubNp;
        prop["defines/p_NVfields"] = nVFields;

        kernelName = "subCycleStrongVolume" + suffix;
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, prop);

        kernelName = "subCycleStrongVolumeScalar" + suffix;
        fileName = oklpath + kernelName + ".okl";
        platform->kernelRequests.add(section + kernelName, fileName, prop);
      }

      {
        int nelgt, nelgv;
        const std::string meshFile = platform->options.getArgs("MESH FILE");
        re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm());

        bool verbose = platform->verbose();
        const int verbosity = verbose ? 2 : 1;

        const auto dealiasing = true;

        auto subCycleKernel =
            benchmarkAdvsub(nVFields,
                            nelgv / platform->comm.mpiCommSize(),
                            Nq,
                            cubNq,
                            nEXT,
                            dealiasing,
                            false,
                            verbosity,
                            targetTimeBenchmark,
                            platform->options.compareArgs("KERNEL AUTOTUNING", "FALSE") ? false : true);

        kernelName = "subCycleStrongCubatureVolume" + suffix;
        platform->kernelRequests.add(section + kernelName, subCycleKernel);

        auto subCycleScalarKernel =
            benchmarkAdvsub(1,
                            nelgv / platform->comm.mpiCommSize(),
                            Nq,
                            cubNq,
                            nEXT,
                            dealiasing,
                            true,
                            verbosity,
                            targetTimeBenchmark,
                            platform->options.compareArgs("KERNEL AUTOTUNING", "FALSE") ? false : true);

        kernelName = "subCycleStrongCubatureVolumeScalar" + suffix;
        platform->kernelRequests.add(section + kernelName, subCycleScalarKernel);
      }
    }
  }

  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    kernelName = "maskCopy";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

    kernelName = "maskCopy2";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

    kernelName = "nStagesSum3";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

    kernelName = "nStagesSum3Vector";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

    kernelName = "nStagesSumMany";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

    kernelName = "extrapolate";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "gradientVolume" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "wGradientVolume" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "wDivergenceVolume" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "divergenceVolume" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "curl" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    {
      auto p = meshProps;
      p["includes"].asArray();
      p["includes"] += oklpath + "mesh/constantGLLDifferentiationMatrices.h";
      p["defines/p_inputAdd"] = 0;

      kernelName = "weakLaplacian" + suffix;
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + kernelName, fileName, p);
    }

    // register stabilization kernels
    {
      kernelName = "filterRT" + suffix;
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + kernelName, fileName, meshProps);

      kernelName = "vectorFilterRT" + suffix;
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + kernelName, fileName, meshProps);

      kernelName = "tensorProduct1D" + suffix;
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + kernelName, fileName, meshProps);

      kernelName = "relativeMassAveragedMode";
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + "avm::" + kernelName, fileName, meshProps);

      kernelName = "computeAvmMaxVisc";
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + "avm::" + kernelName, fileName, meshProps);

      kernelName = "interpolateP1";
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(section + "avm::" + kernelName, fileName, meshProps);
    }

    // register gjp kernels
    {
      kernelName = "gjp" + suffix;
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(kernelName, fileName, meshProps);

      int N;
      platform->options.getArgs("POLYNOMIAL DEGREE", N);
      const int Nq = N + 1;

      auto props = meshProps;
      props["defines/p_invNqNq"] = 1. / (Nq * Nq);
      kernelName = "gjpHelper" + suffix;
      fileName = oklpath + kernelName + ".okl";
      platform->kernelRequests.add(kernelName, fileName, props);

      nekrsCheck(BLOCKSIZE < Nq * Nq,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "gjpHelper kernel requires BLOCKSIZE >= Nq * Nq\nBLOCKSIZE = %d, Nq*Nq = %d\n",
                 BLOCKSIZE,
                 Nq * Nq);
    }
  }
}
