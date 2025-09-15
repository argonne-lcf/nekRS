#include <registerKernels.hpp>
#include "elliptic.h"
#include "benchmarkAx.hpp"

void registerEllipticKernels(std::string section, bool stressForm)
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerEllipticKernels for " << section << std::endl;
  }

  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const std::string optionsPrefix = createOptionsPrefix(section);

  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();
  kernelInfo += meshKernelProperties(N);

  const auto poisson = platform->options.compareArgs(optionsPrefix + "HELMHOLTZ TYPE", "POISSON");

  const auto blockSolver = [&]() {
    if (platform->options.compareArgs(optionsPrefix + "SOLVER", "BLOCK")) {
      return true;
    }
    if (stressForm) {
      return true;
    }

    return false;
  }();
  const int Nfields = (blockSolver) ? 3 : 1;

  const bool serial = platform->serial();
  const std::string fileNameExtension = (serial) ? ".c" : ".okl";
  const std::string sectionIdentifier = std::to_string(Nfields) + "-";

  {
    const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");
    std::string fileName, kernelName;

    {
      const std::string extension = ".okl";
      occa::properties properties = platform->kernelInfo;

      kernelName = "fusedCopyDfloatToPfloat";
      fileName = oklpath + kernelName + fileNameExtension;
      platform->kernelRequests.add(kernelName, fileName, properties);

      properties["defines/p_Nfields"] = Nfields;

      kernelName = "multiScaledAddwOffset";
      fileName = oklpath + kernelName + extension;
      platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, properties);

      kernelName = "accumulate";
      fileName = oklpath + kernelName + extension;
      platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, properties);
    }
  }

  int nelgt, nelgv;
  const std::string meshFile = platform->options.getArgs("MESH FILE");
  re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm());

  //  if (section.find("elliptic") != std::string::npos) return;

  const int NelemBenchmark = nelgv / platform->comm.mpiCommSize();
  bool verbose = platform->verbose();
  const int verbosity = verbose ? 2 : 1;

  for (auto &&coeffField : {true, false}) {
    if (!platform->options.compareArgs(optionsPrefix + "RHO SPLITTING", "TRUE") &&
        platform->options.compareArgs(optionsPrefix + "ELLIPTIC COEFF FIELD", "TRUE") != coeffField) {
      continue;
    }

    auto addRequest = [&](const std::string &dataType, occa::kernel &kernel) {
      std::string kernelNamePrefix = (poisson) ? "poisson-" : "";
      kernelNamePrefix += "elliptic";
      if (blockSolver) {
        kernelNamePrefix += (stressForm) ? "Stress" : "Block";
      }
      std::string kernelName = "Ax";
      if (coeffField) {
        kernelName += "Var";
      }
      kernelName += "Coeff";
      if (platform->options.compareArgs("ELEMENT MAP", "TRILINEAR")) {
        kernelName += "Trilinear";
      }

      kernelName += "Hex3D_" + std::to_string(N) + dataType;

      platform->kernelRequests.add(kernelNamePrefix + "Partial" + kernelName, kernel);
    };

    auto axKernel = [&](auto typeTag, auto geoTypeTag) {
      return benchmarkAx<decltype(typeTag), decltype(geoTypeTag)>(
          NelemBenchmark,
          N + 1,
          N,
          !coeffField,
          poisson,
          false,
          Nfields,
          stressForm,
          verbosity,
          targetTimeBenchmark,
          platform->options.compareArgs("KERNEL AUTOTUNING", "FALSE") ? false : true);
    };

    {
      auto kernel = axKernel(dfloat{}, dfloat{});
      if (platform->options.compareArgs("BUILD ONLY", "FALSE")) {
        addRequest(dfloatString, kernel);
      }
    }

    if (std::is_same<dfloat, float>::value && platform->options.compareArgs(optionsPrefix + "SOLVER", "IR")) {
      auto kernel = axKernel(double{}, float{}); // required from GMRES-IR
      if (platform->options.compareArgs("BUILD ONLY", "FALSE")) {
        addRequest("double", kernel);
      }
    }
  }

  registerEllipticPreconditionerKernels(section);
}
