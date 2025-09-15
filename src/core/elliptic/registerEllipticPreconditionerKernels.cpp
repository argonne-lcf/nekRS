#include <registerKernels.hpp>
#include "benchmarkFDM.hpp"
#include "benchmarkAx.hpp"
#include "ellipticPrecon.h"

namespace
{

void registerAxKernels(const std::string &section, int N, int poissonEquation)
{
  // hardwired as MG currently does not support Nfields > 1
  constexpr int Nfields{1};
  const auto stressForm = false;

  auto kernelInfo = platform->kernelInfo + meshKernelProperties(N);
  kernelInfo["defines/p_Nfields"] = Nfields;

  std::string fileName;

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");
  const bool serial = platform->serial();
  const std::string fileNameExtension = (serial) ? ".c" : ".okl";
  const std::string optionsPrefix = createOptionsPrefix(section);

  int nelgt, nelgv;
  const std::string meshFile = platform->options.getArgs("MESH FILE");
  re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm());
  const int NelemBenchmark = nelgv / platform->comm.mpiCommSize();

  occa::properties AxKernelInfo = kernelInfo;
  const auto Nq = N + 1;
  for (auto &&coeffField : {true, false}) {
    if (platform->options.compareArgs(optionsPrefix + "ELLIPTIC PRECO COEFF FIELD", "TRUE") != coeffField) {
      continue;
    }
    const auto verbosity = platform->verbose() ? 2 : 1;

    auto axKernel = benchmarkAx<pfloat, pfloat>(
        NelemBenchmark,
        Nq,
        Nq - 1,
        !coeffField,
        poissonEquation,
        false,
        Nfields,
        false, // no stress formulation in preconditioner
        verbosity,
        targetTimeBenchmark,
        platform->options.compareArgs("KERNEL AUTOTUNING", "FALSE") ? false : true);

    std::string kernelNamePrefix = (poissonEquation) ? "poisson-" : "";
    kernelNamePrefix += "elliptic";
    if (Nfields > 1) {
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

    const auto dataType = (std::is_same<pfloat, double>::value) ? "double" : "float";
    kernelName += "Hex3D_" + std::to_string(N) + dataType;

    platform->kernelRequests.add(kernelNamePrefix + "Partial" + kernelName, axKernel);
  }
}

void registerJacobiKernels(const std::string &section, int N, int poissonEquation)
{
  const std::string suffix = "Hex3D";
  const std::string poissonPrefix = poissonEquation ? "poisson-" : "";
  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";
  const std::string optionsPrefix = createOptionsPrefix(section);
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");

  {
    auto props = platform->kernelInfo + meshKernelProperties(N);
    if (poissonEquation) {
      props["defines/p_poisson"] = 1;
    }
    auto kernelName = "ellipticBlockBuildDiagonal" + suffix;
    auto fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(poissonPrefix + kernelName, fileName, props);
  }
}

void registerCommonMGPreconditionerKernels(int N, occa::properties kernelInfo, int poissonEquation)
{
  const std::string prefix = "Hex3D";
  std::string fileName, kernelName;

  kernelInfo["defines/pfloat"] = pfloatString;

  kernelInfo["defines/p_Nfields"] = 1;

  occa::properties pfloatKernelInfo = kernelInfo;
  pfloatKernelInfo["defines/dfloat"] = pfloatString;
  pfloatKernelInfo["defines/pfloat"] = pfloatString;

  const std::string orderSuffix = std::string("_") + std::to_string(N);

  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";

  int p;
  platform->options.getArgs("POLYNOMIAL DEGREE", p);

  if (N != p) {
    std::string fileName;
    const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

    auto meshKernelInfo = platform->kernelInfo + meshKernelProperties(N);

    kernelName = "geometricFactorsHex3D";
    fileName = oklpath + "/core/mesh/" + kernelName + ".okl";
    const std::string meshPrefix = "pMGmesh-";
    platform->kernelRequests.add(meshPrefix + kernelName + orderSuffix,
                                 fileName,
                                 meshKernelInfo,
                                 orderSuffix);
  }

  {
    std::string fileName;
    std::string oklpath;
    oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");

    kernelName = "updateChebyshev";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(kernelName + orderSuffix, fileName, kernelInfo, orderSuffix);

    kernelName = "updateFourthKindChebyshev";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(kernelName + orderSuffix, fileName, kernelInfo, orderSuffix);

    kernelName = "ellipticBlockBuildDiagonalHex3D";
    fileName = oklpath + kernelName + ".okl";
    occa::properties buildDiagInfo = kernelInfo;
    const std::string poissonPrefix = poissonEquation ? "poisson-" : "";
    if (poissonEquation) {
      buildDiagInfo["defines/p_poisson"] = 1;
    }

    occa::properties props = buildDiagInfo;
    props["defines/dfloat"] = pfloatString;
    fileName = oklpath + kernelName + ".okl";
    kernelName = "ellipticBlockBuildDiagonalPfloatHex3D";
    platform->kernelRequests.add(poissonPrefix + kernelName + orderSuffix, fileName, props, orderSuffix);
  }
}

void registerSchwarzKernels(const std::string &section, int N)
{
  const std::string optionsPrefix = createOptionsPrefix(section);
  const int Nq = N + 1;
  const int Nq_e = Nq + 2;
  const int Np = Nq * Nq * Nq;
  const int Np_e = Nq_e * Nq_e * Nq_e;

  const bool serial = platform->serial();
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");

  std::string fileName, kernelName;
  const std::string extension = serial ? ".c" : ".okl";

  {
    occa::properties properties = platform->kernelInfo;
    properties["defines/p_Nq"] = Nq;
    properties["defines/p_Nq_e"] = Nq_e;
    properties["defines/p_restrict"] = 0;
    bool useRAS = platform->options.compareArgs(optionsPrefix + "MULTIGRID SMOOTHER", "RAS");
    const std::string suffix = std::string("_") + std::to_string(Nq_e - 1) + std::string("pfloat");
    if (useRAS) {
      properties["defines/p_restrict"] = 1;
    }

    fileName = oklpath + "MG/fdm/preFDM" + extension;
    platform->kernelRequests.add("preFDM" + suffix, fileName, properties, suffix);

    int nelgt, nelgv;
    const std::string meshFile = platform->options.getArgs("MESH FILE");
    re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm());
    const int NelemBenchmark = nelgv / platform->comm.mpiCommSize();

    const auto verbosity = platform->verbose() ? 2 : 1;

    auto fdmKernel = benchmarkFDM(NelemBenchmark,
                                  Nq_e,
                                  sizeof(pfloat),
                                  useRAS,
                                  verbosity,
                                  targetTimeBenchmark,
                                  platform->options.compareArgs("KERNEL AUTOTUNING", "FALSE") ? false : true,
                                  suffix);
    platform->kernelRequests.add("fusedFDM" + suffix, fdmKernel);

    fileName = oklpath + "MG/fdm/postFDM" + extension;
    platform->kernelRequests.add("postFDM" + suffix, fileName, properties, suffix);
  }
}

void registerFineLevelKernels(const std::string &section, int N, int poissonEquation)
{
  auto gen_suffix = [N](const char *floatString) {
    const std::string precision = std::string(floatString);
    if (precision.find(pfloatString) != std::string::npos) {
      return std::string("_") + std::to_string(N) + std::string("pfloat");
    } else {
      return std::string("_") + std::to_string(N);
    }
  };

  auto kernelInfo = platform->kernelInfo + meshKernelProperties(N);
  registerCommonMGPreconditionerKernels(N, kernelInfo, poissonEquation);

  registerAxKernels(section, N, poissonEquation);
  registerSchwarzKernels(section, N);
}

void registerSEMFEMKernels(const std::string &section, int N, int poissonEquation);

void registerMultigridLevelKernels(const std::string &section,
                                   int Nf,
                                   int N,
                                   int poissonEquation,
                                   bool coarseLevel)
{
  const std::string optionsPrefix = createOptionsPrefix(section);

  const int Nc = N;

  occa::properties kernelInfo = platform->kernelInfo + meshKernelProperties(N);
  const std::string suffix = "Hex3D";
  std::string fileName, kernelName;

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");
  registerCommonMGPreconditionerKernels(N, kernelInfo, poissonEquation);

  const bool serial = platform->serial();
  const std::string fileNameExtension = (serial) ? ".c" : ".okl";

  {
    // sizes for the coarsen and prolongation kernels. degree NFine to degree N
    int NqFine = (Nf + 1);
    int NqCoarse = (Nc + 1);
    occa::properties coarsenProlongateKernelInfo = kernelInfo;
    coarsenProlongateKernelInfo["defines/p_add"] = 1;
    coarsenProlongateKernelInfo["defines/p_NqFine"] = Nf + 1;
    coarsenProlongateKernelInfo["defines/p_NqCoarse"] = Nc + 1;

    const int NpFine = (Nf + 1) * (Nf + 1) * (Nf + 1);
    const int NpCoarse = (Nc + 1) * (Nc + 1) * (Nc + 1);
    coarsenProlongateKernelInfo["defines/p_NpFine"] = NpFine;
    coarsenProlongateKernelInfo["defines/p_NpCoarse"] = NpCoarse;

    const std::string orderSuffix =
        std::string("_Nf_") + std::to_string(Nf) + std::string("_Nc_") + std::to_string(Nc);
    const std::string fileExtension = serial ? ".c" : ".okl";

    const std::string knlpath = getenv("NEKRS_KERNEL_DIR");
    fileName = knlpath + "/core/mesh/coarsen" + suffix + fileNameExtension;
    kernelName = "coarsen" + suffix;
    platform->kernelRequests.add("elliptic::" + kernelName + orderSuffix,
                                 fileName,
                                 coarsenProlongateKernelInfo,
                                 orderSuffix);
    fileName = knlpath + "/core/mesh/prolongate" + suffix + fileNameExtension;
    kernelName = "prolongate" + suffix;
    platform->kernelRequests.add("elliptic::" + kernelName + orderSuffix,
                                 fileName,
                                 coarsenProlongateKernelInfo,
                                 orderSuffix);
  }

  auto smoothedSEMFEM = platform->options.compareArgs(optionsPrefix + "PRECONDITIONER", "MULTIGRID+SEMFEM");

  if (coarseLevel && !smoothedSEMFEM) {
    if (platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVER", "SMOOTHER") ||
        platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVER", "CG") ||
        platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVER", "GMRES")) {
      registerAxKernels(section, N, poissonEquation);
    }
    if (platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVER", "SMOOTHER")) {
      registerSchwarzKernels(section, N);
    }
  } else {
    registerAxKernels(section, N, poissonEquation);
    registerSchwarzKernels(section, N);
  }
}

void registerMultiGridKernels(const std::string &section, int poissonEquation)
{
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const std::string optionsPrefix = createOptionsPrefix(section);

  registerFineLevelKernels(section, N, poissonEquation);

  std::vector<int> levels = determineMGLevels(section);

  if (levels.empty()) {
    return;
  }

  for (auto levelIndex = 1; levelIndex < levels.size(); ++levelIndex) {
    const int levelFine = levels[levelIndex - 1];
    const int levelCoarse = levels[levelIndex];
    auto coarseLevel = (levelIndex == levels.size() - 1);
    registerMultigridLevelKernels(section, levelFine, levelCoarse, poissonEquation, coarseLevel);
  }

  if (platform->options.compareArgs(optionsPrefix + "PRECONDITIONER", "SEMFEM")) {
    const int coarseLevelN = levels.back();
    registerSEMFEMKernels(section, coarseLevelN, poissonEquation);
  }

  if (platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVER", "BOOMERAMG")) {
    const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

    std::string fileName = oklpath + "/core/elliptic/vectorDotStar.okl";
    std::string kernelName = "vectorDotStar";
    platform->kernelRequests.add(kernelName, fileName, platform->kernelInfo);
  }
}

void registerSEMFEMKernels(const std::string &section, int N, int poissonEquation)
{
  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;
  const std::string optionsPrefix = createOptionsPrefix(section);
  const int useFP32 =
      platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE SOLVER PRECISION", "FP32");
  occa::properties SEMFEMKernelProps = platform->kernelInfo;
  if (useFP32) {
    SEMFEMKernelProps["defines/pfloat"] = "float";
  } else {
    SEMFEMKernelProps["defines/pfloat"] = "double";
  }
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");
  std::string fileName = oklpath + "gather.okl";
  platform->kernelRequests.add("gather", fileName, SEMFEMKernelProps);
  fileName = oklpath + "scatter.okl";
  platform->kernelRequests.add("scatter", fileName, SEMFEMKernelProps);
  occa::properties stiffnessKernelInfo = platform->kernelInfo;
  fileName = oklpath + "computeStiffnessMatrix.okl";
  stiffnessKernelInfo["defines/p_Nq"] = Nq;
  stiffnessKernelInfo["defines/p_Np"] = Np;
  stiffnessKernelInfo["defines/p_rows_sorted"] = 1;
  stiffnessKernelInfo["defines/p_cols_sorted"] = 0;

  const bool constructOnHost = !platform->device.deviceAtomic;

  if (!constructOnHost) {
    platform->kernelRequests.add("computeStiffnessMatrix", fileName, stiffnessKernelInfo);
  }
}

} // namespace

void registerEllipticPreconditionerKernels(std::string section)
{
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const std::string optionsPrefix = createOptionsPrefix(section);

  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerEllipticPreconditionerKernels for " << section << std::endl;
  }

  const int poisson = platform->options.compareArgs(optionsPrefix + "HELMHOLTZ TYPE", "POISSON");

  if (platform->options.compareArgs(optionsPrefix + "PRECONDITIONER", "MULTIGRID")) {
    registerMultiGridKernels(section, poisson);
  }
  if (platform->options.getArgs(optionsPrefix + "PRECONDITIONER") == "SEMFEM") {
    registerSEMFEMKernels(section, N, poisson);
  }
  if (platform->options.compareArgs(optionsPrefix + "PRECONDITIONER", "JACOBI")) {
    registerJacobiKernels(section, N, poisson);
  }
}
