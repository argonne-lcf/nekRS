#include "registerKernels.hpp"
#include "linearSolver.hpp"

namespace
{

template <typename T> void registerGMRESKernels(int Nfields)
{
  const std::string fileNameExtension = (platform->serial()) ? ".c" : ".okl";
  const auto sectionIdentifier = std::string("gmres::") +
                                 ((std::is_same<T, double>::value) ? "double::" : "float::") +
                                 std::to_string(Nfields) + "::";
  const auto oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/linearSolver/");

  occa::properties gmresKernelInfo = platform->kernelInfo;
  gmresKernelInfo["defines/dfloat"] = (std::is_same<T, double>::value) ? "double" : "float";

  gmresKernelInfo["defines/p_Nfields"] = Nfields;

  std::string fileName;

  std::string kernelName = "gramSchmidtOrthogonalization";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);

  kernelName = "updatePGMRESSolution";
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);

  kernelName = "PGMRESSolution";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);

  kernelName = "fusedResidualAndNorm";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);
}

template <typename T> void registerCGKernels(int Nfields, bool useFloat = false)
{
  const auto oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/linearSolver/");
  const std::string fileNameExtension = (platform->serial()) ? ".c" : ".okl";

  const auto sectionIdentifier = std::string("cg::") +
                                 ((std::is_same<T, double>::value) ? "double::" : "float::") +
                                 std::to_string(Nfields) + "::";

  occa::properties properties = platform->kernelInfo;
  properties["defines/p_Nfields"] = Nfields;
  properties["defines/dfloat"] = (std::is_same<T, double>::value) ? "double" : "float";

  std::string kernelName;
  std::string fileName;

  kernelName = "blockUpdatePCG";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, properties);

  occa::properties combinedPCGInfo = properties;

  kernelName = "combinedPCGPreMatVec";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, combinedPCGInfo);

  kernelName = "combinedPCGUpdateConvergedSolution";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, combinedPCGInfo);

  combinedPCGInfo["defines/p_nReduction"] = CombinedPCGId::nReduction;
  combinedPCGInfo["defines/p_gamma"] = CombinedPCGId::gamma;
  combinedPCGInfo["defines/p_a"] = CombinedPCGId::a;
  combinedPCGInfo["defines/p_b"] = CombinedPCGId::b;
  combinedPCGInfo["defines/p_c"] = CombinedPCGId::c;
  combinedPCGInfo["defines/p_d"] = CombinedPCGId::d;
  combinedPCGInfo["defines/p_e"] = CombinedPCGId::e;
  combinedPCGInfo["defines/p_f"] = CombinedPCGId::f;

  kernelName = "combinedPCGPostMatVec";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, combinedPCGInfo);
}

} // namespace

void registerLinearSolverKernels()
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerLinearSolverKernels" << std::endl;
  }

  registerGMRESKernels<dfloat>(1);
  registerGMRESKernels<dfloat>(3);
  registerCGKernels<dfloat>(1);
  registerCGKernels<dfloat>(3);

  if (!std::is_same<pfloat, dfloat>::value) {
    registerCGKernels<pfloat>(1); // Krylov based coarse solve
  }
}
