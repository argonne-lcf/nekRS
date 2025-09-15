#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "RANSktau.hpp"
#include "linAlg.hpp"

// private members
namespace
{
nrs_t *nrs;

int kFieldIndex;

dfloat rho;
dfloat mueLam;

std::string model;

occa::memory o_mut;

occa::memory o_k;
occa::memory o_tau;

occa::memory o_implicitKtau;

occa::memory o_wbID;
occa::memory o_ywd;

occa::memory o_SijMag2;
occa::memory o_OiOjSk;
occa::memory o_xk;
occa::memory o_xt;
occa::memory o_xtq;

occa::memory o_dgrd;
occa::memory o_OijMag2;

occa::kernel computeKernel;
occa::kernel mueKernel;
occa::kernel limitKernel;
occa::kernel computeGradKernel;
occa::kernel DESLenScaleKernel;

occa::kernel SijMag2OiOjSkKernel;

bool buildKernelCalled = false;
bool setupCalled = false;
bool movingMesh = false;

dfloat coeff[] = {
    0.6,       // sigma_k
    0.5,       // sigma_tau
    1.0,       // alpinf_str
    0.0708,    // beta0
    0.41,      // kappa
    0.09,      // betainf_str
    0.0,       // sigd_min
    1.0 / 8.0, // sigd_max
    400.0,     // fb_c1st
    400.0,     // fb_c2st
    85.0,      // fb_c1
    100.0,     // fb_c2
    0.52,      // alp_inf
    1e-8,      // TINY
    0,         // Pope correction

    // Additional SST parameters
    0.85,      // sigma_k_SST
    0.075,     // beta0_SST
    5.0 / 9.0, // alp_inf_SST
    0.31,      // alp1
    0.0828,    // beta2
    1.0,       // sigk2
    0.856,     // sigom2
    0.44,      // gamma2
    // Free-stream limiter
    0.0, // edd_free //0.01 for external flows
    0.0, // ywlim    //0.5 for external flows

    // DES parameters
    0.78, // cdes1
    0.61, // cdes2
    20.0, // c_d1
    3.0,  // c_d2
    0.41  // vkappa
};

occa::memory implicitK(double time, int scalarIdx)
{
  auto &scalar = nrs->scalar;

  if (scalarIdx == kFieldIndex) {
    return o_implicitKtau.slice(0 * scalar->fieldOffset(), scalar->fieldOffset());
  }
  if (scalarIdx == kFieldIndex + 1) {
    return o_implicitKtau.slice(1 * scalar->fieldOffset(), scalar->fieldOffset());
  }
  return o_NULL;
}

} // namespace

void RANSktau::buildKernel(occa::properties _kernelInfo)
{
  occa::properties kernelInfo;
  if (!kernelInfo.get<std::string>("defines/p_sigma_k").size()) {
    kernelInfo["defines/p_sigma_k"] = coeff[0];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigma_tau").size()) {
    kernelInfo["defines/p_sigma_tau"] = coeff[1];
  }
  if (!kernelInfo.get<std::string>("defines/p_alpinf_str").size()) {
    kernelInfo["defines/p_alpinf_str"] = coeff[2];
  }
  if (!kernelInfo.get<std::string>("defines/p_beta0").size()) {
    kernelInfo["defines/p_beta0"] = coeff[3];
  }
  if (!kernelInfo.get<std::string>("defines/p_kappa").size()) {
    kernelInfo["defines/p_kappa"] = coeff[4];
  }
  if (!kernelInfo.get<std::string>("defines/p_betainf_str").size()) {
    kernelInfo["defines/p_betainf_str"] = coeff[5];
  }
  if (!kernelInfo.get<std::string>("defines/p_ibetainf_str3").size()) {
    kernelInfo["defines/p_ibetainf_str3"] = 1 / pow(coeff[5], 3);
  }
  if (!kernelInfo.get<std::string>("defines/p_sigd_min").size()) {
    kernelInfo["defines/p_sigd_min"] = coeff[6];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigd_max").size()) {
    kernelInfo["defines/p_sigd_max"] = coeff[7];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c1st").size()) {
    kernelInfo["defines/p_fb_c1st"] = coeff[8];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c2st").size()) {
    kernelInfo["defines/p_fb_c2st"] = coeff[9];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c1").size()) {
    kernelInfo["defines/p_fb_c1"] = coeff[10];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c2").size()) {
    kernelInfo["defines/p_fb_c2"] = coeff[11];
  }
  if (!kernelInfo.get<std::string>("defines/p_alp_inf").size()) {
    kernelInfo["defines/p_alp_inf"] = coeff[12];
  }
  if (!kernelInfo.get<std::string>("defines/p_tiny").size()) {
    kernelInfo["defines/p_tiny"] = coeff[13];
  }
  if (!kernelInfo.get<std::string>("defines/p_pope").size()) {
    kernelInfo["defines/p_pope"] = coeff[14];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigmak_SST").size()) {
    kernelInfo["defines/p_sigmak_SST"] = coeff[15];
  }
  if (!kernelInfo.get<std::string>("defines/p_beta0_SST").size()) {
    kernelInfo["defines/p_beta0_SST"] = coeff[16];
  }
  if (!kernelInfo.get<std::string>("defines/p_alpinf_SST").size()) {
    kernelInfo["defines/p_alpinf_SST"] = coeff[17];
  }
  if (!kernelInfo.get<std::string>("defines/p_alp1").size()) {
    kernelInfo["defines/p_alp1"] = coeff[18];
  }
  if (!kernelInfo.get<std::string>("defines/p_beta2").size()) {
    kernelInfo["defines/p_beta2"] = coeff[19];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigk2").size()) {
    kernelInfo["defines/p_sigk2"] = coeff[20];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigom2").size()) {
    kernelInfo["defines/p_sigom2"] = coeff[21];
  }
  if (!kernelInfo.get<std::string>("defines/p_gamma2").size()) {
    kernelInfo["defines/p_gamma2"] = coeff[22];
  }
  if (!kernelInfo.get<std::string>("defines/p_edd_free").size()) {
    kernelInfo["defines/p_edd_free"] = coeff[23];
  }
  if (!kernelInfo.get<std::string>("defines/p_ywlim").size()) {
    kernelInfo["defines/p_ywlim"] = coeff[24];
  }
  if (!kernelInfo.get<std::string>("defines/p_cdes1").size()) {
    kernelInfo["defines/p_cdes1"] = coeff[25];
  }
  if (!kernelInfo.get<std::string>("defines/p_cdes2").size()) {
    kernelInfo["defines/p_cdes2"] = coeff[26];
  }
  if (!kernelInfo.get<std::string>("defines/p_cd1").size()) {
    kernelInfo["defines/p_cd1"] = coeff[27];
  }
  if (!kernelInfo.get<std::string>("defines/p_cd2").size()) {
    kernelInfo["defines/p_cd2"] = coeff[28];
  }
  if (!kernelInfo.get<std::string>("defines/p_vkappa").size()) {
    kernelInfo["defines/p_vkappa"] = coeff[29];
  }

  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "\nRANSktau settings\n";
    std::cout << kernelInfo << std::endl;
  }

  kernelInfo += _kernelInfo;

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/plugins/");
    const auto fileName = path + "RANSktau.okl";
    const auto reqName = "RANSktau::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  computeKernel = buildKernel("RANSktauCompute");
  mueKernel = buildKernel("mue");
  limitKernel = buildKernel("limit");
  SijMag2OiOjSkKernel = buildKernel("SijMag2OiOjSk");
  computeGradKernel = buildKernel("RANSGradHex3D");
  DESLenScaleKernel = buildKernel("DESLenScale");

  int Nscalar;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalar);

  nekrsCheck(Nscalar < 2, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "Nscalar needs to be >= 2!");
  platform->options.setArgs("FLUID STRESSFORMULATION", "TRUE");
}

void RANSktau::updateProperties()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  platform->options.getArgs("FLUID VISCOSITY", mueLam);
  platform->options.getArgs("FLUID DENSITY", rho);

  auto mesh = nrs->fluid->mesh;

  limitKernel(mesh->Nlocal, o_k, o_tau);

  auto o_SijOij = nrs->strainRotationRate();

  bool ifktau = 1;
  if (model != "KTAU") {
    ifktau = 0;
  }

  SijMag2OiOjSkKernel(mesh->Nlocal,
                      nrs->fluid->fieldOffset,
                      static_cast<int>(ifktau),
                      o_SijOij,
                      o_OiOjSk,
                      o_SijMag2);

  if (model == "KTAUSST+DDES" || model == "KTAUSST+IDDES") {
    auto o_Oij = o_SijOij.slice(6 * nrs->fluid->fieldOffset);
    platform->linAlg->magSqrVector(mesh->Nlocal, nrs->fluid->fieldOffset, o_Oij, o_OijMag2);
  }

  computeGradKernel(mesh->Nelements,
                    nrs->scalar->fieldOffset(),
                    mesh->o_vgeo,
                    mesh->o_D,
                    o_k,
                    o_tau,
                    o_xk,
                    o_xt,
                    o_xtq);

  if (movingMesh && !ifktau) {
    o_ywd = mesh->minDistance(o_wbID.size(), o_wbID, "cheap_dist");
  }

  mueKernel(mesh->Nlocal,
            nrs->fluid->fieldOffset,
            rho,
            mueLam,
            static_cast<int>(ifktau),
            o_k,
            o_tau,
            o_SijMag2,
            o_xk,
            o_ywd,
            o_mut,
            nrs->fluid->o_mue,
            nrs->scalar->o_diff + nrs->scalar->fieldOffsetScan[kFieldIndex]);
}

const deviceMemory<dfloat> RANSktau::o_mue_t()
{
  deviceMemory<dfloat> out(o_mut);
  return out;
}

void RANSktau::updateSourceTerms()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  auto mesh = nrs->fluid->mesh;
  auto &scalar = nrs->scalar;

  platform->options.getArgs("FLUID VISCOSITY", mueLam);
  platform->options.getArgs("FLUID DENSITY", rho);

  bool ifktau = 1;
  if (model != "KTAU") {
    ifktau = 0;
  }

  int ifdes = 0; // DES model type
  if (model == "KTAUSST+DDES") {
    ifdes = 1;
  }
  if (model == "KTAUSST+IDDES") {
    ifdes = 2;
  }

  if (ifdes && movingMesh) {
    DESLenScaleKernel(mesh->Nelements, nrs->fluid->fieldOffset, mesh->o_x, mesh->o_y, mesh->o_z, o_dgrd);
  }

  computeKernel(mesh->Nlocal,
                nrs->fluid->fieldOffset,
                static_cast<int>(ifktau),
                ifdes,
                rho,
                mueLam,
                o_k,
                o_tau,
                o_SijMag2,
                o_OiOjSk,
                o_xk,
                o_xt,
                o_xtq,
                o_dgrd,
                o_ywd,
                o_OijMag2,
                o_implicitKtau,
                scalar->o_EXT + scalar->fieldOffsetScan[kFieldIndex]);
}

void RANSktau::setup(int ifld, std::string modelIn)
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  model = upperCase(modelIn);

  if (platform->comm.mpiRank() == 0) {
    printf("RANS model: %s\n", model.c_str());
  }

  nekrsCheck(model != "KTAU" && model != "KTAUSST" && model != "KTAUSST+DDES" && model != "KTAUSST+IDDES",
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "Specified RANS model not supported!\nAvailable RANS models "
             "are:\nKTAU\nKTAUSST\nKTAUSST+DDES\nKTAUSST+IDDES");

  nrs = dynamic_cast<nrs_t *>(platform->app);
  kFieldIndex = ifld; // tauFieldIndex is assumed to be kFieldIndex+1

  auto &scalar = nrs->scalar;
  nekrsCheck(scalar->NSfields < kFieldIndex + 1,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "number of scalar fields too low!");

  for (int i = 0; i < 2; i++) {
    auto &scalar = nrs->scalar;

    platform->options.getArgs("FLUID DENSITY", rho);
    auto o_rho =
        scalar->o_rho.slice(scalar->fieldOffsetScan[kFieldIndex + i], scalar->mesh(kFieldIndex + i)->Nlocal);
    platform->linAlg->fill(o_rho.size(), rho, o_rho);

    const std::string sid = scalarDigitStr(kFieldIndex + i);
    nekrsCheck(!platform->options.getArgs("SCALAR" + sid + " DIFFUSIVITY").empty() ||
                   !platform->options.getArgs("SCALAR" + sid + " DENSITY").empty(),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "illegal property specification for k/tau in par!");
  }

  o_k = scalar->o_S + scalar->fieldOffsetScan[kFieldIndex];
  o_tau = scalar->o_S + scalar->fieldOffsetScan[kFieldIndex + 1];
  o_mut = platform->device.malloc<dfloat>(scalar->mesh(kFieldIndex)->Nlocal);
  o_implicitKtau = platform->device.malloc<dfloat>(2 * scalar->fieldOffset());

  scalar->userImplicitLinearTerm = implicitK;

  o_OiOjSk = platform->device.malloc<dfloat>(nrs->fluid->fieldOffset);
  o_SijMag2 = platform->device.malloc<dfloat>(nrs->fluid->fieldOffset);
  o_xk = platform->device.malloc<dfloat>(scalar->fieldOffset());
  o_xt = platform->device.malloc<dfloat>(scalar->fieldOffset());
  o_xtq = platform->device.malloc<dfloat>(scalar->fieldOffset());

  movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  auto mesh = nrs->fluid->mesh;

  if (model != "KTAU") {
    std::vector<int> wbID;
    for (auto &[key, bcID] : platform->app->bc->bIdToTypeId()) {
      const auto field = key.first;
      if (field == "fluid velocity") {
        if (bcID == bdryBase::bcType_zeroDirichlet) {
          wbID.push_back(key.second + 1);
        }
      }
    }
    o_wbID = platform->device.malloc<int>(wbID.size(), wbID.data());

    if (!movingMesh) {
      o_ywd = mesh->minDistance(o_wbID.size(), o_wbID, "cheap_dist");
    }
  }

  if (model == "KTAUSST+DDES" || model == "KTAUSST+IDDES") {
    o_dgrd = platform->device.malloc<dfloat>(mesh->Nelements);
    o_OijMag2 = platform->device.malloc<dfloat>(nrs->fluid->fieldOffset);

    if (!movingMesh) {
      DESLenScaleKernel(mesh->Nelements, nrs->fluid->fieldOffset, mesh->o_x, mesh->o_y, mesh->o_z, o_dgrd);
    }
  }
  setupCalled = true;
}
