#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "udf.hpp"

#include "lowMach.hpp"
#include "linAlg.hpp"

namespace
{

nrs_t *_nrs = nullptr;

int qThermal = 0;
dfloat alpha0 = 1.0;

occa::memory o_beta;
occa::memory o_kappa;

occa::memory o_bID;

occa::kernel qtlKernel;
occa::kernel p0thHelperKernel;

static bool buildKernelCalled = false;
static bool setupCalled = false;

} // namespace

void lowMach::buildKernel(occa::properties kernelInfo)
{
  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/plugins/");
    const auto fileName = path + "lowMach.okl";
    const auto reqName = "lowMach::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  qtlKernel = buildKernel("qtlHex3D");
  p0thHelperKernel = buildKernel("p0thHelper");

  platform->options.setArgs("LOWMACH", "TRUE");
}

void lowMach::setup(dfloat alpha_, const occa::memory &o_beta_, const occa::memory &o_kappa_)
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  _nrs = dynamic_cast<nrs_t *>(platform->app);
  ;

  alpha0 = alpha_;
  _nrs->alpha0Ref = alpha0;
  o_beta = o_beta_;
  o_kappa = o_kappa_;

  nekrsCheck(_nrs->scalar->nameToIndex.find("temperature") == _nrs->scalar->nameToIndex.end(),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "requires solving for temperature!");

  std::vector<int> bID;
  for (auto &[key, bcID] : platform->app->bc->bIdToTypeId()) {
    const auto field = key.first;
    if (field == "fluid velocity") {
      if (bcID == bdryBase::bcType_udfDirichlet || bcID == bdryBase::bcType_interpolation) {
        bID.push_back(key.second + 1);
      }
    }
  }
  o_bID = platform->device.malloc<int>(bID.size());
  o_bID.copyFrom(bID.data());

  setupCalled = true;
}

void lowMach::qThermalSingleComponent(double time)
{
  auto o_div = _nrs->fluid->o_div;
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  qThermal = 1;
  auto nrs = _nrs;
  auto &scalar = nrs->scalar;
  auto mesh = nrs->fluid->mesh;
  linAlg_t *linAlg = platform->linAlg;

  std::string scope = "udfDiv::";

  bool rhsCVODE = false;
  if (scalar->cvode) {
    rhsCVODE = scalar->cvode->isRhsEvaluation();
  }

  auto o_gradT = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * nrs->fluid->fieldOffset);

  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               nrs->fluid->fieldOffset,
               scalar->o_S,
               o_gradT);

  double flopsGrad = 6 * mesh->Np * mesh->Nq + 18 * mesh->Np;
  flopsGrad *= static_cast<double>(mesh->Nelements);

  oogs::startFinish(o_gradT, mesh->dim, nrs->fluid->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);

  platform->linAlg->axmyVector(mesh->Nlocal, nrs->fluid->fieldOffset, 0, 1.0, mesh->o_invLMM, o_gradT);

  auto o_src = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);
  platform->linAlg->fill(mesh->Nlocal, 0.0, o_src);
  if (nrs->userSource) {
    platform->timer.tic(scope + "udfSEqnSource");
    auto o_saveEXT = scalar->o_EXT;
    scalar->o_EXT = o_src;
    platform->callerScope = "qThermal";
    nrs->userSource(time);
    platform->callerScope.clear();
    scalar->o_EXT = o_saveEXT;
    platform->timer.toc(scope + "udfSEqnSource");
  }

  qtlKernel(mesh->Nelements,
            mesh->o_vgeo,
            mesh->o_D,
            nrs->fluid->fieldOffset,
            o_gradT,
            o_beta,
            scalar->o_diff,
            scalar->o_rho,
            o_src,
            o_div);

  o_gradT.free();
  o_src.free();

  double flopsQTL = 18 * mesh->Np * mesh->Nq + 23 * mesh->Np;
  flopsQTL *= static_cast<double>(mesh->Nelements);

  oogs::startFinish(o_div, 1, nrs->fluid->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs);

  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_div);

  double surfaceFlops = 0.0;

  if (!platform->app->bc->hasOutflow("fluid velocity")) {
    nekrsCheck(rhsCVODE,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s\n",
               "computing p0th and dp0thdt using CVODE is not supported!");

    const auto termQ = [&]() {
      auto o_tmp = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);
      linAlg->axmyz(mesh->Nlocal, 1.0, mesh->o_LMM, o_div, o_tmp);
      return linAlg->sum(mesh->Nlocal, o_tmp, platform->comm.mpiComm());
    }();

    auto o_tmp1 = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);
    auto o_tmp2 = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);
    p0thHelperKernel(mesh->Nlocal,
                     alpha0,
                     nrs->p0th[0],
                     o_beta,
                     o_kappa,
                     scalar->o_rho,
                     mesh->o_LMM,
                     o_tmp1,
                     o_tmp2);

    double p0thHelperFlops = 4 * mesh->Nlocal;

    const auto termV = mesh->surfaceAreaNormalMultiplyVectorIntegrate(nrs->fluid->fieldOffset,
                                                                      o_bID,
                                                                      nrs->fluid->o_Ue);

    double surfaceFluxFlops = 13 * mesh->Nq * mesh->Nq;
    surfaceFluxFlops *= static_cast<double>(mesh->Nelements);

    const auto prhs = (termQ - termV) / linAlg->sum(mesh->Nlocal, o_tmp1, platform->comm.mpiComm());
    linAlg->axpby(mesh->Nlocal, -prhs, o_tmp2, 1.0, o_div);
    o_tmp1.free();
    o_tmp2.free();

    std::vector<dfloat> coeff(nrs->o_coeffBDF.size());
    nrs->o_coeffBDF.copyTo(coeff.data());
    dfloat Saqpq = 0.0;
    for (int i = 0; i < nrs->o_coeffBDF.size(); ++i) {
      Saqpq += coeff[i] * nrs->p0th[i];
    }

    const auto g0 = nrs->g0;
    const auto dt = nrs->dt[0];

    const auto pcoef = (g0 - dt * prhs);
    const auto p0thn = Saqpq / pcoef;

    nrs->p0th[2] = nrs->p0th[1];
    nrs->p0th[1] = nrs->p0th[0];
    nrs->p0th[0] = p0thn;

    nrs->dp0thdt = prhs * p0thn;

    surfaceFlops += surfaceFluxFlops + p0thHelperFlops;
  }

  qThermal = 0;

  double flops = surfaceFlops + flopsGrad + flopsQTL;
  platform->flopCounter->add("lowMach::qThermalRealGasSingleComponent", flops);
}

void lowMach::dpdt(occa::memory &o_FU)
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  auto nrs = _nrs;
  auto mesh = nrs->fluid->mesh;

  if (nrs->scalar->cvodeSolve[0]) {
    return; // contribution is not applied here
  }

  if (!qThermal) {
    platform->linAlg->add(mesh->Nlocal, nrs->dp0thdt * alpha0, o_FU);
  }
}
