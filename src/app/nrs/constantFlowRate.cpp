#include "platform.hpp"
#include "linAlg.hpp"
#include "nrs.hpp"
#include "udf.hpp"
#include "alignment.hpp"
#include "bdryBase.hpp"
#include "constantFlowRate.hpp"

namespace
{

double flops = 0;
dfloat rescaleFactor = 0;

occa::memory o_Uc;
occa::memory o_Pc;
occa::memory o_prevProp;

occa::memory o_Urhs;
occa::memory o_Prhs;

inline dfloat distance(dfloat x1, dfloat x2, dfloat y1, dfloat y2, dfloat z1, dfloat z2)
{
  const dfloat dist_x = x1 - x2;
  const dfloat dist_y = y1 - y2;
  const dfloat dist_z = z1 - z2;
  return std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);
}

void computeDirection(dfloat x1, dfloat x2, dfloat y1, dfloat y2, dfloat z1, dfloat z2, dfloat *direction)
{
  direction[0] = x1 - x2;
  direction[1] = y1 - y2;
  direction[2] = z1 - z2;

  const dfloat invMagnitude = 1 / distance(x1, x1, y1, y2, z1, z2);

  direction[0] *= invMagnitude;
  direction[1] *= invMagnitude;
  direction[2] *= invMagnitude;
}

dfloat lengthScale;
dfloat baseFlowRate;
dfloat currentFlowRate;
dfloat postCorrectionFlowRate;
dfloat flowRate;

int fromBID;
int toBID;
dfloat flowDirection[3];

bool checkIfRecomputeDirection(int tstep)
{
  return platform->options.compareArgs("MOVING MESH", "TRUE") || tstep < 2;
}

auto getSolverData(elliptic *solver) 
{
  if (solver) {
    std::tuple<int, dfloat, dfloat, dfloat> val(solver->Niter(),
                                                solver->initialResidual(),
                                                solver->initialGuessResidual(),
                                                solver->finalResidual());
    return val;
  } else {
    std::tuple<int, dfloat, dfloat, dfloat> val(0, 0, 0, 0);
    return val;
  }
}

auto setSolverData(elliptic *solver, int Niter, dfloat res00Norm, dfloat res0Norm, dfloat resNorm) 
{
  solver->Niter(solver->Niter() + Niter);
  solver->initialResidual(res00Norm);
  solver->initialGuessResidual(res0Norm);
  solver->finalResidual(resNorm);
}

} // namespace


flowRate_t::flowRate_t(fluidSolver_t *fluidRef)
        : fluid(fluidRef)
{
  auto& mesh = fluid->mesh;
  flops = 0.0;

  o_Uc = platform->device.malloc<dfloat>(mesh->dim * fluid->fieldOffset);
  o_prevProp = platform->device.malloc<dfloat>(fluid->o_prop.size());
  o_prevProp.copyFrom(fluid->o_prop);
}

void flowRate_t::rhsPressure(double time, int iter)
{
  auto &mesh = fluid->mesh;
  const auto fieldOffset = fluid->fieldOffset;

  double flops = 0.0;

  auto o_lambda0 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->adyz(mesh->Nlocal, 1.0, fluid->o_rho, o_lambda0);

  // t_flow \dot grad(1/rho)
  platform->timer.tic(fluid->pressureName + " rhs");
  auto o_gradPCoeff = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * fieldOffset);
  launchKernel("core-wGradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               fieldOffset,
               o_lambda0,
               o_gradPCoeff);

  double flopsGrad = 6 * mesh->Np * mesh->Nq + 18 * mesh->Np;
  flopsGrad *= static_cast<double>(mesh->Nelements);
  flops += flopsGrad;

  if (!o_Prhs.isInitialized()) {
    o_Prhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  }

  launchKernel("nrs-computeFieldDotNormal",
               mesh->Nlocal,
               fieldOffset,
               flowDirection[0],
               flowDirection[1],
               flowDirection[2],
               o_gradPCoeff,
               o_Prhs);

  flops += 5 * mesh->Nlocal;
  platform->flopCounter->add("ConstantFlowRate::compute", flops);

  platform->timer.toc(fluid->pressureName + " rhs");
}

void flowRate_t::solvePressure(double time, int iter)
{
  auto &mesh = fluid->mesh;
  auto &pSolver = fluid->ellipticSolverP;
  const auto [NiterP, res00NormP, res0NormP, resNormP] = getSolverData(pSolver);

  if (!o_Pc.isInitialized()) {
    o_Pc = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  }
  platform->linAlg->fill(o_Pc.size(), 0, o_Pc);

  fluid->solvePressure(time, iter, o_Prhs, o_Pc);
  o_Prhs.free();

  setSolverData(pSolver, NiterP, res00NormP, res0NormP, resNormP);
}

void flowRate_t::rhsVelocity(double time, int iter)
{
  auto &mesh = fluid->mesh;
  const auto fieldOffset = fluid->fieldOffset;

  double flops = 0.0;

  platform->timer.tic(fluid->velocityName + " rhs");

  if (!o_Urhs.isInitialized()){
    o_Urhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * fieldOffset);
  }

  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               fieldOffset,
               o_Pc,
               o_Urhs);

  double flopsGrad = 6 * mesh->Np * mesh->Nq + 18 * mesh->Np;
  flopsGrad *= static_cast<double>(mesh->Nelements);
  flops += flopsGrad;

  platform->linAlg->scaleMany(mesh->Nlocal, mesh->dim, fieldOffset, -1.0, o_Urhs);

  auto o_JwF = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * fieldOffset);
  o_JwF.copyFrom(mesh->o_LMM, mesh->Nlocal, 0 * fieldOffset, 0);
  o_JwF.copyFrom(mesh->o_LMM, mesh->Nlocal, 1 * fieldOffset, 0);
  o_JwF.copyFrom(mesh->o_LMM, mesh->Nlocal, 2 * fieldOffset, 0);

  for (int dim = 0; dim < mesh->dim; ++dim) {
    const dlong offset = dim * fieldOffset;
    const dfloat n_dim = flowDirection[dim];
    platform->linAlg->axpby(mesh->Nlocal, n_dim, o_JwF, 1.0, o_Urhs, offset, offset);
  }
  platform->timer.toc(fluid->velocityName + " rhs");
}

void flowRate_t::solveVelocity(double time, int iter)
{
  auto &mesh = fluid->mesh;

  auto &uvwSolver = fluid->ellipticSolver.at(0);
  const auto [NiterUVW, res00NormUVW, res0NormUVW, resNormUVW] = getSolverData(uvwSolver);

  auto uSolver = (fluid->ellipticSolver.size() == 1) ? fluid->ellipticSolver.at(0) : nullptr;
  auto vSolver = (fluid->ellipticSolver.size() == 2) ? fluid->ellipticSolver.at(1) : nullptr;
  auto wSolver = (fluid->ellipticSolver.size() == 3) ? fluid->ellipticSolver.at(2) : nullptr;
  const auto [NiterU, res00NormU, res0NormU, resNormU] = getSolverData(uSolver);
  const auto [NiterV, res00NormV, res0NormV, resNormV] = getSolverData(vSolver);
  const auto [NiterW, res00NormW, res0NormW, resNormW] = getSolverData(wSolver);

  fluid->solveVelocity(time, 2, o_Urhs, o_Uc);
  o_Urhs.free(); 

  platform->flopCounter->add("ConstantFlowRate::compute", flops);
  if (fluid->ellipticSolver.at(0)->Nfields() == mesh->dim) {
    setSolverData(uvwSolver, NiterUVW, res00NormUVW, res0NormUVW, resNormUVW);
  } else {
    setSolverData(uSolver, NiterU, res00NormU, res0NormU, resNormU);
    setSolverData(vSolver, NiterV, res00NormV, res0NormV, resNormV);
    setSolverData(wSolver, NiterW, res00NormW, res0NormW, resNormW);
  }

  // Jw * n \dot Uc
  auto o_baseFlowRate = platform->deviceMemoryPool.reserve<dfloat>(fluid->fieldOffset);
  launchKernel("nrs-computeFieldDotNormal",
               mesh->Nlocal,
               fluid->fieldOffset,
               flowDirection[0],
               flowDirection[1],
               flowDirection[2],
               o_Uc,
               o_baseFlowRate);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_baseFlowRate);
  flops += 4 * mesh->Nlocal;

  baseFlowRate = platform->linAlg->sum(mesh->Nlocal, o_baseFlowRate, platform->comm.mpiComm()) / lengthScale;
}

void flowRate_t::printInfo(int tstep, bool verboseInfo) const
{
  auto &mesh = fluid->mesh;

  if (platform->comm.mpiRank() != 0) {
    return;
  }

  std::string flowRateType = "flowRate";

  dfloat currentRate = currentFlowRate;
  dfloat finalFlowRate = postCorrectionFlowRate;
  dfloat userSpecifiedFlowRate = flowRate * mesh->volume / lengthScale;

  dfloat err = std::abs(userSpecifiedFlowRate - finalFlowRate);

  dfloat scale = rescaleFactor; // rho * meanGradP

  if (!platform->options.compareArgs("CONSTANT FLOW RATE TYPE", "VOLUMETRIC")) {
    flowRateType = "uBulk";

    // put in bulk terms, instead of volumetric
    currentRate *= lengthScale / mesh->volume;
    finalFlowRate *= lengthScale / mesh->volume;
    userSpecifiedFlowRate = flowRate;
    err = std::abs(userSpecifiedFlowRate - finalFlowRate);
  }
  if (verboseInfo) {
    printf("step=%-8d %-20s: %s0 %.2e  %s %.2e  err %.2e  scale %.5e\n",
           tstep,
           "flowrate",
           flowRateType.c_str(),
           currentRate,
           flowRateType.c_str(),
           finalFlowRate,
           err,
           scale);
  }
}

bool flowRate_t::computationRequired(double time, int tstep)
{
  if (platform->options.compareArgs(upperCase(fluid->velocityName) + " SOLVER", "NONE")) {
    return false;
  }

  static dfloat prevTime = -1;

  platform->options.getArgs("FLOW RATE", flowRate);

  const bool X_aligned = platform->options.compareArgs("CONSTANT FLOW DIRECTION", "X");
  const bool Y_aligned = platform->options.compareArgs("CONSTANT FLOW DIRECTION", "Y");
  const bool Z_aligned = platform->options.compareArgs("CONSTANT FLOW DIRECTION", "Z");
  const bool directionAligned = X_aligned || Y_aligned || Z_aligned;

  nekrsCheck(!directionAligned,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "Flow direction is not aligned in (X,Y,Z)");

  auto mesh = fluid->mesh;

  const bool recomputeDirection = checkIfRecomputeDirection(tstep);

  if (recomputeDirection) {
    if (directionAligned) {
      occa::memory o_coord;
      if (X_aligned) {
        o_coord = mesh->o_x;
        flowDirection[0] = 1.0;
        flowDirection[1] = 0.0;
        flowDirection[2] = 0.0;
      }
      if (Y_aligned) {
        o_coord = mesh->o_y;
        flowDirection[0] = 0.0;
        flowDirection[1] = 1.0;
        flowDirection[2] = 0.0;
      }
      if (Z_aligned) {
        o_coord = mesh->o_z;
        flowDirection[0] = 0.0;
        flowDirection[1] = 0.0;
        flowDirection[2] = 1.0;
      }

      const dfloat maxCoord = platform->linAlg->max(mesh->Nlocal, o_coord, platform->comm.mpiComm());
      const dfloat minCoord = platform->linAlg->min(mesh->Nlocal, o_coord, platform->comm.mpiComm());
      lengthScale = maxCoord - minCoord;
    } else {

      platform->options.getArgs("CONSTANT FLOW FROM BID", fromBID);
      platform->options.getArgs("CONSTANT FLOW TO BID", toBID);

      auto o_centroid =
          platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * mesh->Nelements * mesh->Nfaces);
      platform->linAlg->fill(mesh->Nelements * mesh->Nfaces * 3, 0.0, o_centroid);

      auto o_counts = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements * mesh->Nfaces);
      platform->linAlg->fill(mesh->Nelements * mesh->Nfaces, 0.0, o_counts);

      launchKernel("nrs-computeFaceCentroid",
                   mesh->Nelements,
                   fromBID,
                   mesh->o_EToB,
                   mesh->o_vmapM,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   o_centroid,
                   o_counts);
      flops += 3 * mesh->Nlocal;

      dfloat NfacesContrib =
          platform->linAlg->sum(mesh->Nelements * mesh->Nfaces, o_counts, platform->comm.mpiComm());
      dfloat sumFaceAverages_x = platform->linAlg->sum(mesh->Nelements * mesh->Nfaces,
                                                       o_centroid,
                                                       platform->comm.mpiComm(),
                                                       0 * mesh->Nelements * mesh->Nfaces);
      dfloat sumFaceAverages_y = platform->linAlg->sum(mesh->Nelements * mesh->Nfaces,
                                                       o_centroid,
                                                       platform->comm.mpiComm(),
                                                       1 * mesh->Nelements * mesh->Nfaces);
      dfloat sumFaceAverages_z = platform->linAlg->sum(mesh->Nelements * mesh->Nfaces,
                                                       o_centroid,
                                                       platform->comm.mpiComm(),
                                                       2 * mesh->Nelements * mesh->Nfaces);

      const dfloat centroidFrom_x = sumFaceAverages_x / NfacesContrib;
      const dfloat centroidFrom_y = sumFaceAverages_y / NfacesContrib;
      const dfloat centroidFrom_z = sumFaceAverages_z / NfacesContrib;

      platform->linAlg->fill(mesh->Nelements * mesh->Nfaces * 3, 0.0, o_centroid);
      platform->linAlg->fill(mesh->Nelements * mesh->Nfaces, 0.0, o_counts);
      launchKernel("nrs-computeFaceCentroid",
                   mesh->Nelements,
                   toBID,
                   mesh->o_EToB,
                   mesh->o_vmapM,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   o_centroid,
                   o_counts);

      flops += 3 * mesh->Nlocal;

      NfacesContrib =
          platform->linAlg->sum(mesh->Nelements * mesh->Nfaces, o_counts, platform->comm.mpiComm());
      sumFaceAverages_x = platform->linAlg->sum(mesh->Nelements * mesh->Nfaces,
                                                o_centroid,
                                                platform->comm.mpiComm(),
                                                0 * mesh->Nelements * mesh->Nfaces);
      sumFaceAverages_y = platform->linAlg->sum(mesh->Nelements * mesh->Nfaces,
                                                o_centroid,
                                                platform->comm.mpiComm(),
                                                1 * mesh->Nelements * mesh->Nfaces);
      sumFaceAverages_z = platform->linAlg->sum(mesh->Nelements * mesh->Nfaces,
                                                o_centroid,
                                                platform->comm.mpiComm(),
                                                2 * mesh->Nelements * mesh->Nfaces);

      const dfloat centroidTo_x = sumFaceAverages_x / NfacesContrib;
      const dfloat centroidTo_y = sumFaceAverages_y / NfacesContrib;
      const dfloat centroidTo_z = sumFaceAverages_z / NfacesContrib;

      lengthScale =
          distance(centroidFrom_x, centroidTo_x, centroidFrom_y, centroidTo_y, centroidFrom_z, centroidTo_z);

      computeDirection(centroidFrom_x,
                       centroidTo_x,
                       centroidFrom_y,
                       centroidTo_y,
                       centroidFrom_z,
                       centroidTo_z,
                       flowDirection);
    }
  }


  auto compute = [&]() {
    bool compute = false;
    const auto delta = platform->linAlg->maxRelativeError(mesh->Nlocal,
                                                          fluid->o_prop.size() / fluid->fieldOffset,
                                                          fluid->fieldOffset,
                                                          0,
                                                          o_prevProp,
                                                          fluid->o_prop,
                                                          platform->comm.mpiComm());

    if (delta > 10 * std::numeric_limits<dfloat>::epsilon()) {
      o_prevProp.copyFrom(fluid->o_prop);
      compute = true;
    }

    compute |= platform->options.compareArgs("MOVING MESH", "TRUE");
    compute |= tstep <= std::max(fluid->o_coeffEXT.size(), fluid->o_coeffBDF.size());
    compute |= abs(time - prevTime) > 1e-10;

    static dfloat prevFlowRate = 0;
    if (std::abs(flowRate - prevFlowRate) > 10 * std::numeric_limits<dfloat>::epsilon()) {
      compute |= true;
      prevFlowRate = flowRate;
    }

    return compute;
  }();

  prevTime = time;
  return compute;
}

void flowRate_t::adjust()
{
  if (platform->options.compareArgs(upperCase(fluid->velocityName) + " SOLVER", "NONE")) {
    return;
  }

  auto& mesh = fluid->mesh;

  rescaleFactor = [&]() {
    auto o_currentFlowRate = platform->deviceMemoryPool.reserve<dfloat>(fluid->fieldOffset);
    launchKernel("nrs-computeFieldDotNormal",
                 mesh->Nlocal,
                 fluid->fieldOffset,
                 flowDirection[0],
                 flowDirection[1],
                 flowDirection[2],
                 fluid->o_U,
                 o_currentFlowRate);

    flops += 5 * mesh->Nlocal;

    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_currentFlowRate);
    currentFlowRate =
        platform->linAlg->sum(mesh->Nlocal, o_currentFlowRate, platform->comm.mpiComm()) / lengthScale;

    const auto targetRate = platform->options.compareArgs("CONSTANT FLOW RATE TYPE", "VOLUMETRIC")
                                ? flowRate
                                : flowRate * mesh->volume / lengthScale;

    return (targetRate - currentFlowRate) / baseFlowRate;
  }();

  // superimpose
  platform->linAlg
      ->axpbyMany(mesh->Nlocal, mesh->dim, fluid->fieldOffset, rescaleFactor, o_Uc, 1.0, fluid->o_U);
  platform->linAlg->axpby(mesh->Nlocal, rescaleFactor, o_Pc, 1.0, fluid->o_P);
  o_Pc.free();

  // just for diagnostics
  postCorrectionFlowRate = [&]() {
    auto o_currentFlowRate = platform->deviceMemoryPool.reserve<dfloat>(fluid->fieldOffset);
    launchKernel("nrs-computeFieldDotNormal",
                 mesh->Nlocal,
                 fluid->fieldOffset,
                 flowDirection[0],
                 flowDirection[1],
                 flowDirection[2],
                 fluid->o_U,
                 o_currentFlowRate);

    flops += 5 * mesh->Nlocal;

    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_currentFlowRate);
    return platform->linAlg->sum(mesh->Nlocal, o_currentFlowRate, platform->comm.mpiComm()) / lengthScale;
  }();

  platform->flopCounter->add("ConstantFlowRate::adjust", flops);
}

dfloat flowRate_t::scaleFactor() const
{
  return rescaleFactor;
}
