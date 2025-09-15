#include "platform.hpp"
#include "linAlg.hpp"
#include "nrs.hpp"
#include "udf.hpp"
#include "alignment.hpp"
#include "bdryBase.hpp"

namespace
{

double flops = 0;
dfloat rescaleFactor = 0;

occa::memory o_Uc;
occa::memory o_Pc;
occa::memory o_prevProp;

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

bool checkIfRecomputeDirection(nrs_t *nrs, int tstep)
{
  return platform->options.compareArgs("MOVING MESH", "TRUE") || tstep < 2;
}

} // namespace

void nrs_t::computeHomogenousStokesSolution(double time)
{
  auto &mesh = fluid->mesh;
  const auto fieldOffset = fluid->fieldOffset;

  double flops = 0.0;

  auto o_lambda0 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->adyz(mesh->Nlocal, 1.0, fluid->o_rho, o_lambda0);

  auto o_Prhs = [&]() {
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

    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
    launchKernel("nrs-computeFieldDotNormal",
                 mesh->Nlocal,
                 fieldOffset,
                 flowDirection[0],
                 flowDirection[1],
                 flowDirection[2],
                 o_gradPCoeff,
                 o_rhs);

    flops += 5 * mesh->Nlocal;
    platform->timer.toc(fluid->pressureName + " rhs");
    return o_rhs;
  }();

  platform->timer.tic(fluid->pressureName + "Solve");
  fluid->ellipticSolverP->solve(o_lambda0, o_NULL, o_Prhs, o_Pc);
  platform->timer.toc(fluid->pressureName + "Solve");
  o_Prhs.free();
  o_lambda0.free();

  auto o_RhsVel = [&]() {
    platform->timer.tic(fluid->velocityName + " rhs");

    occa::memory o_rhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * fieldOffset);

    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_Pc,
                 o_rhs);

    double flopsGrad = 6 * mesh->Np * mesh->Nq + 18 * mesh->Np;
    flopsGrad *= static_cast<double>(mesh->Nelements);
    flops += flopsGrad;

    platform->linAlg->scaleMany(mesh->Nlocal, mesh->dim, fieldOffset, -1.0, o_rhs);

    occa::memory o_JwF = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * fieldOffset);
    o_JwF.copyFrom(mesh->o_LMM, mesh->Nlocal, 0 * fieldOffset, 0);
    o_JwF.copyFrom(mesh->o_LMM, mesh->Nlocal, 1 * fieldOffset, 0);
    o_JwF.copyFrom(mesh->o_LMM, mesh->Nlocal, 2 * fieldOffset, 0);

    for (int dim = 0; dim < mesh->dim; ++dim) {
      const dlong offset = dim * fieldOffset;
      const dfloat n_dim = flowDirection[dim];
      platform->linAlg->axpby(mesh->Nlocal, n_dim, o_JwF, 1.0, o_rhs, offset, offset);
    }
    platform->timer.toc(fluid->velocityName + " rhs");
    return o_rhs;
  }();

  platform->timer.tic(fluid->velocityName + "Solve");

  o_lambda0 = fluid->o_mue;
  auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->axpby(mesh->Nlocal, g0 / dt[0], fluid->o_rho, 0.0, o_lambda1);

  if (fluid->ellipticSolver.size() == 1) {
    fluid->ellipticSolver[0]->solve(o_lambda0, o_lambda1, o_RhsVel, o_Uc);
  } else {
    occa::memory o_Ucx = o_Uc + 0 * fieldOffset;
    occa::memory o_Ucy = o_Uc + 1 * fieldOffset;
    occa::memory o_Ucz = o_Uc + 2 * fieldOffset;
    fluid->ellipticSolver[0]->solve(o_lambda0, o_lambda1, o_RhsVel.slice(0 * fieldOffset), o_Ucx);
    fluid->ellipticSolver[1]->solve(o_lambda0, o_lambda1, o_RhsVel.slice(1 * fieldOffset), o_Ucy);
    fluid->ellipticSolver[2]->solve(o_lambda0, o_lambda1, o_RhsVel.slice(2 * fieldOffset), o_Ucz);
  }
  platform->timer.toc(fluid->velocityName + "Solve");

  platform->flopCounter->add("ConstantFlowRate::compute", flops);
}

void nrs_t::computeBaseFlowRate(double time, int tstep)
{
  if (platform->verbose() && platform->comm.mpiRank() == 0) {
    printf("computing base flow rate (dir: %g, %g, %g)\n",
           flowDirection[0],
           flowDirection[1],
           flowDirection[2]);
  }

  auto getSolverData = [](elliptic *solver) {
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
  };

  auto &uvwSolver = fluid->ellipticSolver.at(0);
  const auto [NiterUVW, res00NormUVW, res0NormUVW, resNormUVW] = getSolverData(uvwSolver);

  auto uSolver = (fluid->ellipticSolver.size() == 1) ? fluid->ellipticSolver.at(0) : nullptr;
  auto vSolver = (fluid->ellipticSolver.size() == 2) ? fluid->ellipticSolver.at(1) : nullptr;
  auto wSolver = (fluid->ellipticSolver.size() == 3) ? fluid->ellipticSolver.at(2) : nullptr;
  const auto [NiterU, res00NormU, res0NormU, resNormU] = getSolverData(uSolver);
  const auto [NiterV, res00NormV, res0NormV, resNormV] = getSolverData(vSolver);
  const auto [NiterW, res00NormW, res0NormW, resNormW] = getSolverData(wSolver);

  auto &pSolver = fluid->ellipticSolverP;
  const auto [NiterP, res00NormP, res0NormP, resNormP] = getSolverData(pSolver);

  computeHomogenousStokesSolution(time);

  // restore norms + update iteration count
  auto setSolverData = [](elliptic *solver, int Niter, dfloat res00Norm, dfloat res0Norm, dfloat resNorm) {
    solver->Niter(solver->Niter() + Niter);
    solver->initialResidual(res00Norm);
    solver->initialGuessResidual(res0Norm);
    solver->finalResidual(resNorm);
  };

  if (fluid->ellipticSolver.size() == 1) {
    setSolverData(uvwSolver, NiterUVW, res00NormUVW, res0NormUVW, resNormUVW);
  } else {
    setSolverData(uSolver, NiterU, res00NormU, res0NormU, resNormU);
    setSolverData(vSolver, NiterV, res00NormV, res0NormV, resNormV);
    setSolverData(wSolver, NiterW, res00NormW, res0NormW, resNormW);
  }
  setSolverData(pSolver, NiterP, res00NormP, res0NormP, resNormP);

  auto &mesh = fluid->mesh;

  occa::memory o_baseFlowRate = platform->deviceMemoryPool.reserve<dfloat>(fluid->fieldOffset);
  launchKernel("nrs-computeFieldDotNormal",
               mesh->Nlocal,
               fluid->fieldOffset,
               flowDirection[0],
               flowDirection[1],
               flowDirection[2],
               o_Uc,
               o_baseFlowRate);
  flops += 5 * mesh->Nlocal;

  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_baseFlowRate);
  baseFlowRate = platform->linAlg->sum(mesh->Nlocal, o_baseFlowRate, platform->comm.mpiComm()) / lengthScale;
}

void nrs_t::flowRatePrintInfo(int tstep, bool verboseInfo)
{
  auto mesh = this->meshV;

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

void nrs_t::adjustFlowRate(int tstep, double time)
{
  flops = 0.0;

  platform->options.getArgs("FLOW RATE", flowRate);

  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

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

  if (!o_Uc.isInitialized()) {
    o_Uc = platform->device.malloc<dfloat>(mesh->dim * fluid->fieldOffset);
    o_Pc = platform->device.malloc<dfloat>(mesh->Nlocal);
    o_prevProp = platform->device.malloc<dfloat>(fluid->o_prop.size());
    o_prevProp.copyFrom(fluid->o_prop);
  }

  const bool recomputeDirection = checkIfRecomputeDirection(this, tstep);

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

      occa::memory o_centroid =
          platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * mesh->Nelements * mesh->Nfaces);
      platform->linAlg->fill(mesh->Nelements * mesh->Nfaces * 3, 0.0, o_centroid);

      occa::memory o_counts = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements * mesh->Nfaces);
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
    compute |= tstep <= std::max(o_coeffEXT.size(), o_coeffBDF.size());
    compute |= abs(dt[0] - dt[1]) > 1e-10;

    static dfloat prevFlowRate = 0;
    if (std::abs(flowRate - prevFlowRate) > 10 * std::numeric_limits<dfloat>::epsilon()) {
      compute |= true;
      prevFlowRate = flowRate;
    }

    return compute;
  }();

  // Stokes solution
  if (compute) {
    computeBaseFlowRate(time, tstep);
  }

  rescaleFactor = [&]() {
    occa::memory o_currentFlowRate = platform->deviceMemoryPool.reserve<dfloat>(fluid->fieldOffset);
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
      ->axpbyMany(mesh->Nlocal, mesh->dim, fluid->fieldOffset, rescaleFactor, o_Uc, 1.0, this->fluid->o_U);
  platform->linAlg->axpby(mesh->Nlocal, rescaleFactor, o_Pc, 1.0, this->fluid->o_P);

  // diagnostics
  postCorrectionFlowRate = [&]() {
    occa::memory o_currentFlowRate = platform->deviceMemoryPool.reserve<dfloat>(fluid->fieldOffset);
    launchKernel("nrs-computeFieldDotNormal",
                 mesh->Nlocal,
                 fluid->fieldOffset,
                 flowDirection[0],
                 flowDirection[1],
                 flowDirection[2],
                 this->fluid->o_U,
                 o_currentFlowRate);

    flops += 5 * mesh->Nlocal;

    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_currentFlowRate);
    return platform->linAlg->sum(mesh->Nlocal, o_currentFlowRate, platform->comm.mpiComm()) / lengthScale;
  }();

  platform->flopCounter->add("ConstantFlowRate::adjust", flops);
}

dfloat nrs_t::flowRateScaleFactor()
{
  return rescaleFactor;
}
