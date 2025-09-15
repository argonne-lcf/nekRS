#include "nrs.hpp"
#include "gjp.hpp"
#include "lowPassFilter.hpp"
#include "advectionSubCycling.hpp"
#include "registerKernels.hpp"
#include "nekInterfaceAdapter.hpp"

fluidSolver_t::fluidSolver_t(const fluidSolverCfg_t &cfg, const std::unique_ptr<geomSolver_t> &_geom)
    : geom(_geom)
{
  name = cfg.name;

  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ "
              << "SETUP " + upperCase(name) << " ================" << std::endl;
  }

  mesh = cfg.mesh;
  fieldOffset = cfg.fieldOffset;
  fieldOffsetSum = mesh->dim * fieldOffset;
  cubatureOffset = cfg.cubatureOffset;

  const char nullChar[] = {'\0'};

  velocityName = cfg.velocityName;
  o_velocityName = platform->device.malloc<char>(velocityName.size() + 1);
  o_velocityName.copyFrom(velocityName.data());
  o_velocityName.copyFrom(nullChar, 1, velocityName.size());

  pressureName = cfg.pressureName;
  o_pressureName = platform->device.malloc<char>(pressureName.size() + 1);
  o_pressureName.copyFrom(pressureName.data());
  o_pressureName.copyFrom(nullChar, 1, pressureName.size());

  g0 = cfg.g0;
  dt = cfg.dt;
  o_coeffEXT = cfg.o_coeffEXT;
  o_coeffBDF = cfg.o_coeffBDF;

  nameToIndex["x"] = 0;
  nameToIndex["y"] = 1;
  nameToIndex["z"] = 2;

  Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  o_U = platform->device.malloc<dfloat>(fieldOffsetSum * std::max(o_coeffBDF.size(), o_coeffEXT.size()));

  if (!platform->options.compareArgs(upperCase(velocityName) + " EXTRAPOLATION", "FALSE")) {
    o_Ue = platform->device.malloc<dfloat>(fieldOffsetSum);
  } else {
    o_Ue = o_U.slice(0, fieldOffsetSum);
  }

  if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
    int nEXT = 2;
    const auto key = upperCase(pressureName) + " EXT ORDER";
    if (platform->options.getArgs(key).empty()) {
      platform->options.setArgs(key, std::to_string(nEXT));
    }
    platform->options.getArgs(key, nEXT);
    o_Pe = platform->device.malloc<dfloat>(fieldOffset);
    o_coeffEXTP = platform->device.malloc<dfloat>(std::min(nEXT, static_cast<int>(o_coeffBDF.size())));
  }
  o_P = platform->device.malloc<dfloat>(fieldOffset * std::max(static_cast<int>(o_coeffEXTP.size()), 1));

  o_div = platform->device.malloc<dfloat>(fieldOffset);

  o_JwF = platform->device.malloc<dfloat>(fieldOffsetSum);
  o_EXT = platform->device.malloc<dfloat>(o_coeffEXT.size() * fieldOffsetSum);
  o_ADV = platform->device.malloc<dfloat>(o_coeffEXT.size() * fieldOffsetSum);

  {
    const dlong Nstates = Nsubsteps ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
    o_relUrst = platform->device.malloc<dfloat>(Nstates * mesh->dim * cubatureOffset);
  }

  o_prop = [&]() {
    dfloat mue = 1;
    dfloat rho = 1;
    platform->options.getArgs(upperCase(name) + " VISCOSITY", mue);
    platform->options.getArgs(upperCase(name) + " DENSITY", rho);
    auto o_u = platform->device.malloc<dfloat>(2 * fieldOffset);
    auto o_u0 = o_u + 0 * fieldOffset;
    auto o_u1 = o_u + 1 * fieldOffset;
    platform->linAlg->fill(mesh->Nlocal, mue, o_u0);
    platform->linAlg->fill(mesh->Nlocal, rho, o_u1);
    return o_u;
  }();
  o_mue = o_prop.slice(0 * fieldOffset, mesh->Nlocal);
  o_rho = o_prop.slice(1 * fieldOffset, mesh->Nlocal);

  if (!platform->options.compareArgs(upperCase(velocityName) + " SOLVER", "NONE")) {
    platform->app->bc->printBcTypeMapping(velocityName);

    if (platform->options.compareArgs(upperCase(velocityName) + " REGULARIZATION METHOD", "HPFRT")) {
      int nModes = -1;
      platform->options.getArgs(upperCase(velocityName) + " HPFRT MODES", nModes);
      o_filterRT = lowPassFilterSetup(mesh, nModes);
    }

    o_EToB = [&]() {
      auto u = mesh->createEToB([&](int bID) -> int { return platform->app->bc->typeId(bID, velocityName); });
      auto o_u = platform->device.malloc<int>(u.size());
      o_u.copyFrom(u.data());
      return o_u;
    }();

    auto verifyBC = [&]() {
      nekrsCheck(mesh->Nbid != platform->app->bc->size(velocityName),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 velocityName.c_str(),
                 platform->app->bc->size(velocityName),
                 mesh->Nbid);

      platform->app->bc->checkAlignment(mesh);
    };

    verifyBC();
  }
}

void fluidSolver_t::solvePressure(double time, int stage)
{
  if (!ellipticSolverP) {
    return;
  }

  platform->timer.tic(pressureName + "Solve");

  const auto g0idt = *g0 / dt[0];

  double flopCount = 0.0;
  platform->timer.tic(pressureName + " rhs");

  auto o_lambda0 = [&](bool variable = true) {
    auto o_lambda = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE") && !variable) {
      platform->linAlg->fill(mesh->Nlocal, 1 / rho0, o_lambda);
    } else {
      platform->linAlg->adyz(mesh->Nlocal, 1.0, o_rho, o_lambda);
    }
    return o_lambda;
  };

  const auto o_rhoSplitTerm = [&]() {
    occa::memory o_del;
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
      // 1/rho - 1/rho0
      auto o_lambda = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      platform->linAlg->adyz(mesh->Nlocal, 1.0, o_rho, o_lambda);
      platform->linAlg->add(mesh->Nlocal, -1 / rho0, o_lambda);

      o_del = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

      const auto valSave = ellipticSolverP->options().getArgs("ELLIPTIC COEFF FIELD");
      ellipticSolverP->options().setArgs("ELLIPTIC COEFF FIELD", "TRUE");
      ellipticSolverP->Ax(o_lambda, o_NULL, o_Pe, o_del);
      ellipticSolverP->options().setArgs("ELLIPTIC COEFF FIELD", valSave);
    }
    return o_del;
  }();

  const auto o_stressTerm = [&]() {
    auto o_curl = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    launchKernel("core-curlHex3D", mesh->Nelements, 1, mesh->o_vgeo, mesh->o_D, fieldOffset, o_Ue, o_curl);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

    oogs::startFinish(o_curl, mesh->dim, fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);

    platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, mesh->o_invLMM, o_curl);
    flopCount += mesh->Nlocal;

    auto o_stressTerm = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
    launchKernel("core-curlHex3D",
                 mesh->Nelements,
                 1,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_curl,
                 o_stressTerm);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

    if (platform->options.compareArgs(upperCase(name) + " STRESSFORMULATION", "TRUE")) {
      launchKernel("fluidSolver_t::pressureStressHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_D,
                   fieldOffset,
                   o_mue,
                   o_Ue,
                   o_div,
                   o_stressTerm);
      flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Nq * mesh->Np + 100 * mesh->Np);
    }
    return o_stressTerm;
  }();

  auto o_rhs = [&]() {
    auto o_gradDiv = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_div,
                 o_gradDiv);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

#if 1
    launchKernel("fluidSolver_t::pressureRhsHex3D",
                 mesh->Nlocal,
                 fieldOffset,
                 o_mue,
                 o_lambda0(),
                 o_JwF,
                 o_stressTerm,
                 o_gradDiv,
                 o_rhs);
    flopCount += 12 * static_cast<double>(mesh->Nlocal);
#else
    o_rhs.copyFrom(this->o_JwF);
#endif

    oogs::startFinish(o_rhs, mesh->dim, fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);
    platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, mesh->o_invLMM, o_rhs);

    return o_rhs;
  }();

  auto o_pRhs = [&]() {
    auto o_pRhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);

    launchKernel("core-wDivergenceVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_rhs,
                 o_pRhs);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    launchKernel("fluidSolver_t::pressureAddQtl", mesh->Nlocal, mesh->o_LMM, g0idt, o_div, o_pRhs);
    flopCount += 3 * mesh->Nlocal;

    launchKernel("fluidSolver_t::divergenceSurfaceHex3D",
                 mesh->Nelements,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 o_EToB,
                 g0idt,
                 fieldOffset,
                 o_rhs,
                 o_U,
                 o_pRhs);
    flopCount += 25 * static_cast<double>(mesh->Nelements) * mesh->Nq * mesh->Nq;

    if (o_rhoSplitTerm.isInitialized()) {
      platform->linAlg->axpby(mesh->Nlocal, -1.0, o_rhoSplitTerm, 1.0, o_pRhs);
    }

    return o_pRhs;
  }();

  platform->timer.toc(pressureName + " rhs");
  platform->flopCounter->add(pressureName + " rhs", flopCount);

  ellipticSolverP->solve(o_lambda0(false), o_NULL, o_pRhs, o_P.slice(0, mesh->Nlocal));

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 1,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_P,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("p norm: %.15e\n", debugNorm);
    }
  }

  platform->timer.toc(pressureName + "Solve");
}

void fluidSolver_t::solveVelocity(double time, int stage)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  platform->timer.tic(velocityName + "Solve");

  const auto g0idt = *g0 / dt[0];

  double flopCount = 0.0;
  platform->timer.tic(velocityName + " rhs");

  const auto o_gradMueDiv = [&]() {
    dfloat scale = 1. / 3;
    if (platform->options.compareArgs(upperCase(name) + " STRESSFORMULATION", "TRUE")) {
      scale = -2 * scale;
    }

    auto o_mueDiv = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
    platform->linAlg->axmyz(mesh->Nlocal, scale, o_mue, o_div, o_mueDiv);

    auto o_gradMueDiv = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_mueDiv,
                 o_gradMueDiv);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    return o_gradMueDiv;
  }();

  const auto o_rhoSplitTerm = [&]() {
    occa::memory o_del;
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
      o_del = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
      auto o_delta = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

      platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_P, -1.0, o_Pe, o_delta);
      launchKernel("core-gradientVolumeHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_D,
                   fieldOffset,
                   o_delta,
                   o_del);

      flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

      // o_del * rho / rho0
      platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1 / rho0, o_rho, o_del);
    }
    return o_del;
  }();

  const auto o_gradP = [&]() {
    occa::memory o_gradP = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    auto o_P = this->o_P;
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
      o_P = o_Pe;
    }

#if 1
    // use weak formulation to allow for non-zero Dirichlet pressure BC when using stressFormulation
    launchKernel("core-wGradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_P,
                 o_gradP);
#else
    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_P,
                 o_gradP);
    platform->linAlg->scale(o_gradP.size(), -1.0, o_gradP);
#endif
    flopCount += static_cast<double>(mesh->Nelements) * 18 * (mesh->Np * mesh->Nq + mesh->Np);

    return o_gradP;
  }();

  const auto o_rhs = [&]() {
    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    // o_rho * o_JwF + o_gradP + o_gradMueDiv
    launchKernel("fluidSolver_t::velocityRhsHex3D",
                 mesh->Nlocal,
                 fieldOffset,
                 o_rho,
                 o_JwF,
                 o_gradMueDiv,
                 o_gradP,
                 o_rhs);
    flopCount += 9 * mesh->Nlocal;

    launchKernel("fluidSolver_t::velocityNeumannBCHex3D",
                 o_velocityName,
                 o_pressureName,
                 mesh->Nelements,
                 fieldOffset,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 o_EToB,
                 time,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 o_rho,
                 o_mue,
                 platform->app->bc->o_usrwrk,
                 o_Ue,
                 o_rhs);
    flopCount += static_cast<double>(mesh->Nelements) * (3 * mesh->Np + 36 * mesh->Nq * mesh->Nq);

    if (o_rhoSplitTerm.isInitialized()) {
      platform->linAlg->axpbyMany(mesh->Nlocal, mesh->dim, fieldOffset, -1.0, o_rhoSplitTerm, 1.0, o_rhs);
    }

    return o_rhs;
  }();

  platform->timer.toc(velocityName + " rhs");
  platform->flopCounter->add(velocityName + " rhs", flopCount);

  const auto o_lambda0 = o_mue;
  const auto o_lambda1 = [&]() {
    auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    if (userImplicitLinearTerm) {
      auto o_implicitLT = userImplicitLinearTerm(time);
      platform->linAlg->axpbyz(mesh->Nlocal, g0idt, o_rho, 1.0, o_implicitLT, o_lambda1);
    } else {
      platform->linAlg->axpby(mesh->Nlocal, g0idt, o_rho, 0.0, o_lambda1);
    }
    return o_lambda1;
  }();

  if (platform->options.compareArgs(upperCase(velocityName) + " INITIAL GUESS", "EXTRAPOLATION") &&
      stage == 1) {
    o_U.copyFrom(o_Ue, fieldOffsetSum);
  }

  if (ellipticSolver.at(0)->Nfields() > 1) {
    ellipticSolver.at(0)->solve(o_lambda0, o_lambda1, o_rhs, o_U.slice(0, fieldOffsetSum));
  } else {
    const auto o_rhsX = o_rhs.slice(0 * fieldOffset, mesh->Nlocal);
    const auto o_rhsY = o_rhs.slice(1 * fieldOffset, mesh->Nlocal);
    const auto o_rhsZ = o_rhs.slice(2 * fieldOffset, mesh->Nlocal);
    ellipticSolver.at(0)->solve(o_lambda0, o_lambda1, o_rhsX, o_U.slice(0 * fieldOffset, mesh->Nlocal));
    ellipticSolver.at(1)->solve(o_lambda0, o_lambda1, o_rhsY, o_U.slice(1 * fieldOffset, mesh->Nlocal));
    ellipticSolver.at(2)->solve(o_lambda0, o_lambda1, o_rhsZ, o_U.slice(2 * fieldOffset, mesh->Nlocal));
  }

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_U,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("U norm: %.15e\n", debugNorm);
    }
  }

  platform->timer.toc(velocityName + "Solve");
}

void fluidSolver_t::setupEllipticSolver()
{
  if (platform->options.compareArgs(upperCase(velocityName) + " SOLVER", "NONE")) {
    return;
  }

  auto o_lambda0 = o_mue;
  auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->axpby(mesh->Nlocal, *g0 / dt[0], o_rho, 0.0, o_lambda1);

  const auto unalignedBoundary = platform->app->bc->hasUnalignedMixed(velocityName);

  if (platform->options.compareArgs(upperCase(velocityName) + " SOLVER", "BLOCK")) {
    if (platform->options.compareArgs(upperCase(name) + " STRESSFORMULATION", "TRUE")) {
      platform->options.setArgs(upperCase(velocityName) + " STRESSFORMULATION", "TRUE");
    }

    ellipticSolver.push_back(new elliptic(velocityName, mesh, fieldOffset, o_lambda0, o_lambda1));

    if (unalignedBoundary) {
      o_zeroNormalMask = mesh->createZeroNormalMask(fieldOffset, ellipticSolver[0]->o_EToB());

      auto f = [this](dlong Nelements, const occa::memory &o_elementList, occa::memory &o_x) {
        launchKernel("mesh-applyZeroNormalMask",
                     Nelements,
                     this->fieldOffset,
                     o_elementList,
                     this->mesh->o_sgeo,
                     this->o_zeroNormalMask,
                     this->mesh->o_vmapM,
                     this->ellipticSolver.at(0)->o_EToB(),
                     o_x);
      };
      ellipticSolver.at(0)->applyZeroNormalMask(f);
    }
  } else {
    nekrsCheck(unalignedBoundary,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "unaligned mixed boundary conditions require using block solver for %s\n",
               velocityName.c_str());
    auto EToBx = mesh->createEToB(
        [&](int bID) -> int { return platform->app->bc->typeElliptic(bID, velocityName, "x"); });

    auto EToBy = mesh->createEToB(
        [&](int bID) -> int { return platform->app->bc->typeElliptic(bID, velocityName, "y"); });

    auto EToBz = mesh->createEToB(
        [&](int bID) -> int { return platform->app->bc->typeElliptic(bID, velocityName, "z"); });

    ellipticSolver.push_back(new elliptic(velocityName, mesh, fieldOffset, EToBx, o_lambda0, o_lambda1));
    ellipticSolver.push_back(new elliptic(velocityName, mesh, fieldOffset, EToBy, o_lambda0, o_lambda1));
    ellipticSolver.push_back(new elliptic(velocityName, mesh, fieldOffset, EToBz, o_lambda0, o_lambda1));
  }

  {
    const auto o_lambda0 = [&]() {
      auto o_lambda = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
        rho0 = platform->linAlg->min(mesh->Nlocal, o_rho, platform->comm.mpiComm());
        platform->linAlg->fill(mesh->Nlocal, 1 / rho0, o_lambda);
      } else {
        platform->linAlg->adyz(mesh->Nlocal, 1.0, o_rho, o_lambda);
      }
      return o_lambda;
    }();

    // derive from velocity
    auto EToBP = mesh->createEToB([&](int bID) -> int {
      auto bcType = platform->app->bc->typeId(bID, velocityName);
      if (bID < 1) {
        return ellipticBcType::NO_OP;
      }
      return platform->app->bc->isOutflow(bcType) ? ellipticBcType::DIRICHLET : ellipticBcType::NEUMANN;
    });

    ellipticSolverP = new elliptic(pressureName, mesh, fieldOffset, EToBP, o_lambda0, o_NULL);
  }
}

void fluidSolver_t::applyDirichlet(double time)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  if (platform->app->bc->hasUnalignedMixed(velocityName)) {
    launchKernel("mesh-applyZeroNormalMask",
                 mesh->Nelements,
                 fieldOffset,
                 mesh->o_elementList,
                 mesh->o_sgeo,
                 o_zeroNormalMask,
                 mesh->o_vmapM,
                 ellipticSolver.at(0)->o_EToB(),
                 o_U);
    launchKernel("mesh-applyZeroNormalMask",
                 mesh->Nelements,
                 fieldOffset,
                 mesh->o_elementList,
                 mesh->o_sgeo,
                 o_zeroNormalMask,
                 mesh->o_vmapM,
                 ellipticSolver.at(0)->o_EToB(),
                 o_Ue);
  }

  // lower than any other possible Dirichlet value
  static constexpr dfloat TINY = -1e30;
  occa::memory o_tmp = platform->deviceMemoryPool.reserve<dfloat>((mesh->dim + 1) * fieldOffset);
  platform->linAlg->fill(o_tmp.size(), TINY, o_tmp);

  auto &neknek = platform->app->neknek;

  for (int sweep = 0; sweep < 2; sweep++) {
    launchKernel("fluidSolver_t::pressureDirichletBCHex3D",
                 o_pressureName,
                 mesh->Nelements,
                 time,
                 fieldOffset,
                 mesh->o_sgeo,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 o_EToB,
                 o_rho,
                 o_mue,
                 platform->app->bc->o_usrwrk,
                 o_Ue,
                 o_tmp);

    launchKernel("fluidSolver_t::velocityDirichletBCHex3D",
                 o_velocityName,
                 mesh->Nelements,
                 fieldOffset,
                 time,
                 mesh->o_sgeo,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 o_EToB,
                 o_rho,
                 o_mue,
                 neknek ? neknek->intValOffset() : 0,
                 neknek ? neknek->o_pointMap() : o_NULL,
                 neknek ? neknek->getField(velocityName).o_intVal : o_NULL,
                 platform->app->bc->o_usrwrk,
                 o_U,
                 o_tmp.slice(fieldOffset));

    oogs::startFinish(o_tmp,
                      1 + mesh->dim,
                      fieldOffset,
                      ogsDfloat,
                      (sweep == 0) ? ogsMax : ogsMin,
                      mesh->oogs3);
  }

  if (ellipticSolverP->Nmasked()) {
    auto o_dirichletValues = o_tmp.slice(0, fieldOffset);
    launchKernel("core-maskCopy",
                 ellipticSolverP->Nmasked(),
                 0,
                 0,
                 ellipticSolverP->o_maskIds(),
                 o_dirichletValues,
                 o_P);
  }

  int cnt = 0;
  for (auto &solver : ellipticSolver) {
    if (solver->Nmasked()) {
      const auto offset = cnt * fieldOffset;
      auto o_dirichletValues = o_tmp.slice(fieldOffset, fieldOffsetSum);
      launchKernel("core-maskCopy2",
                   solver->Nmasked(),
                   offset,
                   offset,
                   solver->o_maskIds(),
                   o_dirichletValues,
                   o_U,
                   o_Ue);
    }
    cnt++;
  }
}

void fluidSolver_t::makeForcing()
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  launchKernel("fluidSolver_t::sumMakef",
               mesh->Nlocal,
               mesh->o_Jw,
               1 / dt[0],
               o_coeffEXT,
               o_coeffBDF,
               fieldOffset,
               fieldOffsetSum,
               mesh->fieldOffset, /* o_Jw offset */
               o_rho,
               o_U,
               o_ADV,
               o_EXT,
               o_JwF);

  dfloat flops = Nsubsteps ? (6 + 6 * o_coeffEXT.size()) : (6 * o_coeffEXT.size() + 12 * o_coeffBDF.size());
  platform->flopCounter->add(velocityName + " sumMakef", flops * static_cast<double>(mesh->Nlocal));

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_JwF,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s JwF norm: %.15e\n", name.c_str(), debugNorm);
    }
  }

  for (int s = o_coeffEXT.size(); s > 1; s--) {
    o_EXT.copyFrom(o_EXT, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    if (o_ADV.isInitialized()) {
      o_ADV.copyFrom(o_ADV, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    }
  }
}

void fluidSolver_t::makeAdvection(double time, int tstep)
{
  if (Nsubsteps) {
    advectionSubcycling(std::min(tstep, static_cast<int>(o_coeffEXT.size())), time);
  } else {
    auto &o_Uconv = o_relUrst;

    double flopCount = 0.0;

    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      launchKernel("core-strongAdvectionCubatureVolumeHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_cubDiffInterpT,
                   mesh->o_cubInterpT,
                   mesh->o_cubProjectT,
                   fieldOffset,
                   cubatureOffset,
                   o_U,
                   o_Uconv,
                   o_ADV);

      flopCount += 4. * mesh->Nq *
                   (mesh->cubNp + mesh->cubNq * mesh->cubNq * mesh->Nq +
                    mesh->cubNq * mesh->Nq * mesh->Nq); // interpolation
      flopCount += 6. * mesh->cubNp * mesh->cubNq;      // apply Dcub
      flopCount += 5 * mesh->cubNp;                     // compute advection term on cubature mesh
      flopCount += mesh->Np;                            // weight by inv. mass matrix
    } else {
      launchKernel("core-strongAdvectionVolumeHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_D,
                   fieldOffset,
                   o_U,
                   o_Uconv,
                   o_ADV);

      flopCount += 8 * (mesh->Np * mesh->Nq + mesh->Np);
    }

    flopCount *= mesh->Nelements;
    flopCount *= mesh->dim;
    platform->flopCounter->add(velocityName + " advection", flopCount);
  }
}

void fluidSolver_t::makeExplicit(double time, int tstep)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  if (platform->options.compareArgs(upperCase(velocityName) + " REGULARIZATION METHOD", "HPFRT")) {
    dfloat strength = NAN;
    platform->options.getArgs(upperCase(velocityName) + " HPFRT STRENGTH", strength);

    launchKernel("core-vectorFilterRTHex3D", mesh->Nelements, o_filterRT, strength, fieldOffset, o_U, o_EXT);

    double flops = 24 * mesh->Np * mesh->Nq + 3 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    platform->flopCounter->add(velocityName + " filterRT", flops);
  }

  if (platform->options.compareArgs(upperCase(velocityName) + " REGULARIZATION METHOD", "GJP")) {
    dfloat tauFactor;
    platform->options.getArgs(upperCase(velocityName) + " REGULARIZATION GJP SCALING COEFF", tauFactor);

    for (int i = 0; i < mesh->dim; i++) {
      auto o_EXTi = o_EXT.slice(i * fieldOffset, mesh->Nlocal);
      addGJP(mesh, o_EToB, fieldOffset, o_U, o_U.slice(i * fieldOffset, mesh->Nlocal), o_EXTi, tauFactor);
    }
  }

  if (geom && !Nsubsteps) {
    launchKernel("fluidSolver_t::advectMeshVelocityHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 geom->fieldOffset,
                 geom->o_U,
                 o_U,
                 o_EXT);
    double flops = 54 * mesh->Np * mesh->Nq + 63 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    platform->flopCounter->add(velocityName + " advectMeshVelocity", flops);
  }
}

void fluidSolver_t::saveSolutionState()
{
  if (!o_U0.isInitialized()) {
    o_U0 = platform->device.malloc<dfloat>(o_U.size());
    o_P0 = platform->device.malloc<dfloat>(o_P.size());
    o_EXT0 = platform->device.malloc<dfloat>(o_EXT.size());
    o_prop0 = platform->device.malloc<dfloat>(o_prop.size());
    o_relUrst0 = platform->device.malloc<dfloat>(o_relUrst.size());
  }

  o_U0.copyFrom(o_U);
  o_P0.copyFrom(o_P);
  o_EXT0.copyFrom(o_EXT);
  o_prop0.copyFrom(o_prop);
  o_relUrst0.copyFrom(o_relUrst);
}

void fluidSolver_t::restoreSolutionState()
{
  o_U.copyFrom(o_U0);
  o_P.copyFrom(o_P0);
  o_EXT.copyFrom(o_EXT0);
  o_prop.copyFrom(o_prop0);
  o_relUrst.copyFrom(o_relUrst0);
}

void fluidSolver_t::extrapolateSolution()
{
  if (!platform->options.compareArgs(upperCase(velocityName) + " EXTRAPOLATION", "FALSE")) {
    launchKernel("core-extrapolate",
                 mesh->Nlocal,
                 mesh->dim,
                 static_cast<int>(o_coeffEXT.size()),
                 fieldOffset,
                 o_coeffEXT,
                 o_U,
                 o_Ue);
  }

  if (o_coeffEXTP.size()) {
    launchKernel("core-extrapolate",
                 mesh->Nlocal,
                 1,
                 static_cast<int>(o_coeffEXTP.size()),
                 fieldOffset,
                 o_coeffEXTP,
                 o_P,
                 o_Pe);
  }
}

void fluidSolver_t::lagSolution()
{
  {
    const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
    for (int s = n; s > 1; s--) {
      o_U.copyFrom(o_U, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    }
  }

  {
    const auto n = o_coeffEXTP.size();
    for (int s = n; s > 1; s--) {
      o_P.copyFrom(o_P, fieldOffset, (s - 1) * fieldOffset, (s - 2) * fieldOffset);
    }
  }
}

void fluidSolver_t::advectionSubcycling(int nEXT, double time)
{
  const auto nFields = mesh->dim;
  const auto fieldOffsetSum = nFields * fieldOffset;

  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      kernel = platform->kernelRequests.load("core-subCycleStrongCubatureVolumeHex3D");
    } else {
      kernel = platform->kernelRequests.load("core-subCycleStrongVolumeHex3D");
    }
  }

  platform->linAlg->fill(o_JwF.size(), 0, o_JwF);

  advectionSubcyclingRK(mesh,
                        mesh,
                        time,
                        dt,
                        Nsubsteps,
                        o_coeffBDF,
                        nEXT,
                        nFields,
                        kernel,
                        mesh->oogs3,
                        mesh->fieldOffset,
                        fieldOffset,
                        cubatureOffset,
                        fieldOffsetSum,
                        (geom) ? geom->o_div : o_NULL,
                        o_relUrst,
                        o_U,
                        o_JwF);

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_JwF,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s advSub norm: %.15e\n", name.c_str(), debugNorm);
    }
  }
}

void registerFluidSolverKernels(occa::properties kernelInfoBC)
{
  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";
  const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  const std::string suffix = "Hex3D";
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/fluid/");
  std::string section = "fluidSolver_t::";

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;

  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int nBDF = 0;
  int nEXT = 0;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);

  int Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);
  if (Nsubsteps) {
    nEXT = nBDF;
  }

  std::string fileName, kernelName;

  occa::properties meshProps = kernelInfo;
  meshProps += meshKernelProperties(N);

  {
    occa::properties prop = kernelInfo;
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

  kernelName = "divergenceSurface" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "advectMeshVelocityHex3D";
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureRhs" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureStress" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureDirichletBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "velocityDirichletBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "velocityNeumannBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "velocityRhs" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureAddQtl";
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);
}

void fluidSolver_t::setTimeIntegrationCoeffs(int tstep)
{
  if (o_coeffEXTP.size()) {
    std::vector<dfloat> coeff(o_coeffEXTP.size());
    const auto extOrder = std::min(tstep, static_cast<int>(o_coeffEXTP.size()));
    nek::extCoeff(coeff.data(), dt, extOrder, extOrder);
    for (int i = coeff.size(); i > extOrder; i--) {
      coeff[i - 1] = 0;
    }
    o_coeffEXTP.copyFrom(coeff.data());
  }
}
