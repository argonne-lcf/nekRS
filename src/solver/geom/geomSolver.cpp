#include "nrs.hpp"
#include "registerKernels.hpp"
#include "nekInterfaceAdapter.hpp"

geomSolver_t::geomSolver_t(const geomSolverCfg_t &cfg)
{
  name = cfg.name;

  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ "
              << "SETUP " + upperCase(name) << " ================" << std::endl;
  }

  mesh = cfg.mesh;
  meshV = cfg.meshV;

  fieldOffset = cfg.fieldOffset;
  fieldOffsetSum = mesh->dim * fieldOffset;

  g0 = cfg.g0;
  dt = cfg.dt;
  o_coeffEXT = cfg.o_coeffEXT;

  deriveBCFromVelocity = cfg.deriveBCFromVelocity;

  nameToIndex["x"] = 0;
  nameToIndex["y"] = 1;
  nameToIndex["z"] = 2;

  int nAB;
  platform->options.getArgs(upperCase(name) + " INTEGRATION ORDER", nAB);
  o_coeffAB = platform->device.malloc<dfloat>(nAB);

  const auto n = std::max(o_coeffEXT.size(), o_coeffAB.size());
  o_U = platform->device.malloc<dfloat>(n * fieldOffsetSum);
  o_Ue = platform->device.malloc<dfloat>(o_U.size());

  o_prop = platform->device.malloc<dfloat>(mesh->Nlocal);
  platform->linAlg->fill(o_prop.size(), 1.0, o_prop);

  int Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);
  if (Nsubsteps) {
    o_div = platform->device.malloc<dfloat>(fieldOffset * n);
  }

  auto resize = [&](mesh_t *mesh, occa::memory &in) {
    auto o_new = platform->device.malloc<dfloat>(mesh->fieldOffset * n);
    o_new.copyFrom(in, in.size());

    in.free();
    return o_new;
  };

  mesh->o_Jw = resize(mesh, mesh->o_Jw);
  mesh->o_invAJw = resize(mesh, mesh->o_invAJw);

  if (!platform->options.compareArgs(upperCase(name) + " SOLVER", "NONE")) {
    platform->app->bc->printBcTypeMapping(name);

    auto EToB = mesh->createEToB([&](int bID) -> int { return platform->app->bc->typeId(bID, name); });
    o_EToB = platform->device.malloc<int>(EToB.size());
    o_EToB.copyFrom(EToB.data());

    auto verifyBC = [&]() {
      auto msh = (mesh != meshV) ? mesh : meshV;
      nekrsCheck(msh->Nbid != platform->app->bc->size(name),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 name.c_str(),
                 platform->app->bc->size(name),
                 msh->Nbid);

      platform->app->bc->checkAlignment(msh);
    };

    verifyBC();
  }
};

void geomSolver_t::integrate(bool lag)
{
  const int nAB = o_coeffAB.size();

  if (lag) {
    const auto n = std::max(o_coeffEXT.size(), o_coeffAB.size());
    const auto offset = mesh->fieldOffset;
    for (int s = n; s > 1; s--) {
      mesh->o_Jw.copyFrom(mesh->o_Jw, offset, (s - 1) * offset, (s - 2) * offset);
      mesh->o_invAJw.copyFrom(mesh->o_invAJw, offset, (s - 1) * offset, (s - 2) * offset);
    }
  }

  launchKernel("core-nStagesSum3Vector",
               mesh->Nlocal,
               fieldOffset,
               nAB,
               o_coeffAB,
               o_U,
               mesh->o_x,
               mesh->o_y,
               mesh->o_z);
  double flops = 6 * static_cast<double>(mesh->Nlocal) * nAB;
  platform->flopCounter->add("meshSolve", flops);

  mesh->update();
  if (mesh != meshV) {
    meshV->computeInvLMM();
  }

  if (ellipticSolver.size()) {
    updateZeroNormalMask();
  }
}

void geomSolver_t::solve(double time, int iter)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  platform->timer.tic("geomSolve");

  auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
  platform->linAlg->fill(o_rhs.size(), 0, o_rhs);

  auto o_lambda0 = o_prop;

  if (platform->options.compareArgs(upperCase(name) + " INITIAL GUESS", "EXTRAPOLATION") && iter == 1) {
    o_U.copyFrom(o_Ue);
  }

  ellipticSolver.at(0)->solve(o_lambda0, o_NULL, o_rhs, o_U.slice(0, fieldOffsetSum));

  platform->timer.toc("geomSolve");
}

void geomSolver_t::setupEllipticSolver()
{
  if (platform->options.compareArgs("GEOM SOLVER", "NONE")) {
    return;
  }

  nekrsCheck(!platform->options.compareArgs(upperCase(name) + " SOLVER", "BLOCK"),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "geom requires block solver!");

  auto o_lambda0 = o_prop;

  ellipticSolver.push_back(new elliptic(name, mesh, fieldOffset, o_lambda0, o_NULL));

  const auto unalignedBoundary = platform->app->bc->hasUnalignedMixed(name);
  if (unalignedBoundary) {
    o_zeroNormalMask = mesh->createZeroNormalMask(fieldOffset, ellipticSolver.at(0)->o_EToB());

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
}

void geomSolver_t::applyDirichlet(double time)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  if (platform->app->bc->hasUnalignedMixed(name)) {
    static occa::kernel kernel;
    if (!kernel.isInitialized()) {
      kernel = platform->kernelRequests.load("mesh-applyZeroNormalMask");
    }
    kernel(mesh->Nelements,
           fieldOffset,
           mesh->o_elementList,
           mesh->o_sgeo,
           o_zeroNormalMask,
           mesh->o_vmapM,
           ellipticSolver.at(0)->o_EToB(),
           o_U);
    kernel(mesh->Nelements,
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
  occa::memory o_UDirichlet = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
  platform->linAlg->fill(o_UDirichlet.size(), TINY, o_UDirichlet);

  for (int sweep = 0; sweep < 2; sweep++) {
    launchKernel("geomSolver_t::meshDirichletBCHex3D",
                 mesh->Nelements,
                 fieldOffset,
                 time,
                 static_cast<int>(deriveBCFromVelocity),
                 mesh->o_sgeo,
                 o_zeroNormalMask,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 o_EToB,
                 o_prop,
                 platform->app->bc->o_usrwrk,
                 o_Ufluid,
                 o_UDirichlet);

    oogs::startFinish(o_UDirichlet,
                      mesh->dim,
                      fieldOffset,
                      ogsDfloat,
                      (sweep == 0) ? ogsMax : ogsMin,
                      mesh->oogs3);
  }

  if (ellipticSolver.at(0)->Nmasked()) {
    launchKernel("core-maskCopy2",
                 ellipticSolver.at(0)->Nmasked(),
                 0,
                 0,
                 ellipticSolver.at(0)->o_maskIds(),
                 o_UDirichlet,
                 o_U,
                 o_Ue);
  }
}

void geomSolver_t::saveSolutionState()
{
  if (!o_U0.isInitialized()) {
    o_U0 = platform->device.malloc<dfloat>(o_U.size());
    o_prop0 = platform->device.malloc<dfloat>(o_prop.size());

    o_Jw0 = platform->device.malloc<dfloat>(mesh->o_Jw.size());
    o_invAJw0 = platform->device.malloc<dfloat>(mesh->o_invAJw.size());
    o_x0 = platform->device.malloc<dfloat>(mesh->o_x.size());
    o_y0 = platform->device.malloc<dfloat>(mesh->o_y.size());
    o_z0 = platform->device.malloc<dfloat>(mesh->o_z.size());
  }

  o_U0.copyFrom(o_U);
  o_prop0.copyFrom(o_prop);
  o_Jw0.copyFrom(mesh->o_Jw);
  o_invAJw0.copyFrom(mesh->o_invAJw);
  o_x0.copyFrom(mesh->o_x);
  o_y0.copyFrom(mesh->o_y);
  o_z0.copyFrom(mesh->o_z);
}

void geomSolver_t::restoreSolutionState()
{
  o_U.copyFrom(o_U0);
  o_prop.copyFrom(o_prop0);
  mesh->o_Jw.copyFrom(o_Jw0);
  mesh->o_invAJw.copyFrom(o_invAJw0);
  mesh->o_x.copyFrom(o_x0);
  mesh->o_y.copyFrom(o_y0);
  mesh->o_z.copyFrom(o_z0);

  mesh->update();
}

void geomSolver_t::extrapolateSolution()
{
  if (ellipticSolver.size() > 0) {
    launchKernel("core-extrapolate",
                 mesh->Nlocal,
                 mesh->dim,
                 static_cast<int>(o_coeffEXT.size()),
                 fieldOffset,
                 o_coeffEXT,
                 o_U,
                 o_Ue);
  }
}

void geomSolver_t::lagSolution()
{
  const auto n = std::max(o_coeffEXT.size(), o_coeffAB.size());
  for (int s = n; s > 1; s--) {
    o_U.copyFrom(o_U, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
  }
}

void geomSolver_t::updateZeroNormalMask()
{
  if (platform->app->bc->hasUnalignedMixed(name)) {
    o_zeroNormalMask = mesh->createZeroNormalMask(fieldOffset, ellipticSolver[0]->o_EToB());
  }
};

void geomSolver_t::computeDiv()
{
  if (!o_div.isInitialized()) {
    return;
  }

  for (int s = o_coeffEXT.size(); s > 1; s--) {
    o_div.copyFrom(o_div, fieldOffset, (s - 1) * fieldOffset, (s - 2) * fieldOffset);
  }
  opSEM::strongDivergence(mesh, fieldOffset, o_U, o_div, false);
};

void registerGeomSolverKernels(occa::properties kernelInfoBC)
{
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/geom/");
  const std::string suffix = "Hex3D";
  std::string kernelName;
  std::string fileName;
  std::string section = "geomSolver_t::";

  kernelName = "meshDirichletBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);
}

void geomSolver_t::setTimeIntegrationCoeffs(int tstep)
{
  if (o_coeffAB.size()) {
    const int meshOrder = std::min(tstep, static_cast<int>(o_coeffAB.size()));

    std::vector<dfloat> coeff(o_coeffAB.size());
    nek::coeffAB(coeff.data(), dt, meshOrder);
    for (int i = 0; i < meshOrder; ++i) {
      coeff[i] *= dt[0];
    }
    for (int i = o_coeffAB.size(); i > meshOrder; i--) {
      coeff[i - 1] = 0.0;
    }
    o_coeffAB.copyFrom(coeff.data());
  }
}
