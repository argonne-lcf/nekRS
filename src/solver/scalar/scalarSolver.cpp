#include "scalarSolver.hpp"
#include "lowPassFilter.hpp"
#include "advectionSubCycling.hpp"
#include "avm.hpp"
#include "gjp.hpp"
#include <registerKernels.hpp>

static void advectionFlops(mesh_t *mesh, int Nfields)
{
  const auto cubNq = mesh->cubNq;
  const auto cubNp = mesh->cubNp;
  const auto Nq = mesh->Nq;
  const auto Np = mesh->Np;
  const auto Nelements = mesh->Nelements;
  double flopCount = 0.0; // per elem basis
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    flopCount += 4. * Nq * (cubNp + cubNq * cubNq * Nq + cubNq * Nq * Nq); // interpolation
    flopCount += 6. * cubNp * cubNq;                                       // apply Dcub
    flopCount += 5 * cubNp; // compute advection term on cubature mesh
    flopCount += mesh->Np;  // weight by inv. mass matrix
  } else {
    flopCount += 8 * (Np * Nq + Np);
  }

  flopCount *= Nelements;
  flopCount *= Nfields;

  platform->flopCounter->add("advection", flopCount);
}

void scalar_t::advectionSubcycling(int nEXT, double time, int is)
{
  const auto mesh = this->_mesh[is];

  const auto nFields = 1;

  auto o_Si = o_S.slice(fieldOffsetScan[is], mesh->Nlocal);
  auto o_JwFi = o_JwF.slice(fieldOffsetScan[is], mesh->Nlocal);

  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      kernel = platform->kernelRequests.load("core-subCycleStrongCubatureVolumeScalarHex3D");
    } else {
      kernel = platform->kernelRequests.load("core-subCycleStrongVolumeScalarHex3D");
    }
  }

  platform->linAlg->fill(o_JwFi.size(), 0, o_JwFi);

  advectionSubcyclingRK(mesh,
                        meshV,
                        time,
                        dt,
                        Nsubsteps,
                        o_coeffBDF,
                        nEXT,
                        nFields,
                        kernel,
                        meshV->oogs,
                        mesh->fieldOffset,
                        fieldOffset(),
                        vCubatureOffset,
                        fieldOffsetSum,
                        (geom) ? geom->o_div : o_NULL,
                        o_relUrst,
                        o_Si,
                        o_JwFi);

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 1,
                                                                 0,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_JwFi,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s%s advSub norm: %.15e\n", "scalar", scalarDigitStr(is).c_str(), debugNorm);
    }
  }
}

scalar_t::scalar_t(scalarConfig_t &cfg, const std::unique_ptr<geomSolver_t> &_geom) : geom(_geom)
{

  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ " << "SETUP SCALAR" << " ===============\n";
  }

  auto &options = platform->options;
  const std::string section = "scalar_t::";
  platform_t *platform = platform_t::getInstance();

  Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  NSfields = cfg.Nscalar;

  qqt.resize(NSfields);
  fieldOffsetScan.resize(NSfields);
  ellipticSolver.resize(NSfields);
  compute.resize(NSfields);
  cvodeSolve.resize(NSfields);

  meshV = cfg.meshV;

  g0 = cfg.g0;
  dt = cfg.dt;
  o_coeffBDF = cfg.o_coeffBDF;
  o_coeffEXT = cfg.o_coeffEXT;

  _fieldOffset = cfg.fieldOffset; // for now same for all scalars
  vFieldOffset = o_U.isInitialized() ? cfg.vFieldOffset : _fieldOffset;
  vCubatureOffset = cfg.vCubatureOffset;

  o_U = cfg.o_U;
  if (!o_U.isInitialized()) {
    o_U = platform->device.malloc<dfloat>(meshV->dim * std::max(o_coeffBDF.size(), o_coeffEXT.size()) *
                                          vFieldOffset);
  }

  o_Ue = cfg.o_Ue;
  if (!o_Ue.isInitialized()) {
    o_Ue = platform->device.malloc<dfloat>(meshV->dim * vFieldOffset);
  }

  o_relUrst = cfg.o_relUrst;
  if (!o_relUrst.isInitialized()) {
    const dlong Nstates = Nsubsteps ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
    o_relUrst = platform->device.malloc<dfloat>(Nstates * meshV->dim * vCubatureOffset);
  }

  dpdt = cfg.dpdt;
  dp0thdt = cfg.dp0thdt;
  alpha0Ref = cfg.alpha0Ref;

  dlong sum = 0;
  for (int s = 0; s < NSfields; ++s) {
    fieldOffsetScan[s] = (s > 0) ? sum : 0;
    sum += _fieldOffset;
    this->_mesh.push_back(cfg.mesh[s]);
    qqt[s] = new QQt(this->_mesh[s]->oogs);
  }
  fieldOffsetSum = sum;
  o_fieldOffsetScan = platform->device.malloc<dlong>(NSfields, fieldOffsetScan.data());

  o_prop = platform->device.malloc<dfloat>(2 * fieldOffsetSum);
  o_diff = o_prop.slice(0 * fieldOffsetSum, fieldOffsetSum);
  o_rho = o_prop.slice(1 * fieldOffsetSum, fieldOffsetSum);

  for (int is = 0; is < NSfields; is++) {
    const std::string sid = scalarDigitStr(is);

    const auto _name = lowerCase(options.getArgs("SCALAR" + sid + " NAME"));
    name.push_back(_name);
    nameToIndex[_name] = is;

    auto o_tmp = [&]() {
      const auto prefixedName = "scalar " + _name;
      auto tmp = platform->device.malloc<char>(prefixedName.size() + 1);
      tmp.copyFrom(prefixedName.data());
      const char nullChar[] = {'\0'};
      tmp.copyFrom(nullChar, 1, prefixedName.size());
      return tmp;
    }();
    o_name.push_back(o_tmp);

    if (options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
      continue;
    }

    nekrsCheck(options.compareArgs("SCALAR" + sid + " SOLVER", "BLOCK"),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "scalar does not support BLOCK solver!");

    if (platform->comm.mpiRank() == 0) {
      std::cout << "S" << sid << ": " << name[is] << std::endl;
    }
    platform->app->bc->printBcTypeMapping("scalar" + sid);
    if (platform->comm.mpiRank() == 0) {
      std::cout << std::endl;
    }

    dfloat diff = 1;
    dfloat rho = 1;
    options.getArgs("SCALAR" + sid + " DIFFUSIONCOEFF", diff);
    options.getArgs("SCALAR" + sid + " TRANSPORTCOEFF", rho);

    auto o_diff_i = o_diff + fieldOffsetScan[is];
    auto o_rho_i = o_rho + fieldOffsetScan[is];

    std::vector<dfloat> diffTmp(this->_mesh[is]->Nlocal, diff);
    std::vector<dfloat> rhoTmp(this->_mesh[is]->Nlocal, rho);

    dfloat diffSolid = diff;
    dfloat rhoSolid = rho;
    options.getArgs("SCALAR" + sid + " DIFFUSIONCOEF SOLID", diffSolid);
    options.getArgs("SCALAR" + sid + " TRANSPORTCOEF SOLID", rhoSolid);
    for (int i = meshV->Nlocal; i < this->_mesh[is]->Nlocal; i++) {
      diffTmp[i] = diffSolid;
      rhoTmp[i] = rhoSolid;
    }

    o_diff_i.copyFrom(diffTmp.data(), diffTmp.size());
    o_rho_i.copyFrom(rhoTmp.data(), rhoTmp.size());
  }

  anyCvodeSolver = false;
  anyEllipticSolver = false;

  EToBOffset = [&]() {
    dlong NelementsMax = 0;
    for (int is = 0; is < NSfields; is++) {
      NelementsMax = std::max(this->_mesh[is]->Nelements, NelementsMax);
    }
    return NelementsMax * meshV->Nfaces;
  }();

  std::vector<int> EToB(EToBOffset * NSfields);

  for (int is = 0; is < NSfields; is++) {
    std::string sid = scalarDigitStr(is);

    compute[is] = 1;
    if (options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
      compute[is] = 0;
      cvodeSolve[is] = 0;
      continue;
    }

    cvodeSolve[is] = options.compareArgs("SCALAR" + sid + " SOLVER", "CVODE");
    anyCvodeSolver |= cvodeSolve[is];
    anyEllipticSolver |= (!cvodeSolve[is] && compute[is]);

    auto mesh = this->_mesh[is];

    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        EToB[cnt + EToBOffset * is] =
            platform->app->bc->typeId(mesh->EToB[f + e * mesh->Nfaces], "scalar" + sid);
        cnt++;
      }
    }
  }

  o_EToB = platform->device.malloc<int>(EToB.size());
  o_EToB.copyFrom(EToB.data());

  o_compute = platform->device.malloc<dlong>(NSfields, compute.data());
  o_cvodeSolve = platform->device.malloc<dlong>(NSfields, cvodeSolve.data());

  int nFieldsAlloc = anyEllipticSolver ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_S = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);

  nFieldsAlloc = anyEllipticSolver ? o_coeffEXT.size() : 1;
  o_ADV = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  o_EXT = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);

  if (anyEllipticSolver) {
    o_Se = platform->device.malloc<dfloat>(fieldOffsetSum);
    o_JwF = platform->device.malloc<dfloat>(fieldOffsetSum);
  }

  bool filteringEnabled = false;
  bool avmEnabled = false;
  for (int is = 0; is < NSfields; is++) {
    const auto sid = scalarDigitStr(is);

    if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "HPFRT")) {
      filteringEnabled = true;
    }

    if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY")) {
      avmEnabled = true;
    }
  }

  if (filteringEnabled) {
    std::vector<dlong> applyFilterRT(NSfields, 0);
    const dlong Nmodes = meshV->N + 1; // assumed to be the same for all fields
    o_filterRT = platform->device.malloc<dfloat>(NSfields * Nmodes * Nmodes);
    o_filterS = platform->device.malloc<dfloat>(NSfields);
    o_applyFilterRT = platform->device.malloc<dlong>(NSfields);
    std::vector<dfloat> filterS(NSfields, 0);
    for (int is = 0; is < NSfields; is++) {
      std::string sid = scalarDigitStr(is);

      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "NONE")) {
        continue;
      }
      if (!compute[is]) {
        continue;
      }

      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "HPFRT")) {
        int filterNc = -1;
        options.getArgs("SCALAR" + sid + " HPFRT MODES", filterNc);
        dfloat strength = NAN;
        options.getArgs("SCALAR" + sid + " HPFRT STRENGTH", strength);
        filterS[is] = strength;
        this->o_filterRT.copyFrom(lowPassFilterSetup(this->_mesh[is], filterNc),
                                  Nmodes * Nmodes,
                                  is * Nmodes * Nmodes);

        applyFilterRT[is] = 1;
      }
    }

    o_filterS.copyFrom(filterS.data(), NSfields);
    o_applyFilterRT.copyFrom(applyFilterRT.data(), NSfields);
  }

  if (avmEnabled) {
    avm::setup(meshV);
  }

  if (anyCvodeSolver) {
    cvode = new cvode_t(this);
  }

  auto verifyBC = [&]() {
    for (int is = 0; is < NSfields; is++) {
      if (!compute[is]) {
        continue;
      }

      const std::string field = "scalar" + scalarDigitStr(is);
      nekrsCheck(_mesh[is]->Nbid != platform->app->bc->size(field),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 field.c_str(),
                 platform->app->bc->size(field),
                 _mesh[is]->Nbid);
    }
  };

  verifyBC();
}

void scalar_t::makeAdvection(int is, double time, int tstep)
{
  if (Nsubsteps) {
    advectionSubcycling(std::min(tstep, static_cast<int>(o_coeffEXT.size())), time, is);
  } else {
    auto mesh = meshV;

    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      launchKernel("core-strongAdvectionCubatureVolumeScalarHex3D",
                   mesh->Nelements,
                   1, /* nScalars */
                   0, /* weighted */
                   0, /* sharedRho */
                   mesh->o_vgeo,
                   mesh->o_cubDiffInterpT,
                   mesh->o_cubInterpT,
                   mesh->o_cubProjectT,
                   o_compute + is,
                   o_fieldOffsetScan + is,
                   vFieldOffset,
                   vCubatureOffset,
                   o_S,
                   o_relUrst,
                   o_rho,
                   o_ADV);
    } else {
      launchKernel("core-strongAdvectionVolumeScalarHex3D",
                   mesh->Nelements,
                   1, /* nScalars */
                   0, /* weighted */
                   mesh->o_vgeo,
                   mesh->o_D,
                   o_compute + is,
                   o_fieldOffsetScan + is,
                   vFieldOffset,
                   o_S,
                   o_relUrst,
                   o_rho,
                   o_ADV);
    }
    advectionFlops(mesh, 1);
  }
}

void scalar_t::makeExplicit(int is, double time, int tstep)
{
  const std::string sid = scalarDigitStr(is);

  auto mesh = this->_mesh[is];
  const dlong isOffset = fieldOffsetScan[is];

  if (platform->options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "HPFRT")) {
    launchKernel("core-filterRTHex3D",
                 meshV->Nelements,
                 is,
                 1,
                 o_fieldOffsetScan,
                 o_applyFilterRT,
                 o_filterRT,
                 o_filterS,
                 o_rho,
                 o_S,
                 o_EXT);

    double flops = 6 * mesh->Np * mesh->Nq + 4 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    platform->flopCounter->add("scalarFilterRT", flops);
  }

  if (platform->options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "GJP")) {
    dfloat tauFactor;
    platform->options.getArgs("SCALAR" + sid + " REGULARIZATION GJP SCALING COEFF", tauFactor);

    auto o_Si = o_S.slice(fieldOffsetScan[is], mesh->Nlocal);
    auto o_EXTi = o_EXT.slice(fieldOffsetScan[is], mesh->Nlocal);
    auto o_rhoi = o_rho.slice(fieldOffsetScan[is], mesh->Nlocal);

    addGJP(mesh, ellipticSolver[is]->o_EToB(), o_rho, vFieldOffset, o_U, o_Si, o_EXTi, tauFactor);
  }

  const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  if (movingMesh && !Nsubsteps) {
    launchKernel("scalar_t::advectMeshVelocityHex3D",
                 meshV->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 isOffset,
                 (geom) ? geom->fieldOffset : 0,
                 o_rho,
                 (geom) ? geom->o_U : o_NULL,
                 o_S,
                 o_EXT);
    double flops = 18 * mesh->Np * mesh->Nq + 21 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    platform->flopCounter->add("scalar advectMeshVelocity", flops);
  }
}

void scalar_t::saveSolutionState()
{
  if (!o_S0.isInitialized()) {
    o_S0 = platform->device.malloc<dfloat>(o_S.length());
    o_EXT0 = platform->device.malloc<dfloat>(o_EXT.length());
    o_ADV0 = platform->device.malloc<dfloat>(o_ADV.length());
    o_prop0 = platform->device.malloc<dfloat>(o_prop.length());
  }

  o_S0.copyFrom(o_S, o_S.length());
  o_EXT0.copyFrom(o_EXT, o_EXT.length());
  o_ADV0.copyFrom(o_ADV, o_EXT.length());
  o_prop0.copyFrom(o_prop, o_prop.length());
}

void scalar_t::restoreSolutionState()
{
  o_S0.copyTo(o_S, o_S.length());
  o_EXT0.copyTo(o_EXT, o_EXT.length());
  o_ADV0.copyTo(o_ADV, o_ADV.length());
  o_prop0.copyTo(o_prop, o_prop.length());
}

void scalar_t::applyAVM()
{
  auto verbose = platform->verbose();
  auto mesh = this->meshV; // assumes mesh is the same for all scalars
  static std::vector<occa::memory> o_diff0(NSfields);

  static std::vector<occa::memory> o_nuAVM;
  static auto initialized = false;
  if (!initialized) {
    for (int is = 0; is < NSfields; is++) {
      const auto sid = scalarDigitStr(is);

      if (platform->options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD",
                                        "AVM_AVERAGED_MODAL_DECAY")) {
        nekrsCheck(mesh->N < 5,
                   platform->comm.mpiComm(),
                   EXIT_FAILURE,
                   "%s\n",
                   "AVM requires polynomialOrder >= 5!");

        o_diff0[is] = platform->device.malloc<dfloat>(mesh->Nlocal);
        o_diff0[is].copyFrom(o_diff, mesh->Nlocal, 0, fieldOffsetScan[is]);
      }
    }
    initialized = true;
  }

  for (int scalarIndex = 0; scalarIndex < NSfields; scalarIndex++) {
    const auto sid = scalarDigitStr(scalarIndex);

    if (!platform->options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD",
                                       "AVM_AVERAGED_MODAL_DECAY")) {
      continue;
    }

    // restore inital viscosity
    o_diff.copyFrom(o_diff0.at(scalarIndex), o_diff0.at(scalarIndex).size(), fieldOffsetScan[scalarIndex]);

    dfloat kappa = 1.0;
    platform->options.getArgs("SCALAR" + sid + " REGULARIZATION AVM ACTIVATION WIDTH", kappa);

    dfloat logS0 = 2.0; // threshold smoothness exponent (activate for logSk > logS0 - kappa)
    platform->options.getArgs("SCALAR" + sid + " REGULARIZATION AVM DECAY THRESHOLD", logS0);

    dfloat scalingCoeff = 1.0;
    platform->options.getArgs("SCALAR" + sid + " REGULARIZATION AVM SCALING COEFF", scalingCoeff);

    dfloat absTol = 0;
    platform->options.getArgs("SCALAR" + sid + " REGULARIZATION AVM ABSOLUTE TOL", absTol);

    const bool makeCont = platform->options.compareArgs("SCALAR" + sid + " REGULARIZATION AVM C0", "TRUE");

    auto o_Si = o_S.slice(fieldOffsetScan[scalarIndex], mesh->Nlocal);
    auto o_eps = avm::viscosity(vFieldOffset, o_U, o_Si, absTol, scalingCoeff, logS0, kappa, makeCont);

    if (verbose) {
      const dfloat maxEps = platform->linAlg->max(mesh->Nlocal, o_eps, platform->comm.mpiComm());
      const dfloat minEps = platform->linAlg->min(mesh->Nlocal, o_eps, platform->comm.mpiComm());
      occa::memory o_S_slice = o_diff + fieldOffsetScan[scalarIndex];
      const dfloat maxDiff = platform->linAlg->max(mesh->Nlocal, o_S_slice, platform->comm.mpiComm());
      const dfloat minDiff = platform->linAlg->min(mesh->Nlocal, o_S_slice, platform->comm.mpiComm());

      if (platform->comm.mpiRank() == 0) {
        printf("applying a min/max artificial viscosity of (%f,%f) to scalar%s with min/max visc (%f,%f)\n",
               minEps,
               maxEps,
               sid.c_str(),
               minDiff,
               maxDiff);
      }
    }

    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_eps, 1.0, o_diff, 0, fieldOffsetScan[scalarIndex]);

    if (verbose) {
      occa::memory o_S_slice = o_diff + fieldOffsetScan[scalarIndex];
      const dfloat maxDiff = platform->linAlg->max(mesh->Nlocal, o_S_slice, platform->comm.mpiComm());
      const dfloat minDiff = platform->linAlg->min(mesh->Nlocal, o_S_slice, platform->comm.mpiComm());

      if (platform->comm.mpiRank() == 0) {
        printf("scalar%s now has a min/max visc: (%f,%f)\n", sid.c_str(), minDiff, maxDiff);
      }
    }
  }
}

void scalar_t::applyDirichlet(double time)
{
  for (int is = 0; is < NSfields; is++) {
    if (!compute[is]) {
      continue;
    }
    if (cvodeSolve[is]) {
      continue;
    }

    auto mesh = this->_mesh[is];

    auto o_diff_i = o_diff + fieldOffsetScan[is];
    auto o_rho_i = o_rho + fieldOffsetScan[is];

    // lower than any other possible Dirichlet value
    static constexpr dfloat TINY = -1e30;
    occa::memory o_SiDirichlet = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    platform->linAlg->fill(o_SiDirichlet.size(), TINY, o_SiDirichlet);

    auto &neknek = platform->app->neknek;

    auto o_intValU = [&]() {
      if (neknek) {
        if (neknek->hasField("fluid velocity")) {
          return neknek->getField("fluid velocity").o_intVal;
        }
      }
      return o_NULL;
    }();

    auto o_intVal = [&]() {
      if (neknek) {
        if (neknek->hasField("scalar")) {
          return neknek->getField("scalar").o_intVal;
        }
      }
      return o_NULL;
    }();

    auto intValIdx = [&](int is) {
      int idx = -1;
      if (neknek) {
        if (neknek->hasField("scalar")) {
          idx = neknek->getField("scalar").intValFieldIndex(is);
        }
      }
      return idx;
    }(is);

    for (int sweep = 0; sweep < 2; sweep++) {
      launchKernel("scalar_t::dirichletBC",
                   o_name[is],
                   mesh->Nelements,
                   _fieldOffset,
                   is,
                   time,
                   mesh->o_sgeo,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   mesh->o_vmapM,
                   mesh->o_EToB,
                   o_EToB + is * EToBOffset,
                   o_Ue,
                   o_diff_i,
                   o_rho_i,
                   neknek ? neknek->intValOffset() : 0,
                   neknek ? neknek->o_pointMap() : o_NULL,
                   static_cast<int>(o_intValU.isInitialized()),
                   o_intValU,
                   o_intVal,
                   intValIdx,
                   platform->app->bc->o_usrwrk,
                   o_SiDirichlet);

      oogs::startFinish(o_SiDirichlet,
                        1,
                        _fieldOffset,
                        ogsDfloat,
                        (sweep == 0) ? ogsMax : ogsMin,
                        mesh->oogs);
    }
    occa::memory o_Si = o_S.slice(fieldOffsetScan[is], mesh->Nlocal);

    if (o_Se.isInitialized()) {
      occa::memory o_Si_e = o_Se.slice(fieldOffsetScan[is], mesh->Nlocal);

      if (ellipticSolver[is]->Nmasked()) {
        launchKernel("core-maskCopy2",
                     ellipticSolver[is]->Nmasked(),
                     0,
                     0,
                     ellipticSolver[is]->o_maskIds(),
                     o_SiDirichlet,
                     o_Si,
                     o_Si_e);
      }
    } else {
      if (ellipticSolver[is]->Nmasked()) {
        launchKernel("core-maskCopy",
                     ellipticSolver[is]->Nmasked(),
                     0,
                     0,
                     ellipticSolver[is]->o_maskIds(),
                     o_SiDirichlet,
                     o_Si);
      }
    }
  }
}

void scalar_t::setupEllipticSolver()
{
  for (int is = 0; is < NSfields; is++) {
    std::string sid = scalarDigitStr(is);

    if (!compute[is]) {
      continue;
    }

    if (cvodeSolve[is]) {
      continue;
    }

    auto o_rho_i = o_rho.slice(fieldOffsetScan[is], _mesh[is]->Nlocal);
    auto o_lambda0 = o_diff.slice(fieldOffsetScan[is], _mesh[is]->Nlocal);
    auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(_mesh[is]->Nlocal);
    platform->linAlg->axpby(_mesh[is]->Nlocal, *g0 / dt[0], o_rho_i, 0.0, o_lambda1);

    ellipticSolver[is] = new elliptic("scalar" + sid, _mesh[is], _fieldOffset, o_lambda0, o_lambda1);
  }
}

void scalar_t::makeForcing()
{
  for (int is = 0; is < this->NSfields; is++) {
    if (!compute[is] || cvodeSolve[is]) {
      continue;
    }

    launchKernel("scalar_t::sumMakef",
                 _mesh[is]->Nlocal,
                 _mesh[is]->o_LMM,
                 1 / dt[0],
                 o_coeffEXT,
                 o_coeffBDF,
                 fieldOffsetScan[is],
                 fieldOffsetSum,
                 _mesh[is]->fieldOffset,
                 o_rho,
                 o_S,
                 o_ADV,
                 o_EXT,
                 o_JwF);

    dfloat scalarSumMakef = (3 * o_coeffEXT.size() + 3);
    scalarSumMakef += (Nsubsteps) ? 1 : 3 * o_coeffBDF.size();
    platform->flopCounter->add("scalarSumMakef", scalarSumMakef * static_cast<double>(_mesh[is]->Nlocal));

    if (platform->verbose()) {
      const dfloat debugNorm = platform->linAlg->weightedNorm2Many(_mesh[is]->Nlocal,
                                                                   1,
                                                                   0,
                                                                   _mesh[is]->ogs->o_invDegree,
                                                                   o_JwF + fieldOffsetScan[is],
                                                                   platform->comm.mpiComm());
      if (platform->comm.mpiRank() == 0) {
        printf("%s%s Jwf norm: %.15e\n", "scalar", scalarDigitStr(is).c_str(), debugNorm);
      }
    }
  }

  const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    o_EXT.copyFrom(o_EXT, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    if (o_ADV.isInitialized()) {
      o_ADV.copyFrom(o_ADV, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    }
  }
}

void scalar_t::solve(double time, int stage)
{
  platform->timer.tic("scalarSolve");

  for (int is = 0; is < NSfields; is++) {
    if (!compute[is] || cvodeSolve[is]) {
      continue;
    }

    const std::string sid = scalarDigitStr(is);
    auto mesh = this->_mesh[is];

    platform->timer.tic("scalar rhs");

    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    o_rhs.copyFrom(o_JwF, mesh->Nlocal, 0, fieldOffsetScan[is]);

    auto o_lhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

    launchKernel("scalar_t::neumannBCHex3D",
                 o_name[is],
                 mesh->Nelements,
                 1,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 is,
                 time,
                 vFieldOffset,
                 _fieldOffset,
                 0,
                 EToBOffset,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 o_Ue,
                 o_S,
                 o_EToB,
                 o_diff,
                 o_rho,
                 platform->app->bc->o_usrwrk,
                 o_lhs,
                 o_rhs);

    platform->timer.toc("scalar rhs");

    const auto o_diff_i = o_diff.slice(fieldOffsetScan[is], mesh->Nlocal);

    const auto o_lambda0 = o_diff_i;
    const auto o_lambda1 = [&]() {
      const auto o_rho_i = o_rho.slice(fieldOffsetScan[is], mesh->Nlocal);
      auto o_l = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      platform->linAlg->axpby(mesh->Nlocal, *g0 / dt[0], o_rho_i, 0.0, o_l);

      if (userImplicitLinearTerm) {
        auto o_implicitLT = userImplicitLinearTerm(time, is);
        if (o_implicitLT.isInitialized()) {
          platform->linAlg->axpby(mesh->Nlocal, 1.0, o_implicitLT, 1.0, o_l);
        }
      }

      if (platform->app->bc->hasRobin("SCALAR" + sid)) {
        platform->linAlg->axpby(mesh->Nlocal, 1.0, o_lhs, 1.0, o_l);
      }

      return o_l;
    }();

    auto o_Si = [&]() {
      auto o_S0 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      if (platform->options.compareArgs("SCALAR" + sid + " INITIAL GUESS", "EXTRAPOLATION") && stage == 1) {
        o_S0.copyFrom(o_Se, o_S0.size(), 0, fieldOffsetScan[is]);
      } else {
        o_S0.copyFrom(o_S, o_S0.size(), 0, fieldOffsetScan[is]);
      }

      return o_S0;
    }();

    this->ellipticSolver[is]->solve(o_lambda0, o_lambda1, o_rhs, o_Si);
    o_Si.copyTo(o_S, o_Si.size(), fieldOffsetScan[is]);
  }

  platform->timer.toc("scalarSolve");
}

void scalar_t::lagSolution()
{
  if (!anyEllipticSolver) {
    return;
  }

  const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    o_S.copyFrom(o_S, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
  }
}

void scalar_t::extrapolateSolution()
{
  if (!o_Se.isInitialized()) {
    return;
  }
  const auto Nlocal = _fieldOffset; // assumed to be the same for all fields
  launchKernel("core-extrapolate",
               Nlocal,
               NSfields,
               static_cast<int>(o_coeffEXT.size()),
               _fieldOffset,
               o_coeffEXT,
               o_S,
               o_Se);
}

void scalar_t::finalize()
{
  for (int is = 0; is < NSfields; is++) {
    if (ellipticSolver[is]) {
      delete ellipticSolver[is];
    }
  }
  if (cvode) {
    delete cvode;
  }
}

void scalar_t::computeUrst()
{
  auto mesh = meshV;

  if (Nsubsteps) {
    for (int s = std::max(o_coeffBDF.size(), o_coeffEXT.size()); s > 1; s--) {
      auto lagOffset = mesh->dim * vCubatureOffset;
      o_relUrst.copyFrom(o_relUrst, lagOffset, (s - 1) * lagOffset, (s - 2) * lagOffset);
    }
  }

  const auto relative = geom && Nsubsteps;
  double flopCount = 0.0;
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    launchKernel("nrs-UrstCubatureHex3D",
                 mesh->Nelements,
                 static_cast<int>(relative),
                 mesh->o_cubvgeo,
                 mesh->o_cubInterpT,
                 vFieldOffset,
                 vCubatureOffset,
                 o_U,
                 (geom) ? geom->o_U : o_NULL,
                 o_relUrst);
    flopCount += 6 * mesh->Np * mesh->cubNq;
    flopCount += 6 * mesh->Nq * mesh->Nq * mesh->cubNq * mesh->cubNq;
    flopCount += 6 * mesh->Nq * mesh->cubNp;
    flopCount += 24 * mesh->cubNp;
    flopCount *= mesh->Nelements;
  } else {
    launchKernel("nrs-UrstHex3D",
                 mesh->Nelements,
                 static_cast<int>(relative),
                 mesh->o_vgeo,
                 vFieldOffset,
                 o_U,
                 (geom) ? geom->o_U : o_NULL,
                 o_relUrst);
    flopCount += 24 * static_cast<double>(mesh->Nlocal);
  }
  platform->flopCounter->add("Urst", flopCount);
}

void registerScalarKernels(occa::properties kernelInfoBC)
{
  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";
  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;
  constexpr int Nfaces{6};

  constexpr int NVfields{3};
  kernelInfo["defines/p_NVfields"] = NVfields;

  std::string fileName, kernelName;
  const std::string suffix = "Hex3D";
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/scalar/");
  const std::string section = "scalar_t::";
  occa::properties meshProps = kernelInfo;
  meshProps += meshKernelProperties(N);

  {
    kernelName = "advectMeshVelocityHex3D";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    kernelName = "neumannBC" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

    kernelName = "dirichletBC";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

    {
      occa::properties prop = kernelInfo;

      int Nsubsteps = 0;
      platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

      int nBDF = 0;
      int nEXT = 0;
      platform->options.getArgs("BDF ORDER", nBDF);
      platform->options.getArgs("EXT ORDER", nEXT);
      if (Nsubsteps) {
        nEXT = nBDF;
      }

      const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

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
  }

  registerCvodeKernels();
}
