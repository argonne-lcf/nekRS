#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "udf.hpp"
#include "lowPassFilter.hpp"
#include "avm.hpp"
#include "re2Reader.hpp"
#include "scalarSolver.hpp"
#include "advectionSubCycling.hpp"
#include "elliptic.hpp"
#include "iofldFactory.hpp"
#include "ellipticBcTypes.h"

#include <registerKernels.hpp>

int nrs_t::numberActiveFields()
{
  int fields = 0;
  if (platform->options.compareArgs("FLUID", "TRUE")) {
    fields++;
  }
  for (int is = 0; is < Nscalar; ++is) {
    std::string sid = scalarDigitStr(is);
    if (!platform->options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
      fields++;
    }
  }
  return fields;
}

void nrs_t::printSolutionMinMax()
{
  if (platform->comm.mpiRank() == 0) {
    printf("================= INITIAL CONDITION ====================\n");
  }

  {
    auto mesh = meshT;
    auto o_x = mesh->o_x;
    auto o_y = mesh->o_y;
    auto o_z = mesh->o_z;

    const auto minMax =
        platform->linAlg->minMax(mesh->Nlocal, {mesh->o_x, mesh->o_y, mesh->o_z}, platform->comm.mpiComm());

    if (platform->comm.mpiRank() == 0) {
      printf("%-15s min/max: %g %g  %g %g  %g %g\n",
             "MESH X",
             minMax[0].first,
             minMax[0].second,
             minMax[1].first,
             minMax[1].second,
             minMax[2].first,
             minMax[2].second);
    }
  }

  if (geom) {
    auto mesh = geom->mesh;
    auto o_ux = geom->o_U + 0 * geom->fieldOffset;
    auto o_uy = geom->o_U + 1 * geom->fieldOffset;
    auto o_uz = geom->o_U + 2 * geom->fieldOffset;

    const auto minMax = platform->linAlg->minMax(mesh->Nlocal, {o_ux, o_uy, o_uz}, platform->comm.mpiComm());

    if (platform->comm.mpiRank() == 0) {
      printf("%-15s min/max: %g %g  %g %g  %g %g\n",
             "GEOM U",
             minMax[0].first,
             minMax[0].second,
             minMax[1].first,
             minMax[1].second,
             minMax[2].first,
             minMax[2].second);
    }
  }

  if (fluid) {
    auto mesh = fluid->mesh;
    auto o_ux = fluid->o_U + 0 * fluid->fieldOffset;
    auto o_uy = fluid->o_U + 1 * fluid->fieldOffset;
    auto o_uz = fluid->o_U + 2 * fluid->fieldOffset;

    const auto minMax =
        platform->linAlg->minMax(mesh->Nlocal, {o_ux, o_uy, o_uz, fluid->o_P}, platform->comm.mpiComm());

    if (platform->comm.mpiRank() == 0) {
      printf("%-15s min/max: %g %g  %g %g  %g %g\n",
             "FLUID U",
             minMax[0].first,
             minMax[0].second,
             minMax[1].first,
             minMax[1].second,
             minMax[2].first,
             minMax[2].second);
    }

    if (platform->comm.mpiRank() == 0) {
      printf("%-15s min/max: %g %g\n", "FLUID p", minMax[3].first, minMax[3].second);
    }
  }

  if (Nscalar) {
    if (platform->comm.mpiRank() == 0) {
      printf("%-15s min/max:", "SCALAR S");
    }

    int cnt = 0;
    for (int is = 0; is < scalar->NSfields; is++) {
      cnt++;

      auto mesh = scalar->mesh(is);
      auto o_si = scalar->o_S + scalar->fieldOffsetScan[is];
      const auto minMax = platform->linAlg->minMax(mesh->Nlocal, {o_si}, platform->comm.mpiComm());

      if (platform->comm.mpiRank() == 0) {
        if (cnt > 1) {
          printf("  ");
        } else {
          printf(" ");
        }
        printf("%g %g", minMax[0].first, minMax[0].second);
      }
    }
    if (platform->comm.mpiRank() == 0) {
      printf("\n");
    }
  }
}

void nrs_t::setDefaultSettings(setupAide &options)
{
  // some settings are required for JIT compilation
  // user has no option to modify the settings defined here
  if (options.compareArgs("EQUATION TYPE", "NAVIERSTOKES")) {
    const auto dealiasing = (options.compareArgs("OVERINTEGRATION", "TRUE")) ? true : false;
    if (dealiasing) {
      options.setArgs("ADVECTION TYPE", "CUBATURE+CONVECTIVE");
    } else {
      options.setArgs("ADVECTION TYPE", "CONVECTIVE");
    }
  } else {
    options.removeArgs("ADVECTION TYPE");
  }

  if (options.getArgs("MOVING MESH").empty()) {
    options.setArgs("MOVING MESH", "FALSE");
  }
}

nrs_t::nrs_t()
{
  setDefaultSettings(platform->options);
}

void nrs_t::init()
{
  if (platform->options.compareArgs("FLUID STRESSFORMULATION", "TRUE")) {
    nekrsCheck(!platform->options.compareArgs("FLUID VELOCITY SOLVER", "BLOCK"),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "stressformulation requires block solver!");
  }

  const auto meshTRequested = [&]() {
    int N;
    platform->options.getArgs("NUMBER OF SCALARS", N);
    for (int is = 0; is < N; is++) {
      if (platform->options.compareArgs("SCALAR" + scalarDigitStr(is) + " MESH", "SOLID")) {
        return true;
      }
    }
    return false;
  }();

  if (meshTRequested) {
    int nelgt, nelgv;
    const std::string meshFile = platform->options.getArgs("MESH FILE");
    re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm());

    nekrsCheck(nelgt == nelgv,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "No solid mesh elements found!");

    nekrsCheck(platform->options.compareArgs("MOVING MESH", "TRUE"),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "moving mesh not supported for mesh type fluid + solid");
  }

  nek::setup(numberActiveFields());

  dlong cubatureOffset = -1;

  auto getMesh = [&]() {
    int N, cubN;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);
    platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);

    auto [meshT, meshV] = createMesh(platform->comm.mpiComm(), N, cubN, platform->kernelInfo);

    // use same fieldOffset
    auto offset = meshT->Np * (meshT->Nelements);
    fieldOffset = alignStride<dfloat>(offset);
    meshT->fieldOffset = fieldOffset;
    meshV->fieldOffset = fieldOffset;

    auto cubOffset = offset;
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      cubOffset = std::max(cubOffset, meshT->Nelements * meshT->cubNp);
    }
    cubatureOffset = alignStride<dfloat>(cubOffset);

    return std::pair{meshT, meshV};
  }();
  meshT = getMesh.first;
  meshV = getMesh.second;

  printMeshMetrics(meshT);
  if (meshV != meshT) {
    printMeshMetrics(meshV);
  }

  int nBDF;
  int nEXT;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);

  if (platform->options.compareArgs("VARIABLE DT", "TRUE")) {
    platform->options.setArgs("MIN ADJUST DT RATIO", "0.5");
    platform->options.setArgs("MAX ADJUST DT RATIO", "1.5");
  }

  platform->options.getArgs("SUBCYCLING STEPS", advectionSubcycingSteps);
  if (advectionSubcycingSteps) {
    nEXT = nBDF;
    platform->options.setArgs("EXT ORDER", std::to_string(nEXT));
  }

  nekrsCheck(nEXT < nBDF,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "EXT order needs to be >= BDF order!");

  o_coeffEXT = platform->device.malloc<dfloat>(nEXT);
  o_coeffBDF = platform->device.malloc<dfloat>(nBDF);

  qqt = new QQt(meshV->oogs);
  qqtT = new QQt(meshT->oogs);

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    geom = [&]() {
      geomSolverCfg_t cfg;
      cfg.name = "geom";
      cfg.mesh = meshT;
      cfg.meshV = meshV;
      cfg.fieldOffset = fieldOffset;
      cfg.g0 = &g0;
      cfg.dt = dt;
      cfg.o_coeffEXT = o_coeffEXT;
      cfg.deriveBCFromVelocity = bc.useDerivedGeomBoundaryConditions();
      return std::make_unique<geomSolver_t>(cfg);
    }();
  }

  if (platform->options.compareArgs("FLUID", "TRUE")) {
    fluid = [&]() {
      fluidSolverCfg_t cfg;
      cfg.name = "fluid";
      cfg.velocityName = cfg.name + " velocity";
      cfg.pressureName = cfg.name + " pressure";
      cfg.mesh = meshV;
      cfg.fieldOffset = fieldOffset;
      cfg.cubatureOffset = cubatureOffset;
      cfg.g0 = &g0;
      cfg.dt = dt;
      cfg.o_coeffEXT = o_coeffEXT;
      cfg.o_coeffBDF = o_coeffBDF;

      return std::make_unique<fluidSolver_t>(cfg, geom);
    }();
  }

  if (fluid && geom) {
    geom->o_Ufluid = fluid->o_U;
  }

  platform->options.getArgs("NUMBER OF SCALARS", Nscalar);
  if (Nscalar) {
    scalar = [&]() {
      scalarConfig_t cfg;

      cfg.Nscalar = Nscalar;
      cfg.g0 = &g0;
      cfg.dt = dt;
      cfg.dpdt = false;
      cfg.o_coeffBDF = o_coeffBDF;
      cfg.o_coeffEXT = o_coeffEXT;
      cfg.fieldOffset = fieldOffset;
      cfg.vCubatureOffset = cubatureOffset;
      if (fluid) {
        cfg.vFieldOffset = fluid->fieldOffset;
        cfg.dpdt = bc.hasOutflow(fluid->name);
        cfg.o_U = fluid->o_U;
        cfg.o_Ue = fluid->o_Ue;
        cfg.o_relUrst = fluid->o_relUrst;
      }
      cfg.mesh.resize(Nscalar);
      for (int is = 0; is < Nscalar; is++) {
        cfg.mesh[is] =
            (platform->options.compareArgs("SCALAR" + scalarDigitStr(is) + " MESH", "SOLID")) ? meshT : meshV;
      }
      cfg.meshV = meshV;
      cfg.dp0thdt = &dp0thdt;
      cfg.alpha0Ref = &alpha0Ref;

      return std::make_unique<scalar_t>(cfg, geom);
    }();

    if (scalar->cvode) {
      scalar->cvode->setEvaluateProperties(
          std::bind(&nrs_t::evaluateProperties, this, std::placeholders::_1));
      scalar->cvode->setEvaluateDivergence(
          std::bind(&nrs_t::evaluateDivergence, this, std::placeholders::_1));
    }
  }

  // setup initial condition
  {
    setIC();

    double startTime;
    platform->options.getArgs("START TIME", startTime);
    evaluateProperties(startTime);
  }

  g0 = 1;
  platform->options.getArgs("DT", dt[0]);
  if (dt[0] == 0) {
    dt[0] = 1e-3; // in case user specifies 0 (estimate from user forcing)
  }

  if (scalar) {
    if (scalar->cvode) {
      scalar->cvode->initialize(); // needs to be called after setIC and evaluateProperties
    }
  }

  // rho, g0 * dt required for Helmholtz coefficients (eigenvalues for Chebyshev in ellipticSetup)
  if (fluid) {
    fluid->setupEllipticSolver();
  }
  if (scalar) {
    scalar->setupEllipticSolver();
  }
  if (geom) {
    geom->setupEllipticSolver();
  }

  setupNeknek();
}

void nrs_t::setupNeknek()
{
  {
    int intFound = 0;
    for (auto &&field : fieldsToSolve()) {
      for (int bID = 1; bID <= platform->app->bc->size(field); ++bID) {
        if (platform->app->bc->typeId(bID, field) == bdryBase::bcType_interpolation) {
          intFound = 1;
        }
      }
    }

    // findpts functions have to be called collectively across all sessions
    MPI_Allreduce(MPI_IN_PLACE, &intFound, 1, MPI_INT, MPI_MAX, platform->comm.mpiCommParent());
    if (!intFound) {
      return;
    }
  }

  if (platform->comm.mpiRank() == 0) {
    std::cout << "============= NEKNEK ==================" << std::endl;
  }

  if (fluid->ellipticSolverP) {
    nekrsCheck(fluid->ellipticSolverP->nullSpace() && platform->options.compareArgs("LOWMACH", "TRUE"),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "variable p0th is not supported!");
  }

  nekrsCheck(platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE"),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "constant flow rate support not supported");

  int nSessions, sessionID;
  platform->options.getArgs("NEKNEK NUMBER OF SESSIONS", nSessions);
  platform->options.getArgs("NEKNEK SESSION ID", sessionID);

  auto intFound = [&](mesh_t *mesh, const std::string name) {
    int cnt = 0;
    for (dlong f = 0; f < mesh->Nelements * mesh->Nfaces; ++f) {
      if (platform->app->bc->typeId(mesh->EToB[f], name) == bdryBase::bcType_interpolation) {
        cnt = 1;
        break;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &cnt, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm());
    return (cnt != 0) ? true : false;
  };

  struct neknekField {
    std::string name;
    mesh_t *mesh;
    std::vector<int> filter;
    dlong offsetSum;
    dlong offset;
    occa::memory o_fld;
  };

  auto mesh = meshT; // assumption: all neknek fields have the same mesh
                     // is satified as we allocate all fields and offsets using the largest mesh

  const auto neknekFields = [&]() {
    std::vector<neknekField> fields;
    std::vector<int> scalarFilter;
    for (auto &&fieldName : fieldsToSolve()) {
      if (fieldName == "fluid velocity") {
        if (!intFound(fluid->mesh, fieldName)) {
          continue;
        }

        fields.push_back(
            neknekField{fieldName, mesh, {}, fluid->fieldOffsetSum, fluid->fieldOffset, fluid->o_U});
      } else if (fieldName.find("scalar") != std::string::npos) {
        const auto id = std::stoi(fieldName.substr(std::string("scalar").length()));
        if (!intFound(scalar->mesh(id), fieldName)) {
          continue;
        }
        scalarFilter.push_back(id);
      }
    }

    if (scalarFilter.size()) {
      fields.push_back(neknekField{"scalar",
                                   mesh, // same for all scalars
                                   scalarFilter,
                                   scalar->fieldOffsetSum,
                                   scalar->fieldOffset(), // same for all scalars
                                   scalar->o_S});
    }

    return fields;
  }();

  neknek = std::make_unique<neknek_t>(mesh, nSessions, sessionID);

  for (auto &&field : neknekFields) {
    neknek->addVariable(field.name, field.filter, field.offsetSum, field.offset, field.o_fld);
  }

  neknek->setup();
}

void nrs_t::restartFromFiles(const std::vector<std::string> &fileList)
{
  for (const std::string &restartStr : fileList) {
    auto options = serializeString(restartStr, '+');
    const auto fileName = options[0];
    options.erase(options.begin());

    if (platform->comm.mpiRank() == 0) {
      if (options.size()) {
        std::cout << "restart options: ";
      }
      for (const auto &element : options) {
        std::cout << element << "  ";
      }
      std::cout << std::endl;
    }

    auto requestedStep = [&]() {
      auto it = std::find_if(options.begin(), options.end(), [](const std::string &s) {
        return s.find("step") != std::string::npos;
      });

      std::string val;
      if (it != options.end()) {
        val = serializeString(*it, '=').at(1);
        options.erase(it);
      }
      return (val.empty()) ? -1 : std::stoi(val);
    }();

    auto requestedTime = [&]() {
      auto it = std::find_if(options.begin(), options.end(), [](const std::string &s) {
        return s.find("time") != std::string::npos;
      });

      std::string val;
      if (it != options.end()) {
        val = serializeString(*it, '=').at(1);
        options.erase(it);
      }
      return val;
    }();

    auto pointInterpolation = [&]() {
      auto it = std::find_if(options.begin(), options.end(), [](const std::string &s) {
        return s.find("int") != std::string::npos;
      });

      auto found = false;
      if (it != options.end()) {
        found = true;
        options.erase(it);
      }
      return found;
    }();

    const auto requestedFields = [&]() {
      std::vector<std::string> flds;
      for (const auto &entry : {"x", "u", "p", "t", "s"}) {
        auto it = std::find_if(options.begin(), options.end(), [entry](const std::string &s) {
          auto ss = lowerCase(s);
          return ss.find(entry) != std::string::npos;
        });
        if (it != options.end()) {
          auto s = lowerCase(*it);
          flds.push_back(s);
        }
      }

      const int idxStart = (std::find(flds.begin(), flds.end(), "t") != flds.end()) ? 1 : 0;
      for (const auto &entry : flds) {
        if (entry == "s") {
          for (int i = idxStart; i < Nscalar; i++) {
            std::ostringstream oss;
            oss << "s" << std::setw(2) << std::setfill('0') << i;
            flds.push_back(oss.str());
          }

          flds.erase(std::remove(flds.begin(), flds.end(), "s"), flds.end());
          std::sort(flds.begin(), flds.end());
          auto last = std::unique(flds.begin(), flds.end());
          flds.erase(last, flds.end());
        }
      }

      return flds;
    }();

    auto fileNameEndsWithBp = [&]() {
      const std::string suffix = ".bp";
      if (fileName.size() >= suffix.size()) {
        return fileName.compare(fileName.size() - suffix.size(), suffix.size(), suffix) == 0;
      }
      return false;
    }();
    auto iofld = iofldFactory::create((fileNameEndsWithBp) ? "adios" : "");
    iofld->open(meshT, iofld::mode::read, fileName, requestedStep);

    const auto avaiableFields = iofld->availableVariables();
    if (platform->comm.mpiRank() == 0 && platform->verbose()) {
      for (const auto &entry : avaiableFields) {
        std::cout << " found variable " << entry << std::endl;
      }
    }

    double time = -1;
    iofld->addVariable("time", time);
    if (platform->options.compareArgs("LOWMACH", "TRUE")) {
      iofld->addVariable("p0th", p0th[0]);
    }

    auto checkOption = [&](const std::string &name) {
      if (requestedFields.size() == 0) {
        return true; // nothing specfied -> assign all
      }
      if (std::find(requestedFields.begin(), requestedFields.end(), name) != requestedFields.end()) {
        return true;
      }
      return false;
    };

    auto isAvailable = [&](const std::string &name) {
      return std::find(avaiableFields.begin(), avaiableFields.end(), name) != avaiableFields.end();
    };

    if (checkOption("x") && isAvailable("mesh")) {
      std::vector<occa::memory> o_iofldX;
      auto mesh = meshT;
      o_iofldX.push_back(mesh->o_x);
      o_iofldX.push_back(mesh->o_y);
      o_iofldX.push_back(mesh->o_z);
      iofld->addVariable("mesh", o_iofldX);
    }

    if (checkOption("u") && isAvailable("velocity")) {
      std::vector<occa::memory> o_iofldU;
      o_iofldU.push_back(fluid->o_U.slice(0 * fluid->fieldOffset, fluid->mesh->Nlocal));
      o_iofldU.push_back(fluid->o_U.slice(1 * fluid->fieldOffset, fluid->mesh->Nlocal));
      o_iofldU.push_back(fluid->o_U.slice(2 * fluid->fieldOffset, fluid->mesh->Nlocal));
      iofld->addVariable("velocity", o_iofldU);
    }

    if (checkOption("p") && isAvailable("pressure")) {
      std::vector<occa::memory> o_iofldP = {fluid->o_P.slice(0, meshV->Nlocal)};
      iofld->addVariable("pressure", o_iofldP);
    }

    if (Nscalar) {
      std::vector<occa::memory> o_iofldT;
      if (checkOption("t") && isAvailable("temperature")) {
        o_iofldT.push_back(scalar->o_S.slice(0, scalar->mesh(0)->Nlocal));
        iofld->addVariable("temperature", o_iofldT);
      }

      const auto scalarStart = (o_iofldT.size()) ? 1 : 0;
      for (int i = scalarStart; i < Nscalar; i++) {
        const auto sid = scalarDigitStr(i - scalarStart);
        if (checkOption("s" + sid) && isAvailable("scalar" + sid)) {
          auto o_Si = scalar->o_S.slice(scalar->fieldOffsetScan[i], meshV->Nlocal);
          std::vector<occa::memory> o_iofldSi = {o_Si};
          iofld->addVariable("scalar" + sid, o_iofldSi);
        }
      }
    }

    if (pointInterpolation) {
      iofld->readAttribute("interpolate", "true");
    }

    iofld->process();
    iofld->close();

    platform->options.setArgs("START TIME", (requestedTime.size()) ? requestedTime : to_string_f(time));
  }
}

void nrs_t::setIC()
{
  const auto tStart = MPI_Wtime();

  if (platform->comm.mpiRank() == 0) {
    std::cout << "setting IC ... \n" << std::flush;
  }

  if (nek::usrFile()) {
    getICFromNek();
  }

  if (!platform->options.getArgs("RESTART FILE NAME").empty()) {
    std::string restartStr;
    platform->options.getArgs("RESTART FILE NAME", restartStr);
    std::vector<std::string> fileList = serializeString(restartStr, ',');

    restartFromFiles(fileList);
  }

  double startTime;
  platform->options.getArgs("START TIME", startTime);

  if (nek::usrFile()) {
    copyToNek(startTime, 0, true);
    nek::userchk();
    copyFromNek();
  }

  if (platform->comm.mpiRank() == 0) {
    std::cout << "calling UDF_Setup ... \n" << std::flush;
  }
  udf.setup();

  // just in case mesh was modified in udf.setup
  meshT->update();
  if (meshT != meshV) {
    meshV->computeInvLMM();
  }

  auto projC0 = [&](mesh_t *mesh, int nFields, dlong fieldOffset, occa::memory &o_in) {
    platform->linAlg->axmyMany(mesh->Nlocal, nFields, fieldOffset, 0, 1.0, mesh->o_LMM, o_in);
    oogs::startFinish(o_in, nFields, fieldOffset, ogsDfloat, ogsAdd, mesh->oogs);
    platform->linAlg->axmyMany(mesh->Nlocal, nFields, fieldOffset, 0, 1.0, mesh->o_invLMM, o_in);
  };

  if (fluid) {
    projC0(fluid->mesh, fluid->mesh->dim, fluid->fieldOffset, fluid->o_U);
    projC0(fluid->mesh, 1, fluid->fieldOffset, fluid->o_P);
  }

  if (Nscalar) {
    for (int s = 0; s < Nscalar; ++s) {
      const std::string sid = scalarDigitStr(s);
      if (platform->options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
        continue;
      }
      auto o_Si = scalar->o_S + scalar->fieldOffsetScan[s];
      projC0(scalar->mesh(s), 1, 0, o_Si);
    }
  }

  copyToNek(startTime, 0, true); // in case IC was updated in udf_setup

  nekrsCheck(platform->options.compareArgs("LOWMACH", "TRUE") && p0th[0] <= 1e-6,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "Unreasonable p0th value %g!",
             p0th[0]);

  if (platform->comm.mpiRank() == 0) {
    printf("done (%gs)\n", MPI_Wtime() - tStart);
  }
}

void nrs_t::printRunStat(int step)
{
  const int rank = platform->comm.mpiRank();
  auto comm_ = platform->comm.mpiComm();

  double gsTime = ogsTime(/* reportHostTime */ true);
  MPI_Allreduce(MPI_IN_PLACE, &gsTime, 1, MPI_DOUBLE, MPI_MAX, comm_);

  const double tElapsedTime = platform->timer.query("elapsed", "HOST:MAX");

  if (rank == 0) {
    std::cout << "\n>>> runtime statistics (step= " << step << "  totalElapsed= " << tElapsedTime << "s"
              << "):\n";
  }

  std::cout.setf(std::ios::scientific);
  int outPrecisionSave = std::cout.precision();
  std::cout.precision(5);

  if (rank == 0) {
    std::cout << "name                    "
              << "time          "
              << "abs%  "
              << "rel%  "
              << "calls\n";
  }

  const double tElapsedTimeSolve = platform->timer.query("elapsedStepSum", "HOST:MAX");
  platform->timer.printStatSetElapsedTimeSolve(tElapsedTimeSolve);
  const double tSetup = platform->timer.query("setup", "HOST:MAX");

  const double tMinSolveStep = platform->timer.query("minSolveStep", "HOST:MAX");
  const double tMaxSolveStep = platform->timer.query("maxSolveStep", "HOST:MAX");

  const double tScalarCvode = platform->timer.query("cvode_t::solve", "DEVICE:MAX");

  bool printFlops =
      !platform->options.compareArgs("FLUID PRESSURE PRECONDITIONER", "SEMFEM") && tScalarCvode < 0;

  const double flops = platform->flopCounter->get(platform->comm.mpiComm()) /
                       (tElapsedTimeSolve * platform->comm.mpiCommSize());

  platform->timer.printStatEntry("  solve                 ", tElapsedTimeSolve, tElapsedTimeSolve);
  if (tElapsedTimeSolve > 0 && rank == 0) {
    std::cout << "    min                 " << tMinSolveStep << "s\n";
    std::cout << "    max                 " << tMaxSolveStep << "s\n";
    if (printFlops) {
      std::cout << "    flops/rank          " << flops << "\n";
    }
  }

  auto lpmLocalKernelPredicate = [](const std::string &tag) {
    return tag.find("lpm_t::") != std::string::npos && tag.find("localKernel") != std::string::npos;
  };

  auto lpmLocalEvalKernelPredicate = [](const std::string &tag) {
    return tag.find("lpm_t::") != std::string::npos && tag.find("localEvalKernel") != std::string::npos;
  };

  auto neknekLocalKernelPredicate = [](const std::string &tag) {
    return tag.find("neknek_t::") != std::string::npos && tag.find("localKernel") != std::string::npos;
  };

  auto neknekLocalEvalKernelPredicate = [](const std::string &tag) {
    return tag.find("neknek_t::") != std::string::npos && tag.find("localEvalKernel") != std::string::npos;
  };

  platform->timer.printStatEntry("    checkpointing       ",
                                 "checkpointing",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);
  platform->timer.printStatEntry("    udfExecuteStep      ",
                                 "udfExecuteStep",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);
  const double tudf = platform->timer.query("udfExecuteStep", "DEVICE:MAX");
  platform->timer.printStatEntry("      lpm integrate     ", "lpm_t::integrate", "DEVICE:MAX", tudf);
  const double tlpm = platform->timer.query("lpm_t::integrate", "DEVICE:MAX");
  platform->timer.printStatEntry("        userRHS         ", "lpm_t::integrate::userRHS", "DEVICE:MAX", tlpm);
  const double tParticleRHS = platform->timer.query("lpm_t::integrate::userRHS", "DEVICE:MAX");
  platform->timer.printStatEntry("          interpolate   ",
                                 "lpm_t::integrate::userRHS::interpolate",
                                 "DEVICE:MAX",
                                 tParticleRHS);
  const double tInterpPart = platform->timer.query("lpm_t::integrate::userRHS::interpolate", "DEVICE:MAX");
  auto [tLocalKernel, nLocalKernel] =
      platform->timer.sumAllMatchingTags(lpmLocalEvalKernelPredicate, "DEVICE:MAX");
  platform->timer.printStatEntry("            eval kernel ", tLocalKernel, nLocalKernel, tInterpPart);
  platform->timer.printStatEntry("        findpts         ", "lpm_t::integrate::find", "DEVICE:MAX", tlpm);
  const double tFindPart = platform->timer.query("lpm_t::integrate::find", "DEVICE:MAX");
  auto [tFindKernel, nFindKernel] = platform->timer.sumAllMatchingTags(lpmLocalKernelPredicate, "DEVICE:MAX");
  platform->timer.printStatEntry("          find kernel   ", tFindKernel, nFindKernel, tFindPart);
  platform->timer.printStatEntry("        delete          ", "lpm_t::deleteParticles", "DEVICE:MAX", tlpm);
  platform->timer.printStatEntry("      lpm add           ", "lpm_t::addParticles", "DEVICE:MAX", tudf);
  platform->timer.printStatEntry("      lpm write         ", "lpm_t::write", "DEVICE:MAX", tudf);

  const double tDiv = platform->timer.query("udfDiv", "DEVICE:MAX");
  platform->timer.printStatEntry("    udfDiv              ", "udfDiv", "DEVICE:MAX", tElapsedTimeSolve);

  const double tMakef = platform->timer.query("makef", "DEVICE:MAX");
  platform->timer.printStatEntry("    makef               ", "makef", "DEVICE:MAX", tElapsedTimeSolve);
  platform->timer.printStatEntry("      udfUEqnSource     ", "udfUEqnSource", "DEVICE:MAX", tMakef);

  const double tMakeq = platform->timer.query("makeq", "DEVICE:MAX");
  platform->timer.printStatEntry("    makeq               ", "makeq", "DEVICE:MAX", tElapsedTimeSolve);
  platform->timer.printStatEntry("      udfSEqnSource     ", "udfSEqnSource", "DEVICE:MAX", tMakeq);

  platform->timer.printStatEntry("    udfProperties       ",
                                 "udfProperties",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);

  const double tMesh = platform->timer.query("geomSolve", "DEVICE:MAX");
  platform->timer.printStatEntry("    geomSolve           ", "geomSolve", "DEVICE:MAX", tElapsedTimeSolve);
  platform->timer.printStatEntry("      preconditioner    ", "geom preconditioner", "DEVICE:MAX", tMesh);

  platform->timer.set("geom proj",
                      platform->timer.query("geom proj pre", "DEVICE:MAX") +
                          platform->timer.query("geom proj post", "DEVICE:MAX"),
                      platform->timer.count("geom proj pre"));
  platform->timer.printStatEntry("      initial guess     ", "geom proj", "DEVICE:MAX", tMesh);

  const double tNekNek = platform->timer.query("neknek update boundary", "DEVICE:MAX");
  platform->timer.printStatEntry("    neknek              ",
                                 "neknek update boundary",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);
  platform->timer.printStatEntry("      sync              ", "neknek sync", "DEVICE:MAX", tNekNek);
  platform->timer.printStatEntry("      exchange          ", "neknek exchange", "DEVICE:MAX", tNekNek);
  const double tExchange = platform->timer.query("neknek exchange", "DEVICE:MAX");
  std::tie(tLocalKernel, nLocalKernel) =
      platform->timer.sumAllMatchingTags(neknekLocalEvalKernelPredicate, "DEVICE:MAX");
  platform->timer.printStatEntry("        eval kernel     ", tLocalKernel, nLocalKernel, tExchange);
  platform->timer.printStatEntry("      findpts           ",
                                 "neknek updateInterpPoints",
                                 "DEVICE:MAX",
                                 tNekNek);
  const double tFindpts = platform->timer.query("neknek updateInterpPoints", "DEVICE:MAX");

  if (tFindpts > 0.0) {
    std::tie(tFindKernel, nFindKernel) =
        platform->timer.sumAllMatchingTags(neknekLocalKernelPredicate, "DEVICE:MAX");
    platform->timer.printStatEntry("        find kernel     ", tFindKernel, nFindKernel, tFindpts);
  }

  const double tVelocity = platform->timer.query("fluid velocitySolve", "DEVICE:MAX");

  platform->timer.printStatEntry("    velocitySolve       ",
                                 "fluid velocitySolve",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);
  platform->timer.printStatEntry("      rhs               ", "fluid velocity rhs", "DEVICE:MAX", tVelocity);
  platform->timer.printStatEntry("      preconditioner    ",
                                 "fluid velocity preconditioner",
                                 "DEVICE:MAX",
                                 tVelocity);

  platform->timer.set("fluid velocity proj",
                      platform->timer.query("fluid velocity proj pre", "DEVICE:MAX") +
                          platform->timer.query("fluid velocity proj post", "DEVICE:MAX"),
                      platform->timer.count("fluid velocity proj pre"));
  platform->timer.printStatEntry("      initial guess     ", "fluid velocity proj", "DEVICE:MAX", tVelocity);

  const double tPressure = platform->timer.query("fluid pressureSolve", "DEVICE:MAX");
  platform->timer.printStatEntry("    pressureSolve       ",
                                 "fluid pressureSolve",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);
  platform->timer.printStatEntry("      rhs               ", "fluid pressure rhs", "DEVICE:MAX", tPressure);

  const double tPressurePreco = platform->timer.query("fluid pressure preconditioner", "DEVICE:MAX");
  platform->timer.printStatEntry("      preconditioner    ",
                                 "fluid pressure preconditioner",
                                 "DEVICE:MAX",
                                 tPressure);

  auto tags = platform->timer.tags();
  for (int i = 15; i > 0; i--) {
    const std::string tag = "fluid pressure preconditioner smoother N=" + std::to_string(i);
    if (std::find(tags.begin(), tags.end(), tag) == tags.end()) {
      continue;
    }
    platform->timer.printStatEntry("        pMG smoother    ", tag, "DEVICE:MAX", tPressurePreco);
  }

  platform->timer.printStatEntry("        coarse grid     ",
                                 "fluid pressure coarseSolve",
                                 "HOST:MAX",
                                 tPressurePreco);

  platform->timer.set("fluid pressure proj",
                      platform->timer.query("fluid pressure proj pre", "DEVICE:MAX") +
                          platform->timer.query("fluid pressure proj post", "DEVICE:MAX"),
                      platform->timer.count("fluid pressure proj pre"));
  platform->timer.printStatEntry("      initial guess     ", "fluid pressure proj", "DEVICE:MAX", tPressure);

  int nScalar = 0;
  platform->options.getArgs("NUMBER OF SCALARS", nScalar);

  const double tScalar = platform->timer.query("scalarSolve", "DEVICE:MAX");
  platform->timer.printStatEntry("    scalarSolve         ", "scalarSolve", "DEVICE:MAX", tElapsedTimeSolve);
  platform->timer.printStatEntry("      rhs               ", "scalar rhs", "DEVICE:MAX", tScalar);

  auto cvodeMakeQPredicate = [](const std::string &tag) {
    bool match = tag.find("cvode_t::") != std::string::npos && tag.find("makeq") != std::string::npos;
    // ensure children of the timer aren't doubly counted
    return match && tag.find("makeq::") == std::string::npos;
  };
  auto [tMakeqCvode, nMakeqCvode] = platform->timer.sumAllMatchingTags(cvodeMakeQPredicate, "DEVICE:MAX");

  auto cvodeUdfSEqnSourcePredicate = [](const std::string &tag) {
    bool match = tag.find("cvode_t::") != std::string::npos && tag.find("udfSEqnSource") != std::string::npos;
    // ensure children of the timer aren't doubly counted
    return match && tag.find("udfSEqnSource::") == std::string::npos;
  };
  auto [tSEqnSourceCvode, nSEqnSourceCvode] =
      platform->timer.sumAllMatchingTags(cvodeUdfSEqnSourcePredicate, "DEVICE:MAX");

  auto cvodeLocalPointSourcePredicate = [](const std::string &tag) {
    bool match = tag.find("cvode_t::") != std::string::npos && tag.find("pointSource") != std::string::npos;
    // ensure children of the timer aren't doubly counted
    return match;
  };
  auto [tLocalPointSource, nLocalPointSource] =
      platform->timer.sumAllMatchingTags(cvodeLocalPointSourcePredicate, "DEVICE:MAX");

  auto cvodePropertiesPredicate = [](const std::string &tag) {
    bool match =
        tag.find("cvode_t::") != std::string::npos && tag.find("evaluateProperties") != std::string::npos;
    // ensure children of the timer aren't doubly counted
    return match && tag.find("evaluateProperties::") == std::string::npos;
  };
  auto [tPropCvode, nPropCvode] = platform->timer.sumAllMatchingTags(cvodePropertiesPredicate, "DEVICE:MAX");
  platform->timer.printStatEntry("    scalarSolveCvode    ",
                                 "cvode_t::solve",
                                 "DEVICE:MAX",
                                 tElapsedTimeSolve);
  platform->timer.printStatEntry("      makeq             ", tMakeqCvode, nMakeqCvode, tScalarCvode);
  platform->timer.printStatEntry("        udfSEqnSource   ", tSEqnSourceCvode, nSEqnSourceCvode, tMakeqCvode);
  platform->timer.printStatEntry("      local pt src      ",
                                 tLocalPointSource,
                                 nLocalPointSource,
                                 tScalarCvode);
  platform->timer.printStatEntry("      udfProperties     ", tPropCvode, nPropCvode, tScalarCvode);

  auto precoTimeScalars = 0.0;
  auto precoCallsScalars = 0;
  auto projTimeScalars = 0.0;
  auto projCallsScalars = 0;
  for (int is = 0; is < nScalar; is++) {
    std::string sid = scalarDigitStr(is);
    precoTimeScalars += platform->timer.query("scalar" + sid + " preconditioner", "DEVICE:MAX");
    precoCallsScalars += platform->timer.count("scalar" + sid + " preconditioner");
    projTimeScalars += platform->timer.query("scalar" + sid + " proj pre", "DEVICE:MAX") +
                       platform->timer.query("scalar" + sid + " proj post", "DEVICE:MAX");
    projCallsScalars += platform->timer.count("scalar" + sid + " proj pre");
  }
  platform->timer.set("scalar preconditioner", precoTimeScalars, precoCallsScalars);
  platform->timer.set("scalar proj", projTimeScalars, projCallsScalars);

  platform->timer.printStatEntry("      preconditioner    ", "scalar preconditioner", "DEVICE:MAX", tScalar);
  platform->timer.printStatEntry("      initial guess     ", "scalar proj", "DEVICE:MAX", tScalar);

  platform->timer.printStatEntry("    gsMPI               ", gsTime, tElapsedTimeSolve);

  platform->timer.printStatEntry("    dotp                ", "dotp", "DEVICE:MAX", tElapsedTimeSolve);

  platform->timer.printStatEntry("    dotp multi          ", "dotpMulti", "DEVICE:MAX", tElapsedTimeSolve);

  if (platform->comm.mpiRank() == 0) {
    std::cout << std::endl;
  }
  platform->device.printMemoryUsage(platform->comm.mpiComm());
  if (platform->comm.mpiRank() == 0) {
    std::cout << std::endl;
  }

  std::cout.unsetf(std::ios::scientific);
  std::cout.precision(outPrecisionSave);
}

void nrs_t::finalize()
{
  if (fluid) {
    fluid->finalize();
    fluid.reset();
  }

  if (scalar) {
    scalar->finalize();
    scalar.reset();
  }

  if (geom) {
    geom->finalize();
    geom.reset();
  }

  checkpointWriter.reset();
}

void nrs_t::printStepInfo(double time, int tstep, bool printStepInfo, bool solverInfo)
{
  const double elapsedStep = platform->timer.query("elapsedStep", "DEVICE:MAX");
  const double elapsedStepSum = platform->timer.query("elapsedStepSum", "DEVICE:MAX");
  const auto cfl = computeCFL();

  auto printSolverInfo = [tstep](elliptic *solver, const std::string &name) {
    if (!solver) {
      return;
    }
    const auto [prevProjVecs, nProjVecs] = solver->projectionCounters();
    if (nProjVecs > 0) {
      if (prevProjVecs > 0) {
        printf("step=%-8d %-20s: resNorm0 %.2e  resNorm %.2e  ratio = %.3e  %d/%d\n",
               tstep,
               std::string("proj " + name).c_str(),
               solver->initialResidual(),
               solver->initialGuessResidual(),
               solver->initialResidual() / solver->initialGuessResidual(),
               prevProjVecs,
               nProjVecs);
      }
    }
    printf("step=%-8d %-20s: iter %03d  resNorm0 %.2e  resNorm %.2e\n",
           tstep,
           name.c_str(),
           solver->Niter(),
           solver->initialGuessResidual(),
           solver->finalResidual());
  };

  const auto [divUErrVolAvg, divUErrL2] = [&]() {
    auto mesh = fluid->mesh;

    auto o_divErr = opSEM::strongDivergence(mesh, fluid->fieldOffset, fluid->o_U);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, fluid->o_div, -1.0, o_divErr);

    const auto L1 =
        std::abs(platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, o_divErr, platform->comm.mpiComm()));
    const auto L2 =
        platform->linAlg->weightedNorm2(mesh->Nlocal, mesh->o_LMM, o_divErr, platform->comm.mpiComm());

    return std::make_tuple(L1 / mesh->volume, L2 / sqrt(mesh->volume));
  }();

  if (platform->comm.mpiRank() == 0) {
    if (solverInfo) {
      if (neknek) {
        printf("step=%-8d %-20s: sync %.2e  exchange %.2e\n",
               tstep,
               "neknek",
               neknek->tSync(),
               neknek->tExch());
      }

      bool cvodePrinted = false;
      for (int is = 0; is < Nscalar; is++) {
        if (scalar->compute[is] && !scalar->cvodeSolve[is]) {
          const auto sid = scalarDigitStr(is);
          printSolverInfo(scalar->ellipticSolver.at(is), "SCALAR " + scalar->name[is]);
        } else if (scalar->cvodeSolve[is] && !cvodePrinted) {
          scalar->cvode->printInfo(true);
          cvodePrinted = true;
        }
      }

      if (!platform->options.compareArgs("FLUID VELOCITY SOLVER", "NONE")) {
        printSolverInfo(fluid->ellipticSolverP, "FLUID p");

        if (fluid->ellipticSolver.size() == 1) {
          printSolverInfo(fluid->ellipticSolver.at(0), "FLUID U");
        } else if (fluid->ellipticSolver.size() == 3) {
          printSolverInfo(fluid->ellipticSolver.at(0), "FLUID ux");
          printSolverInfo(fluid->ellipticSolver.at(1), "FLUID uy");
          printSolverInfo(fluid->ellipticSolver.at(2), "FLUID uz");
        }

        printf("step=%-8d %-20s: %.2e  %.2e\n", tstep, "FLUID divUErr", divUErrVolAvg, divUErrL2);
      }

      if (geom) {
        if (geom->ellipticSolver.size()) {
          printSolverInfo(geom->ellipticSolver.at(0), "GEOM U");
        }
      }
    }

    if (platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")) {
      flowRatePrintInfo(tstep, solverInfo);
    }

    const auto printTimers = printStepInfo && timeStepConverged;

    if (printStepInfo) {
      printf("step=%-8d t= %.8e  dt=%.1e  CFL= %.3f\n", tstep, time, dt[0], cfl);
    }

    if (printTimers) {
      printf("step=%-8d elapsedStep= %.2es  elapsedStepSum= %.5es\n", tstep, elapsedStep, elapsedStepSum);
    }
  }

  if (scalar) {
    if (scalar->cvode) {
      scalar->cvode->resetCounters();
    }
  }

  const auto cflTooLarge = (cfl > 100) && numberActiveFields();
  if (!platform->options.compareArgs("CHECK FLUID CFL", "FALSE")) {
    nekrsCheck(cflTooLarge || std::isnan(cfl) || std::isinf(cfl),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "Unreasonable FLUID CFL (value: %g)!\n",
               cfl);
  }
}

void nrs_t::writeToFile(const std::string &fileName_,
                        double time,
                        mesh_t *mesh_,
                        const std::vector<std::tuple<std::string, std::vector<deviceMemory<dfloat>>>> &list,
                        bool enforceOutXYZ,
                        bool enforceFP64,
                        int N_,
                        bool uniform)
{
  auto fileNameEndsWithBp = [&]() {
    const std::string suffix = ".bp";
    if (fileName_.size() >= suffix.size()) {
      return fileName_.compare(fileName_.size() - suffix.size(), suffix.size(), suffix) == 0;
    }
    return false;
  }();

  auto iofld = iofldFactory::create((fileNameEndsWithBp) ? "adios" : "");
  iofld->open(mesh_, iofld::mode::write, fileName_);

  iofld->addVariable("time", time);
  for (const auto &[name, o_u] : list) {
    iofld->addVariable(name, o_u);
  }

  const auto outXYZ =
      (enforceOutXYZ) ? true : platform->options.compareArgs("CHECKPOINT OUTPUT MESH", "TRUE");

  const auto Nfld = [&]() {
    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);
    return (N_) ? N_ : N;
  }();
  iofld->writeAttribute("polynomialOrder", std::to_string(Nfld));

  auto FP64 = platform->options.compareArgs("CHECKPOINT PRECISION", "FP64");
  if (enforceFP64) {
    FP64 = true;
  }
  iofld->writeAttribute("precision", (FP64) ? "64" : "32");
  iofld->writeAttribute("uniform", (uniform) ? "true" : "false");
  iofld->writeAttribute("outputMesh", (outXYZ) ? "true" : "false");

  iofld->process();
  iofld->close();
}

void nrs_t::writeCheckpoint(double t, bool enforceOutXYZ, bool enforceFP64, int N_, bool uniform)
{
  if (!checkpointWriter) {
    checkpointWriter = iofldFactory::create();
  }

  const auto outXYZ =
      (enforceOutXYZ) ? true : platform->options.compareArgs("CHECKPOINT OUTPUT MESH", "TRUE");

  if (!checkpointWriter->isInitialized()) {
    auto visMesh = meshT;
    checkpointWriter->open(visMesh, iofld::mode::write, platform->options.getArgs("CASENAME"));

    if (platform->options.compareArgs("LOWMACH", "TRUE")) {
      checkpointWriter->addVariable("p0th", p0th[0]);
    }

    if (fluid) {
      if (platform->options.compareArgs(upperCase(fluid->name) + " CHECKPOINTING", "TRUE")) {
        std::vector<occa::memory> o_V;
        for (int i = 0; i < meshV->dim; i++) {
          o_V.push_back(fluid->o_U.slice(i * fluid->fieldOffset, visMesh->Nlocal));
        }
        checkpointWriter->addVariable("velocity", o_V);

        auto o_p = std::vector<occa::memory>{fluid->o_P.slice(0, visMesh->Nlocal)};
        checkpointWriter->addVariable("pressure", o_p);
      }
    }

    for (int i = 0; i < Nscalar; i++) {
      if (platform->options.compareArgs("SCALAR" + scalarDigitStr(i) + " CHECKPOINTING", "TRUE")) {
        const auto temperatureExists = scalar->nameToIndex.find("temperature") != scalar->nameToIndex.end();
        std::vector<occa::memory> o_Si = {scalar->o_S.slice(scalar->fieldOffsetScan[i], visMesh->Nlocal)};
        if (i == 0 && temperatureExists) {
          checkpointWriter->addVariable("temperature", o_Si);
        } else {
          const auto is = (temperatureExists) ? i - 1 : i;
          checkpointWriter->addVariable("scalar" + scalarDigitStr(is), o_Si);
        }
      }
    }
  }

  const auto Nfld = [&]() {
    int N;
    platform->options.getArgs("POLYNOMIAL DEGREE", N);
    return (N_) ? N_ : N;
  }();
  checkpointWriter->writeAttribute("polynomialOrder", std::to_string(Nfld));

  auto FP64 = platform->options.compareArgs("CHECKPOINT PRECISION", "FP64");
  if (enforceFP64) {
    FP64 = true;
  }
  checkpointWriter->writeAttribute("precision", (FP64) ? "64" : "32");
  checkpointWriter->writeAttribute("uniform", (uniform) ? "true" : "false");
  checkpointWriter->writeAttribute("outputMesh", (outXYZ) ? "true" : "false");

  checkpointWriter->addVariable("time", t);

  for (const auto &entry : userCheckpointFields) {
    checkpointWriter->addVariable(entry.first, entry.second);
  }

  checkpointWriter->process();
}

int nrs_t::lastStepLocalSession(double timeNew, int tstep, double elapsedTime)
{
  double endTime = -1;
  platform->options.getArgs("END TIME", endTime);

  int numSteps = -1;
  platform->options.getArgs("NUMBER TIMESTEPS", numSteps);

  int last = 0;
  if (!platform->options.getArgs("STOP AT ELAPSED TIME").empty()) {
    double maxElaspedTime;
    platform->options.getArgs("STOP AT ELAPSED TIME", maxElaspedTime);
    if (elapsedTime > 60.0 * maxElaspedTime) {
      last = 1;
    }
  } else if (endTime >= 0) {
    const double eps = 1e-10;
    last = fabs(timeNew - endTime) < eps || timeNew > endTime;
  } else {
    last = (tstep == numSteps);
  }
  return last;
}

int nrs_t::setLastStep(double timeNew, int tstep, double elapsedTime)
{
  int last = lastStepLocalSession(timeNew, tstep, elapsedTime);

  lastStep = last;
  return last;
}

void nrs_t::copyToNek(double time, int tstep, bool updateMesh)
{
  *(nekData.istep) = tstep;
  copyToNek(time, updateMesh);
}

void nrs_t::copyToNek(double time, bool updateMesh_)
{
  if (platform->comm.mpiRank() == 0) {
    printf("copying solution to nek\n");
    fflush(stdout);
  }

  *(nekData.time) = time;
  *(nekData.p0th) = p0th[0];

  auto updateMesh = [&]() {
    auto mesh = meshT;

    auto [x, y, z] = mesh->xyzHost();
    for (int i = 0; i < mesh->Nlocal; i++) {
      nekData.xm1[i] = x[i];
      nekData.ym1[i] = y[i];
      nekData.zm1[i] = z[i];
    }
    nek::recomputeGeometry();
  };

  if (fluid) {
    auto U = platform->memoryPool.reserve<dfloat>(fluid->fieldOffsetSum);
    fluid->o_U.copyTo(U, U.size());
    auto vx = U.ptr<dfloat>() + 0 * fluid->fieldOffset;
    auto vy = U.ptr<dfloat>() + 1 * fluid->fieldOffset;
    auto vz = U.ptr<dfloat>() + 2 * fluid->fieldOffset;
    for (int i = 0; i < fluid->mesh->Nlocal; i++) {
      nekData.vx[i] = vx[i];
      nekData.vy[i] = vy[i];
      nekData.vz[i] = vz[i];
    }
  }

  if (geom) {
    auto U = platform->memoryPool.reserve<dfloat>(geom->fieldOffsetSum);
    geom->o_U.copyTo(U, U.size());
    auto wx = U.ptr<dfloat>() + 0 * geom->fieldOffset;
    auto wy = U.ptr<dfloat>() + 1 * geom->fieldOffset;
    auto wz = U.ptr<dfloat>() + 2 * geom->fieldOffset;
    for (int i = 0; i < geom->mesh->Nlocal; i++) {
      nekData.wx[i] = wx[i];
      nekData.wy[i] = wy[i];
      nekData.wz[i] = wz[i];
    }
    updateMesh_ = true;
  }

  if (updateMesh_) {
    updateMesh();
  }

  if (fluid) {
    auto P = platform->memoryPool.reserve<dfloat>(fluid->mesh->Nlocal);
    fluid->o_P.copyTo(P, P.size());
    auto Pptr = P.ptr<dfloat>();
    for (int i = 0; i < fluid->mesh->Nlocal; i++) {
      nekData.pr[i] = Pptr[i];
    }
  }

  if (Nscalar) {
    const dlong nekFieldOffset = nekData.lelt * std::pow(nekData.nx1, nekData.ndim);
    for (int is = 0; is < Nscalar; is++) {
      auto mesh = scalar->mesh(is);

      auto S = platform->memoryPool.reserve<dfloat>(mesh->Nlocal);
      scalar->o_S.copyTo(S, S.size(), 0, scalar->fieldOffsetScan[is]);

      auto Sptr = S.ptr<dfloat>();
      auto Ti = nekData.t + is * nekFieldOffset;
      for (int i = 0; i < mesh->Nlocal; i++) {
        Ti[i] = Sptr[i];
      }
    }
  }
}

void nrs_t::copyFromNek()
{
  double time; // dummy
  copyFromNek(time);
}

void nrs_t::copyFromNek(double &time)
{
  if (platform->comm.mpiRank() == 0) {
    printf("copying solution from nek\n");
    fflush(stdout);
  }

  time = *(nekData.time);
  p0th[0] = *(nekData.p0th);

  if (fluid) {
    auto U = platform->memoryPool.reserve<dfloat>(fluid->fieldOffsetSum);
    auto vx = U.ptr<dfloat>() + 0 * fluid->fieldOffset;
    auto vy = U.ptr<dfloat>() + 1 * fluid->fieldOffset;
    auto vz = U.ptr<dfloat>() + 2 * fluid->fieldOffset;
    for (int i = 0; i < fluid->mesh->Nlocal; i++) {
      vx[i] = nekData.vx[i];
      vy[i] = nekData.vy[i];
      vz[i] = nekData.vz[i];
    }
    fluid->o_U.copyFrom(U, U.size());
  }

  if (geom) {
    auto U = platform->memoryPool.reserve<dfloat>(geom->fieldOffsetSum);
    auto wx = U.ptr<dfloat>() + 0 * geom->fieldOffset;
    auto wy = U.ptr<dfloat>() + 1 * geom->fieldOffset;
    auto wz = U.ptr<dfloat>() + 2 * geom->fieldOffset;
    for (int i = 0; i < geom->mesh->Nlocal; i++) {
      wx[i] = nekData.wx[i];
      wy[i] = nekData.wy[i];
      wz[i] = nekData.wz[i];
    }
    geom->o_U.copyFrom(U, U.size());
  }

  if (fluid) {
    auto P = platform->memoryPool.reserve<dfloat>(fluid->o_P.size());
    auto Pptr = P.ptr<dfloat>();
    for (int i = 0; i < fluid->mesh->Nlocal; i++) {
      Pptr[i] = nekData.pr[i];
    }
    fluid->o_P.copyFrom(P, P.size());
  }

  if (Nscalar) {
    const dlong nekFieldOffset = nekData.lelt * std::pow(nekData.nx1, nekData.ndim);
    for (int is = 0; is < Nscalar; is++) {
      auto mesh = scalar->mesh(is);
      auto Ti = nekData.t + is * nekFieldOffset;

      auto S = platform->memoryPool.reserve<dfloat>(mesh->Nlocal);

      auto Sptr = S.ptr<dfloat>();
      for (int i = 0; i < mesh->Nlocal; i++) {
        Sptr[i] = Ti[i];
      }
      scalar->o_S.copyFrom(S, S.size(), scalar->fieldOffsetScan[is], 0);
    }
  }
}

void nrs_t::getICFromNek()
{
  nek::getIC();
  copyFromNek();
}

void nrs_t::setTimeIntegrationCoeffs(int tstep)
{
  const auto bdfOrder = std::min(tstep, static_cast<int>(o_coeffBDF.size()));
  const auto extOrder = std::min(tstep, static_cast<int>(o_coeffEXT.size()));

  {
    std::vector<dfloat> coeff(o_coeffBDF.size());
    nek::bdfCoeff(&g0, coeff.data(), dt, bdfOrder);
    for (int i = coeff.size(); i > bdfOrder; i--) {
      coeff[i - 1] = 0;
    }
    o_coeffBDF.copyFrom(coeff.data());
  }

  {
    std::vector<dfloat> coeff(o_coeffEXT.size());
    nek::extCoeff(coeff.data(), dt, extOrder, bdfOrder);
    for (int i = coeff.size(); i > extOrder; i--) {
      coeff[i - 1] = 0;
    }
    o_coeffEXT.copyFrom(coeff.data());
  }
}

dfloat nrs_t::adjustDt(int tstep)
{
  static auto firstTime = true;
  static dfloat unitTimeCFL;
  static dfloat CFL;

  const double TOLToZero = 1e-6;

  double dt_ = -1;

  dfloat targetCFL;
  platform->options.getArgs("TARGET CFL", targetCFL);

  if (tstep == 1) {
    unitTimeCFL = computeCFL(1.0);
    CFL = unitTimeCFL;

    if (unitTimeCFL > TOLToZero) {
      dt_ = targetCFL / unitTimeCFL;
    } else {
      if (fluid && userSource) {
        auto &mesh = fluid->mesh;

        const auto absRhoFMax = [&]() {
          double startTime;
          platform->options.getArgs("START TIME", startTime);

          platform->linAlg->fill(fluid->fieldOffsetSum, 0.0, fluid->o_EXT);
          platform->timer.tic("udfUEqnSource");
          userSource(startTime);
          platform->timer.toc("udfUEqnSource");

          platform->linAlg->abs(fluid->o_EXT.size(), fluid->o_EXT);

          occa::memory o_FUx = fluid->o_EXT + 0 * fluid->fieldOffset;
          occa::memory o_FUy = fluid->o_EXT + 1 * fluid->fieldOffset;
          occa::memory o_FUz = fluid->o_EXT + 2 * fluid->fieldOffset;

          const auto maxFUx = platform->linAlg->max(mesh->Nlocal, o_FUx, platform->comm.mpiComm());
          const auto maxFUy = platform->linAlg->max(mesh->Nlocal, o_FUy, platform->comm.mpiComm());
          const auto maxFUz = platform->linAlg->max(mesh->Nlocal, o_FUz, platform->comm.mpiComm());

          return std::max({maxFUx, maxFUy, maxFUz}) /
                 platform->linAlg->min(mesh->Nlocal, fluid->o_rho, platform->comm.mpiComm());
        }();
        nekrsCheck(absRhoFMax <= TOLToZero,
                   platform->comm.mpiComm(),
                   EXIT_FAILURE,
                   "%s\n",
                   "Zero velocity and body force! Please specify an initial timestep!");

        const auto lengthScale = [&]() {
          std::vector<dfloat> Jw(mesh->Nlocal);
          mesh->o_Jw.copyTo(Jw.data(), Jw.size());

          auto scale = std::numeric_limits<dfloat>::max();
          for (int i = 0; i < Jw.size(); i++) {
            scale = std::min(std::cbrt(Jw[i]), scale);
          }
          MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiComm());
          return scale;
        }();

        dt_ = sqrt(targetCFL * lengthScale / absRhoFMax);
      }
    }
    firstTime = false;
    return dt_;
  }

  dt_ = dt[0];
  const auto dtOld = dt[0];

  auto CFLold = CFL;
  CFL = computeCFL();
  if (firstTime) {
    CFLold = CFL;
  }

  auto unitTimeCFLold = unitTimeCFL;
  unitTimeCFL = CFL / dtOld;
  if (firstTime) {
    unitTimeCFLold = unitTimeCFL;
  }

  const auto CFLpred = 2.0 * CFL - CFLold;
  const auto CFLmax = 1.2 * targetCFL;
  const auto CFLmin = 0.8 * targetCFL;

  if (CFL > CFLmax || CFLpred > CFLmax || CFL < CFLmin) {
    const double A = (unitTimeCFL - unitTimeCFLold) / dtOld;
    const double B = unitTimeCFL;
    const double C = -targetCFL;
    const double descriminant = B * B - 4 * A * C;

    const dfloat TOL = 1e-3;

    if (descriminant <= 0.0) {
      dt_ = dtOld * (targetCFL / CFL);
    } else if (std::abs((unitTimeCFL - unitTimeCFLold) / unitTimeCFL) < TOL) {
      dt_ = dtOld * (targetCFL / CFL);
    } else {
      const double dtLow = (-B + sqrt(descriminant)) / (2.0 * A);
      const double dtHigh = (-B - sqrt(descriminant)) / (2.0 * A);
      if (dtHigh > 0.0 && dtLow > 0.0) {
        dt_ = std::min(dtLow, dtHigh);
      } else if (dtHigh <= 0.0 && dtLow <= 0.0) {
        dt_ = dtOld * targetCFL / CFL;
      } else {
        dt_ = std::max(dtHigh, dtLow);
      }
    }
  }
  firstTime = false;

  if (platform->verbose() && platform->comm.mpiRank() == 0) {
    printf("adjustDt: dt=%g CFL= %g CFLpred= %g CFLmax= %g CFLmin= %g\n", dt_, CFL, CFLpred, CFLmax, CFLmin);
  }

  return dt_;
}

void nrs_t::initStep(double time, dfloat _dt, int _tstep)
{
  if (platform->options.compareArgs("NEKNEK MULTIRATE TIMESTEPPER", "TRUE")) {
    initOuterStep(time, _dt, _tstep);
  } else {
    initInnerStep(time, _dt, _tstep);
  }
}

void nrs_t::initInnerStep(double time, dfloat _dt, int _tstep)
{
  timePrevious = time;
  dt[0] = _dt;
  nekrsCheck(dt[0] <= 0 || std::isnan(dt[0]) || std::isinf(dt[0]),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "Unreasonable dt!\n");

  tstep = _tstep;

  setTimeIntegrationCoeffs(tstep);

  if (fluid) {
    fluid->setTimeIntegrationCoeffs(tstep);
    fluid->extrapolateSolution();
  }

  if (geom) {
    geom->setTimeIntegrationCoeffs(tstep);
    geom->extrapolateSolution();
  }

  if (scalar) {
    scalar->extrapolateSolution();
  }

  computeUrst();

  if (geom && advectionSubcycingSteps) { // used in makeAdvection
    geom->computeDiv();
  }

  if (scalar) {
    if (scalar->anyEllipticSolver) {
      platform->linAlg->fill(scalar->fieldOffsetSum, 0.0, scalar->o_EXT);
    }
  }

  if (fluid) {
    platform->linAlg->fill(fluid->fieldOffsetSum, 0.0, fluid->o_EXT);
  }

  if (userSource) {
    platform->timer.tic("udfSource");
    userSource(time);
    platform->timer.toc("udfSource");
  }

  if (Nscalar) {
    if (scalar->anyEllipticSolver) {
      platform->timer.tic("makeq");

      for (int is = 0; is < Nscalar; is++) {
        if (!scalar->compute[is] || scalar->cvodeSolve[is]) {
          continue;
        }
        if (platform->options.compareArgs("EQUATION TYPE", "NAVIERSTOKES")) {
          scalar->makeAdvection(is, time, tstep);
        }
        scalar->makeExplicit(is, time, tstep);
      }
      scalar->makeForcing();

      platform->timer.toc("makeq");
    }
  }

  if (fluid) {
    platform->timer.tic("makef");

    if (platform->options.compareArgs("EQUATION TYPE", "NAVIERSTOKES")) {
      fluid->makeAdvection(time, tstep);
    }
    fluid->makeExplicit(time, tstep);
    fluid->makeForcing();

    platform->timer.toc("makef");
  }
}

void nrs_t::finishStep()
{
  if (platform->options.compareArgs("NEKNEK MULTIRATE TIMESTEPPER", "TRUE")) {
    finishOuterStep();
  } else {
    finishInnerStep();
  }
}

void nrs_t::finishInnerStep()
{
  dt[2] = dt[1];
  dt[1] = dt[0];
}

bool nrs_t::runStep(std::function<bool(int)> convergenceCheck, int iter)
{
  timeStepConverged = false;

  if (platform->options.compareArgs("NEKNEK MULTIRATE TIMESTEPPER", "TRUE")) {
    runOuterStep(convergenceCheck, iter);
  } else {
    runInnerStep(convergenceCheck, iter, true);
  }

  return timeStepConverged;
}

bool nrs_t::runInnerStep(std::function<bool(int)> convergenceCheck, int iter, bool outerConverged)
{
  outerCorrector = iter;

  const auto timeNew = timePrevious + setPrecision(this->dt[0], 5);

  const auto checkpointStep0 = checkpointStep;

  if (geom && iter == 1) {
    geom->integrate();

    if (fluid) {
      fluid->updateZeroNormalMask();
    }
  }

  if (neknek) {
    neknek->updateBoundary(tstep, iter, dt, timeNew);
  }

  if (scalar) {
    if (scalar->cvode && iter == 1) {
      scalar->cvode->solve(timePrevious, timeNew, tstep);
    }
  }

  if (iter == 1) {
    if (fluid) {
      fluid->lagSolution();
    }

    if (scalar) {
      scalar->lagSolution();
    }

    if (geom) {
      geom->lagSolution();
    }
  }

  if (fluid) {
    fluid->applyDirichlet(timeNew);
    if (neknek && fieldsToSolveContains("fluid velocity")) {
      if (!platform->app->bc->hasOutflow("fluid velocity")) {
        neknek->fixCoupledSurfaceFlux(fluid->o_EToB, fluid->fieldOffset, fluid->o_U);
      }
    }
  }

  if (Nscalar) {
    scalar->applyDirichlet(timeNew);
  }

  if (geom) {
    geom->applyDirichlet(timeNew);
  }

  if (Nscalar) {
    scalar->solve(timeNew, iter);
  }

  if (postScalar) {
    postScalar(timeNew, tstep);
  }

  evaluateProperties(timeNew);

  evaluateDivergence(timeNew);

  if (preFluid) {
    preFluid(timeNew, tstep);
  }

  if (fluid) {
    fluid->solve(timeNew, iter);

    if (platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")) {
      adjustFlowRate(tstep, timeNew);
    }
  }

  if (geom) {
    geom->solve(timeNew, iter);
  }

  const auto converged = convergenceCheck(iter);

  timeStepConverged = outerConverged && converged;

  nek::ifoutfld(0);
  checkpointStep = 0;
  if (checkpointStep0 && timeStepConverged) {
    nek::ifoutfld(1);
    checkpointStep = 1;
  }

  platform->timer.tic("udfExecuteStep");
  if (udf.executeStep) {
    if (platform->verbose() && platform->comm.mpiRank() == 0) {
      std::cout << "calling UDF_ExecuteStep ...\n";
    }

    udf.executeStep(timeNew, tstep);
  }
  platform->timer.toc("udfExecuteStep");

  return converged;
}

void nrs_t::saveSolutionState()
{
  if (geom) {
    geom->saveSolutionState();
  }
  if (fluid) {
    fluid->saveSolutionState();
  }
  if (scalar) {
    scalar->saveSolutionState();
  }
}

void nrs_t::restoreSolutionState()
{
  if (geom) {
    geom->restoreSolutionState();
  }
  if (fluid) {
    fluid->restoreSolutionState();
  }
  if (scalar) {
    scalar->restoreSolutionState();
  }
}

dfloat nrs_t::computeCFL()
{
  return computeCFL(fluid->mesh, fluid->o_U, dt[0]);
}

dfloat nrs_t::computeCFL(dfloat dt)
{
  return computeCFL(fluid->mesh, fluid->o_U, dt);
}

dfloat nrs_t::computeCFL(mesh_t *mesh, const occa::memory &o_U, dfloat dt)
{
  auto o_invDxRst = [&]() {
    static occa::memory o_dx;
    if (o_dx.isInitialized()) {
      return o_dx;
    }

    o_dx = platform->device.malloc<dfloat>(mesh->N + 1);

    auto mesh = meshT;
    std::vector<dfloat> dx(mesh->N + 1);
    for (int n = 0; n < (mesh->N + 1); n++) {
      if (n == 0) {
        dx[n] = mesh->gllz[n + 1] - mesh->gllz[n];
      } else if (n == mesh->N) {
        dx[n] = mesh->gllz[n] - mesh->gllz[n - 1];
      } else {
        dx[n] = 0.5 * (mesh->gllz[n + 1] - mesh->gllz[n - 1]);
      }
      dx[n] = 1.0 / dx[n];
    }
    o_dx.copyFrom(dx.data());

    return o_dx;
  }();

  auto o_cfl = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements);
  launchKernel("nrs-cflHex3D",
               mesh->Nelements,
               dt,
               mesh->o_vgeo,
               o_invDxRst,
               fluid->fieldOffset,
               o_U,
               (geom) ? geom->o_U : o_NULL,
               o_cfl);

  auto scratch = platform->memoryPool.reserve<dfloat>(o_cfl.size());
  auto scratchPtr = scratch.ptr<dfloat>();

  o_cfl.copyTo(scratchPtr);

  dfloat cflMax = 0;
  for (dlong n = 0; n < mesh->Nelements; ++n) {
    cflMax = std::max(cflMax, scratchPtr[n]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &cflMax, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm());
  return cflMax;
}

void nrs_t::evaluateProperties(const double timeNew)
{
  const auto tag = [&]() {
    bool rhsCVODE = false;
    if (scalar) {
      if (scalar->cvode) {
        rhsCVODE = scalar->cvode->isRhsEvaluation();
      }
    }
    return rhsCVODE ? "udfPropertiesCVODE" : "udfProperties";
  }();

  platform->timer.tic(tag, 1);

  if (userProperties) {
    userProperties(timeNew);
  } else {
    if (Nscalar) {
      scalar->applyAVM();
    }
  }

  platform->timer.toc(tag);
}

void nrs_t::evaluateDivergence(const double time)
{
  if (fluid && userDivergence) {
    platform->timer.tic("udfDiv");
    platform->linAlg->fill(fluid->o_div.size(), 0.0, fluid->o_div);
    userDivergence(time);
    platform->timer.toc("udfDiv");
  }
}

void nrs_t::registerKernels(occa::properties kernelInfoBC)
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerNrsKernels" << std::endl;
  }

  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";
  const std::string suffix = "Hex3D";
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/");
  const std::string section = "nrs-";

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;
  constexpr int Nfaces{6};

  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  constexpr int NVfields{3};
  kernelInfo["defines/p_NVfields"] = NVfields;

  int nBDF = 0;
  int nEXT = 0;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);

  advectionSubcycingSteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", advectionSubcycingSteps);
  if (advectionSubcycingSteps) {
    nEXT = nBDF;
  }

  std::string fileName, kernelName;

  {

    kernelName = "computeFieldDotNormal";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

    occa::properties centroidProp = kernelInfo;
    centroidProp["defines/p_Nfp"] = Nq * Nq;
    centroidProp["defines/p_Nfaces"] = Nfaces;
    {
      int N;
      platform->options.getArgs("POLYNOMIAL DEGREE", N);
      const int Nq = N + 1;
      nekrsCheck(BLOCKSIZE < Nq * Nq,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "computeFaceCentroid kernel requires BLOCKSIZE >= Nq * Nq\nBLOCKSIZE = %d, Nq*Nq = %d\n",
                 BLOCKSIZE,
                 Nq * Nq);
    }
    kernelName = "computeFaceCentroid";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, centroidProp);

    occa::properties meshProps = kernelInfo;
    meshProps += meshKernelProperties(N);

    kernelName = "SijOij" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, meshProps);

    occa::properties prop = meshProps;
    prop["defines/p_cubNq"] = cubNq;
    prop["defines/p_cubNp"] = cubNp;

    kernelName = "UrstCubature" + suffix;
    fileName = oklpath + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);

    kernelName = "Urst" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, prop);

    const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

    {
      int N;
      platform->options.getArgs("POLYNOMIAL DEGREE", N);
      const int Nq = N + 1;
      nekrsCheck(BLOCKSIZE < Nq * Nq,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "CFL kernel requires BLOCKSIZE >= Nq * Nq\nBLOCKSIZE = %d, Nq*Nq = %d\n",
                 BLOCKSIZE,
                 Nq * Nq);
    }

    occa::properties cflProps = meshProps;
    cflProps["defines/p_MovingMesh"] = movingMesh;
    kernelName = "cfl" + suffix;
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, cflProps);
  }

  registerPostProcessingKernels();

  registerFluidSolverKernels(kernelInfoBC);

  registerGeomSolverKernels(kernelInfoBC);

  Nscalar = 0;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalar);
  if (Nscalar) {
    registerScalarKernels(kernelInfoBC);
  }

  if (platform->options.compareArgs("LOWMACH", "TRUE")) { // // rho is varying
    if (platform->options.compareArgs("FLUID PRESSURE RHO SPLITTING", "TRUE")) {
      platform->options.setArgs("FLUID PRESSURE ELLIPTIC COEFF FIELD", "FALSE");
      platform->options.setArgs("FLUID PRESSURE ELLIPTIC PRECO COEFF FIELD", "FALSE");
    } else {
      platform->options.setArgs("FLUID PRESSURE ELLIPTIC COEFF FIELD", "TRUE");
    }
  }

  const auto stressForm = [&](const std::string &field) {
    if (field == "fluid velocity" && platform->options.compareArgs("FLUID STRESSFORMULATION", "TRUE")) {
      return true;
    }
    return false;
  };

  {
    auto ellipticFieldsToRegister = fieldsToSolve();

    auto list = serializeString(platform->options.getArgs("USER ELLIPTIC FIELDS"), ' ');
    for (auto &&entry : list) {
      if (!platform->options.compareArgs(std::string("ELLIPTIC ") + upperCase(entry) + " SOLVER", "NONE")) {
        ellipticFieldsToRegister.push_back("elliptic " + lowerCase(entry));
      }
    }

    for (auto &&entry : ellipticFieldsToRegister) {
      registerEllipticKernels(entry, stressForm(entry));
    }
  }
}

void nrs_t::initOuterStep(double time, dfloat _dt, int tstep)
{
  saveSolutionState();
  tStepOuterStart = tstep;
  timeOuterStart = time;

  if (tstep == 1) {
    const bool exchangeAllTimes = false;
    const bool lagState = true;
    neknek->exchange(exchangeAllTimes, lagState);
  }

  neknek->exchangeTimes(std::vector<dfloat>(this->dt, this->dt + sizeof(this->dt) / sizeof(dfloat)), time);
}

void nrs_t::finishOuterStep() {}

void nrs_t::runOuterStep(std::function<bool(int)> convergenceCheck, int stage)
{
  int NsubTimeSteps = 1;
  platform->options.getArgs("NEKNEK MULTIRATE STEPS", NsubTimeSteps);

  int requiredCorrectorSteps = 0;
  platform->options.getArgs("NEKNEK MULTIRATE CORRECTOR STEPS", requiredCorrectorSteps);

  const auto correctorStep = stage - 1;
  const bool predictorStep = (correctorStep == 0);
  neknek->setPredictor(predictorStep);

  const bool outerConverged = correctorStep >= requiredCorrectorSteps;

  if (stage > 1) {
    restoreSolutionState();
  }

  // run sub-stepping
  auto tstep = tStepOuterStart;
  auto time = timeOuterStart;
  for (int step = 1; step <= NsubTimeSteps; ++step) {
    initInnerStep(time, dt[0], tstep);
    time += setPrecision(dt[0], 5);

    int innerStage = 1;
    bool converged = false;
    do {
      converged = runInnerStep(convergenceCheck, innerStage++, outerConverged && (step == NsubTimeSteps));
    } while (!converged);

    finishInnerStep();

    if (step != NsubTimeSteps) {
      printStepInfo(time, tStepOuterStart, true, true);
    }

    tstep++;
  }

  const bool exchangeAllTimeStates = outerConverged;
  const bool lagState = outerConverged;
  neknek->exchange(exchangeAllTimeStates, lagState);
  if (!outerConverged) {
    neknek->setCorrectorTime(time);
  }
}

void nrs_t::computeUrst()
{
  auto mesh = meshV;
  auto [fieldOffset, cubatureOffset, o_U, o_relUrst] = [&]() {
    if (fluid) {
      return std::make_tuple(fluid->fieldOffset, fluid->cubatureOffset, fluid->o_U, fluid->o_relUrst);
    }
    if (scalar) {
      return std::make_tuple(scalar->fieldOffset(), scalar->vCubatureOffset, scalar->o_U, scalar->o_relUrst);
    }
    return std::make_tuple(0, 0, o_NULL, o_NULL);
  }();

  if (!o_relUrst.isInitialized()) {
    return;
  }

  if (advectionSubcycingSteps) {
    for (int s = o_coeffEXT.size(); s > 1; s--) {
      auto lagOffset = mesh->dim * cubatureOffset;
      o_relUrst.copyFrom(o_relUrst, lagOffset, (s - 1) * lagOffset, (s - 2) * lagOffset);
    }
  }

  const auto relative = static_cast<int>(geom && advectionSubcycingSteps);

  double flopCount = 0.0;
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    launchKernel("nrs-UrstCubatureHex3D",
                 mesh->Nelements,
                 relative,
                 mesh->o_cubvgeo,
                 mesh->o_cubInterpT,
                 fieldOffset,
                 (geom) ? geom->fieldOffset : 0,
                 cubatureOffset,
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
                 relative,
                 mesh->o_vgeo,
                 fieldOffset,
                 (geom) ? geom->fieldOffset : 0,
                 o_U,
                 (geom) ? geom->o_U : o_NULL,
                 o_relUrst);
    flopCount += 24 * static_cast<double>(mesh->Nlocal);
  }
  platform->flopCounter->add("Urst", flopCount);

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_relUrst,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("relUrst norm: %.15e\n", debugNorm);
    }
  }
}

nrs_t::tavgLegacy_t::tavgLegacy_t()
{
  auto nrs = dynamic_cast<nrs_t *>(platform->app);
  auto &fluid = nrs->fluid;

  std::vector<tavg::field> avgFields;
  deviceMemory<dfloat> o_u(fluid->o_U.slice(0 * fluid->fieldOffset, fluid->fieldOffset));
  deviceMemory<dfloat> o_v(fluid->o_U.slice(1 * fluid->fieldOffset, fluid->fieldOffset));
  deviceMemory<dfloat> o_w(fluid->o_U.slice(2 * fluid->fieldOffset, fluid->fieldOffset));
  avgFields.push_back({"", std::vector{o_u}});
  avgFields.push_back({"", std::vector{o_v}});
  avgFields.push_back({"", std::vector{o_w}});

  std::vector<tavg::field> rmsFields;
  rmsFields.push_back({"", std::vector{o_u, o_u}});
  rmsFields.push_back({"", std::vector{o_v, o_v}});
  rmsFields.push_back({"", std::vector{o_w, o_w}});

  for (int i = 0; i < nrs->Nscalar; i++) {
    deviceMemory<dfloat> o_temp(
        nrs->scalar->o_S.slice(nrs->scalar->fieldOffsetScan[i], nrs->scalar->fieldOffset()));
    avgFields.push_back({"", std::vector{o_temp}});
    rmsFields.push_back({"", std::vector{o_temp, o_temp}});
  }

  std::vector<tavg::field> rm2Fields;
  rm2Fields.push_back({"", std::vector{o_u, o_v}});
  rm2Fields.push_back({"", std::vector{o_v, o_w}});
  rm2Fields.push_back({"", std::vector{o_w, o_u}});

  _avg = std::make_unique<tavg>(fluid->fieldOffset, avgFields);
  _rms = std::make_unique<tavg>(fluid->fieldOffset, rmsFields);
  _rm2 = std::make_unique<tavg>(fluid->fieldOffset, rm2Fields);
}

void nrs_t::tavgLegacy_t::writeToFile(mesh_t *mesh)
{
  static int outfldCounter = 0;
  const auto outXYZ = mesh && outfldCounter == 0;

  auto nrs = dynamic_cast<nrs_t *>(platform->app);

  nekrsCheck(_avg->fieldOffset() < mesh->Nlocal,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "tavg field size is smaller than the mesh size!\n");

  static std::unique_ptr<iofld> avgWriter;
  if (!avgWriter) {
    avgWriter = iofldFactory::create("nek");
    avgWriter->open(mesh, iofld::mode::write, "avg");

    if (platform->options.compareArgs("TAVG OUTPUT PRECISION", "FP32")) {
      avgWriter->writeAttribute("precision", "32");
    } else {
      avgWriter->writeAttribute("precision", "64");
    }
    avgWriter->writeAttribute("outputmesh", (outXYZ) ? "true" : "false");

    avgWriter->addVariable("time", const_cast<double &>(_avg->time()));

    const auto fieldOffset = _avg->fieldOffset();
    {
      std::vector<deviceMemory<double>> list;
      list.push_back(_avg->o_data().slice(0 * fieldOffset, mesh->Nlocal));
      list.push_back(_avg->o_data().slice(1 * fieldOffset, mesh->Nlocal));
      list.push_back(_avg->o_data().slice(2 * fieldOffset, mesh->Nlocal));

      avgWriter->addVariable("velocity", list);
    }
    for (int i = 0; i < nrs->Nscalar; i++) {
      avgWriter->addVariable("scalar" + scalarDigitStr(i),
                             std::vector<deviceMemory<double>>{
                                 _avg->o_data().slice((mesh->dim + i) * fieldOffset, mesh->Nlocal)});
    }
  }
  avgWriter->process();

  static std::unique_ptr<iofld> rmsWriter;
  if (!rmsWriter) {
    rmsWriter = iofldFactory::create("nek");
    rmsWriter->open(mesh, iofld::mode::write, "rms");

    rmsWriter->writeAttribute("precision", "64");
    rmsWriter->writeAttribute("outputmesh", (outXYZ) ? "true" : "false");

    rmsWriter->addVariable("time", const_cast<double &>(_rms->time()));

    const auto fieldOffset = _rms->fieldOffset();
    {
      std::vector<deviceMemory<double>> list;
      list.push_back(_rms->o_data().slice(0 * fieldOffset, mesh->Nlocal));
      list.push_back(_rms->o_data().slice(1 * fieldOffset, mesh->Nlocal));
      list.push_back(_rms->o_data().slice(2 * fieldOffset, mesh->Nlocal));

      rmsWriter->addVariable("velocity", list);
    }
    for (int i = 0; i < nrs->Nscalar; i++) {
      rmsWriter->addVariable(
          "scalar" + scalarDigitStr(i),
          std::vector<occa::memory>{_rms->o_data().slice((mesh->dim + i) * fieldOffset, mesh->Nlocal)});
    }
  }
  rmsWriter->process();

  static std::unique_ptr<iofld> rm2Writer;
  if (!rm2Writer) {
    rm2Writer = iofldFactory::create("nek");
    rm2Writer->open(mesh, iofld::mode::write, "rm2");

    rm2Writer->writeAttribute("precision", "64");
    rm2Writer->writeAttribute("outputmesh", (outXYZ) ? "true" : "false");

    rm2Writer->addVariable("time", const_cast<double &>(_rm2->time()));

    const auto fieldOffset = _rm2->fieldOffset();
    {
      std::vector<deviceMemory<double>> list;
      list.push_back(_rm2->o_data().slice(0 * fieldOffset, mesh->Nlocal));
      list.push_back(_rm2->o_data().slice(1 * fieldOffset, mesh->Nlocal));
      list.push_back(_rm2->o_data().slice(2 * fieldOffset, mesh->Nlocal));

      rm2Writer->addVariable("velocity", list);
    }
  }
  rm2Writer->process();

  outfldCounter++;
}

void nrs_t::tavgLegacy_t::reset()
{
  _avg->reset();
  _rms->reset();
  _rm2->reset();
}

void nrs_t::tavgLegacy_t::run(double time)
{
  _avg->run(time);
  _rms->run(time);
  _rm2->run(time);
}

const deviceMemory<double> nrs_t::tavgLegacy_t::o_avg()
{
  return _avg->o_data();
}

const deviceMemory<double> nrs_t::tavgLegacy_t::o_rms()
{
  return _rms->o_data();
}

const deviceMemory<double> nrs_t::tavgLegacy_t::o_rm2()
{
  return _rm2->o_data();
}
