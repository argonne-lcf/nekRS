#include <cfloat>
#include "platform.hpp"
#include "neknek.hpp"
#include "nekInterfaceAdapter.hpp"
#include "pointInterpolation.hpp"
#include "solver.hpp"

#include "sha1.hpp"

neknek_t::neknek_t(mesh_t *_mesh, dlong nsessions, dlong sessionID)
    : mesh(_mesh), nsessions_(nsessions), sessionID_(sessionID)
{
  this->nEXT_ = 1;
  if (!platform->options.getArgs("NEKNEK BOUNDARY EXT ORDER").empty()) {
    platform->options.getArgs("NEKNEK BOUNDARY EXT ORDER", this->nEXT_);
  }

  // set boundary ext order to report to user, if not specified
  platform->options.setArgs("NEKNEK BOUNDARY EXT ORDER", std::to_string(this->nEXT_));

  this->multirate_ = platform->options.compareArgs("NEKNEK MULTIRATE TIMESTEPPER", "TRUE");

  this->coeffEXT.resize(this->nEXT_);
  this->o_coeffEXT = platform->device.malloc<dfloat>(this->nEXT_);
}

void neknek_t::setup()
{
  if (platform->comm.mpiRank() == 0) {
    printf("found %d sessions\n", nsessions_);
    std::fflush(stdout);
  }

  nekrsCheck(fields_.size() < 1,
             platform->comm.mpiCommParent(),
             EXIT_FAILURE,
             "%s\n",
             "no neknek fields specified!");

  globalMovingMesh = [&]() {
    int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
    MPI_Allreduce(MPI_IN_PLACE, &movingMesh, 1, MPI_INT, MPI_MAX, platform->comm.mpiCommParent());
    return movingMesh;
  }();

  // ensure all exchanged fields (within the same session) share the same INT boundaries
  {
    std::ostringstream errorLogger;
    std::set<int> intBIDFields;
    for (auto &&field : fields_) {
      const int nEntries = (field.name == "scalar") ? field.filter.size() : 1;

      for (int n = 0; n < nEntries; n++) {
        auto fieldName = field.name;
        if (field.name == "scalar") {
          fieldName += scalarDigitStr(field.filter.at(n));
        }

        for (int bID = 1; bID <= platform->app->bc->size(fieldName); ++bID) {
          const auto isInt = (platform->app->bc->typeId(bID, fieldName) == bdryBase::bcType_interpolation);

          if (isInt) {
            intBIDFields.insert(bID);
          }

          if ((intBIDFields.find(bID) != intBIDFields.end()) && !isInt) {
            errorLogger << "ERROR: expected INT boundary condition on boundary id " << bID << " for field "
                        << fieldName << "\n";
          }
        }
      }
    }
    int errorLength = errorLogger.str().length();
    nekrsCheck(errorLength > 0,
               platform->comm.mpiCommParent(),
               EXIT_FAILURE,
               "%s\n",
               errorLogger.str().c_str());

    for (auto &&entry : intBIDFields) {
      intBIDs.push_back(entry);
    }
  }

#if 0
  for (int bID = 1; bID <= platform->app->bc->size(fields_[0].name); ++bID) {
    if (platform->app->bc->typeId(bID, fields_[0].name) == bdryBase::bcType_interpolation)
      ;
    intBIDs.push_back(bID);
  }
#endif

  npt_ = [&]() {
    dlong nFaces = 0;
    for (dlong e = 0; e < mesh->Nelements; ++e) {
      for (dlong f = 0; f < mesh->Nfaces; ++f) {
        auto bID = mesh->EToB[f + mesh->Nfaces * e];
        auto fieldName = fields_[0].name;
        if (fieldName == "scalar") {
          fieldName += scalarDigitStr(fields_[0].filter.at(0));
        }

        if (bID > 0 && platform->app->bc->typeId(bID, fieldName) == bdryBase::bcType_interpolation) {
          nFaces++;
        }
      }
    }

    nekrsCheck(nFaces < 1,
               platform->comm.mpiCommParent(),
               EXIT_FAILURE,
               "%s\n",
               "no interpolation boundaries found!");

    return nFaces * mesh->Nfp;
  }();

  intValOffset_ = alignStride<dlong>(npt_);
  o_pointMap_ = platform->device.malloc<dlong>(mesh->Nlocal);

  this->findIntPoints();

  for (auto &&field : fields_) {
    int nStates = nEXT_ + 1;
    if (multirate()) {
      nStates++;
    }
    field.o_intVal = platform->device.malloc<dfloat>(field.o_filter.size() * intValOffset_ * nStates);
  }

  if (platform->comm.mpiRank() == 0) {
    std::cout << "exchanged fields: ";
    for (auto &&field : fields_) {
      const int nEntries = (field.name == "scalar") ? field.filter.size() : 1;
      for (int n = 0; n < nEntries; n++) {
        auto fieldName = field.name;
        if (fieldName == "scalar") {
          fieldName += scalarDigitStr(field.filter.at(n));
        }
        std::cout << fieldName << "  ";
      }
    }
    std::cout << "\n" << std::flush;
  }

  // check if fields across all sessions match
  {
    std::string s;
    for (auto &&field : fields_) {
      s += field.name;
    }

    SHA1 sha;
    sha.update(s);
    const auto hash = sha.final();
    const auto hashTruncated = hash.substr(hash.length() - 8);
    unsigned long intHash = std::stoul("0x" + hashTruncated, nullptr, 0);

    unsigned long intHashMin;
    unsigned long intHashMax;

    MPI_Allreduce(&intHash, &intHashMin, 1, MPI_UNSIGNED_LONG, MPI_MIN, platform->comm.mpiCommParent());
    MPI_Allreduce(&intHash, &intHashMax, 1, MPI_UNSIGNED_LONG, MPI_MAX, platform->comm.mpiCommParent());

    nekrsCheck(intHashMin != intHashMax,
               platform->comm.mpiCommParent(),
               EXIT_FAILURE,
               "%s\n",
               "neknek fields do not match across all sessions");
  }

  if (platform->comm.mpiRank() == 0) {
    std::cout << "done\n" << std::flush;
  }
}

void neknek_t::updateInterpPoints()
{
  if (!this->globalMovingMesh) {
    return; // no need to continue if mesh is not moving
  }

  this->interpolator.reset();
  this->interpolator =
      std::make_shared<pointInterpolation_t>(mesh, platform->comm.mpiCommParent(), true, intBIDs);
  this->interpolator->setTimerName("neknek_t::");

  launchKernel("neknek::copyNekNekPoints",
               mesh->Nlocal,
               this->o_pointMap_,
               mesh->o_x,
               mesh->o_y,
               mesh->o_z,
               this->o_x_,
               this->o_y_,
               this->o_z_);

  this->interpolator->setPoints(this->o_x_, this->o_y_, this->o_z_, this->o_session_);

  const auto verboseLevel = pointInterpolation_t::VerbosityLevel::Detailed;
  this->interpolator->find(verboseLevel);
}

void neknek_t::findIntPoints()
{
  const dlong sessionID = sessionID_;

  interpolator.reset();
  interpolator = std::make_shared<pointInterpolation_t>(mesh, platform->comm.mpiCommParent(), true, intBIDs);
  interpolator->setTimerName("neknek_t::");

  this->o_x_ = platform->device.malloc<dfloat>(this->npt_);
  this->o_y_ = platform->device.malloc<dfloat>(this->npt_);
  this->o_z_ = platform->device.malloc<dfloat>(this->npt_);
  std::vector<dfloat> neknekX(o_x_.size(), 0.0);
  std::vector<dfloat> neknekY(o_y_.size(), 0.0);
  std::vector<dfloat> neknekZ(o_z_.size(), 0.0);

  this->o_session_ = platform->device.malloc<dlong>(this->npt_);
  std::vector<dlong> session(o_session_.size(), -1);

  std::vector<dlong> pointMap(o_pointMap_.size(), -1);

  if (this->fields_.size()) {
    auto [x, y, z] = mesh->xyzHost();

    dlong ip = 0;
    for (dlong e = 0; e < mesh->Nelements; ++e) {
      for (dlong f = 0; f < mesh->Nfaces; ++f) {

        for (dlong m = 0; m < mesh->Nfp; ++m) {
          dlong id = mesh->Nfaces * mesh->Nfp * e + mesh->Nfp * f + m;
          dlong idM = mesh->vmapM[id];

          auto bID = mesh->EToB[f + mesh->Nfaces * e];
          auto fieldName = fields_[0].name;
          if (fieldName == "scalar") {
            fieldName += scalarDigitStr(fields_[0].filter.at(0));
          }

          if (platform->app->bc->typeId(bID, fieldName) == bdryBase::bcType_interpolation) {
            neknekX[ip] = x[idM];
            neknekY[ip] = y[idM];
            neknekZ[ip] = z[idM];
            session[ip] = sessionID;
            pointMap[idM] = ip;
            ++ip;
          }
        }
      }
    }
  }

  this->interpolator->setPoints(neknekX, neknekY, neknekZ, session);

  const auto verboseLevel = pointInterpolation_t::VerbosityLevel::Detailed;
  this->interpolator->find(verboseLevel);

  this->o_x_.copyFrom(neknekX.data());
  this->o_y_.copyFrom(neknekY.data());
  this->o_z_.copyFrom(neknekZ.data());

  this->o_session_.copyFrom(session.data());

  this->o_pointMap_.copyFrom(pointMap.data());
}

void neknek_t::updateBoundary(int tstep, int stage, dfloat *dt, double time)
{
  if (multirate()) {
    extrapolateBoundary(tstep, time, predictorStep);
    return;
  }

  // do not invoke barrier -- this is performed later
  platform->timer.tic("neknek update boundary");

  const bool exchangeAllTimes = false;
  const bool lagState = (stage == 1);
  exchange(exchangeAllTimes, lagState);

  // lag state, update timestepper coefficients and compute extrapolated state
  if (stage == 1) {
    extrapolate(tstep, dt);
  }

  platform->timer.toc("neknek update boundary");
}

occa::memory neknek_t::partitionOfUnity()
{
  if (!this->o_partition_.isInitialized()) {
    this->o_partition_ = platform->device.malloc<dfloat>(mesh->Nlocal);
  }

  if (!recomputePartition) {
    return this->o_partition_;
  }
  recomputePartition = false;

  auto pointInterp = pointInterpolation_t(mesh, platform->comm.mpiCommParent(), true, intBIDs);

  auto o_dist = pointInterp.distanceINT();

  auto o_sess = platform->deviceMemoryPool.reserve<dlong>(mesh->Nlocal);
  auto o_sumDist = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  auto o_found = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  auto o_interpDist = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  o_sumDist.copyFrom(o_dist, mesh->Nlocal);

  std::vector<dfloat> found(mesh->Nlocal);
  std::vector<dlong> sessions(mesh->Nlocal);

  for (int sess = 0; sess < this->nsessions_; ++sess) {
    auto id = (sess + this->sessionID_) % this->nsessions_;
    if (id == this->sessionID_) {
      continue;
    }
    std::fill(sessions.begin(), sessions.end(), id);
    o_sess.copyFrom(sessions.data(), mesh->Nlocal);

    pointInterp.setPoints(mesh->o_x, mesh->o_y, mesh->o_z, o_sess);
    pointInterp.find(pointInterpolation_t::VerbosityLevel::None, true);

    auto &data = pointInterp.data();
    for (int n = 0; n < mesh->Nlocal; ++n) {
      found[n] = (data.code[n] == pointInterpolation_t::CODE_NOT_FOUND) ? 0.0 : 1.0;
    }

    o_found.copyFrom(found.data());
    pointInterp.eval(1, 0, o_dist, 0, o_interpDist);

    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_found, o_interpDist);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_interpDist, 1.0, o_sumDist);
  }

  // \Xi(x) = \dfrac{\delta^s(x)}{\sum_{i=1}^S \delta^s(x_i)}
  this->o_partition_.copyFrom(o_dist, mesh->Nlocal);
  platform->linAlg->aydx(mesh->Nlocal, 1.0, o_sumDist, this->o_partition_);

  o_sess.free();
  o_sumDist.free();
  o_found.free();
  o_interpDist.free();

  return this->o_partition_;
}

void neknek_t::lag()
{
  int nStates = this->nEXT_ + 1;
  if (this->multirate()) {
    nStates += 1;
  }

  for (auto &&field : fields_) {
    const auto N = field.o_filter.size() * intValOffset_;
    for (int s = nStates; s > 1; s--) {
      field.o_intVal.copyFrom(field.o_intVal, N, (s - 1) * N, (s - 2) * N);
    }
  }
}

void neknek_t::extrapolate(int tstep, dfloat *dt)
{
  int nBDF;
  platform->options.getArgs("BDF ORDER", nBDF);
  int bdfOrder = std::min(tstep, nBDF);

  int extOrder = std::min(tstep, this->nEXT_);

  nek::extCoeff(this->coeffEXT.data(), dt, extOrder, extOrder);

  for (int i = this->nEXT_; i > extOrder; i--) {
    this->coeffEXT[i - 1] = 0.0;
  }

  this->o_coeffEXT.copyFrom(this->coeffEXT.data(), this->nEXT_);

  if (this->npt_) {
    for (auto &&field : fields_) {
      launchKernel("core-extrapolate",
                   npt_,
                   static_cast<int>(field.o_filter.size()),
                   nEXT_,
                   intValOffset_,
                   o_coeffEXT,
                   field.o_intVal + intValOffset_ * field.o_filter.size(),
                   field.o_intVal);
    }
  }
}

void neknek_t::exchange(bool allTimeStates, bool lagState)
{
  // do not invoke barrier in timer_t::tic
  platform->timer.tic("neknek sync");
  MPI_Barrier(platform->comm.mpiCommParent());
  platform->timer.toc("neknek sync");
  this->tSync_ = platform->timer.query("neknek sync", "HOST:MAX");

  if (this->globalMovingMesh) {
    platform->timer.tic("neknek updateInterpPoints");
    this->updateInterpPoints();
    platform->timer.toc("neknek updateInterpPoints");

    this->recomputePartition = true;
  }

  if (allTimeStates) {
    int nBDF;
    int nEXT;
    platform->options.getArgs("BDF ORDER", nBDF);
    platform->options.getArgs("EXT ORDER", nEXT);
    auto n = std::max(nBDF, nEXT);
    nekrsCheck(n < this->nEXT_,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "neknek extrapolation order (%d) exceeds (%d)\n",
               this->nEXT_,
               n);
  }

  const auto nStates = allTimeStates ? nEXT_ : 1;

  platform->timer.tic("neknek exchange");

  auto containsSequence = [](const std::vector<int> &vec, int startValue) {
    auto it = std::find(vec.begin(), vec.end(), startValue);
    if (it == vec.end()) {
      return false;
    }
    std::vector<int> sequence(vec.end() - it);
    std::iota(sequence.begin(), sequence.end(), startValue);
    return std::search(vec.begin(), vec.end(), sequence.begin(), sequence.end()) != vec.end();
  };

  for (auto &&field : fields_) {
    const int nFieldsFilter = field.o_filter.size();
    const int nFields = field.offsetSum / field.offset;
    auto o_field = [&]() {
      if (!containsSequence(field.filter, 0) || nFields != nFieldsFilter) {
        auto o_fld = platform->deviceMemoryPool.reserve<dfloat>(nStates * nFieldsFilter * field.offset);
        launchKernel("neknek::pack",
                     mesh->Nlocal,
                     field.offsetSum,
                     field.offset,
                     nStates,
                     nFieldsFilter,
                     field.o_filter,
                     field.o_field,
                     o_fld);
        return o_fld;
      }
      return field.o_field;
    }();

    this->interpolator->eval(nStates * nFieldsFilter, field.offset, o_field, intValOffset_, field.o_intVal);
  }

  platform->timer.toc("neknek exchange");

  this->tExch_ = platform->timer.query("neknek exchange", "DEVICE:MAX");
  this->ratio_ = this->tSync_ / this->tExch_;

  if (lagState) {
    lag();
  }
}

double neknek_t::adjustDt(double dt)
{
  if (!this->multirate()) {
    double minDt = dt;
    MPI_Allreduce(MPI_IN_PLACE, &minDt, 1, MPI_DOUBLE, MPI_MIN, platform->comm.mpiCommParent());
    double maxDt = dt;
    MPI_Allreduce(MPI_IN_PLACE, &maxDt, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiCommParent());

    const auto relErr = std::abs(maxDt - minDt) / maxDt;
    nekrsCheck(relErr > 100 * std::numeric_limits<double>::epsilon(),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "Time step size needs to be the same across all sessions.\n"
               "Max dt = %e, Min dt = %e\n",
               maxDt,
               minDt);

    return dt;
  }

  double maxDt = dt;
  MPI_Allreduce(MPI_IN_PLACE, &maxDt, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiCommParent());

  double ratio = maxDt / dt;
  int timeStepRatio = std::floor(ratio);
  double maxErr = std::abs(ratio - timeStepRatio);

  MPI_Allreduce(MPI_IN_PLACE, &maxErr, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiCommParent());
  nekrsCheck(maxErr > 1e-4,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "Multirate time stepping requires a fixed integer time step size ratio\n"
             "Max dt = %e, dt = %e, ratio = %e, ratioErr = %e\n",
             maxDt,
             dt,
             ratio,
             maxErr);

  // rescale dt to be an _exact_ integer multiple of minDt
  dt = maxDt / timeStepRatio;
  platform->options.setArgs("NEKNEK MULTIRATE STEPS", std::to_string(timeStepRatio));
  return dt;
}

void neknek_t::addVariable(const std::string &name, dlong fieldOffset, const occa::memory &o_fld)
{
  std::vector<int> filter = {0};
  addVariable(name, filter, fieldOffset, fieldOffset, o_fld);
}

void neknek_t::addVariable(const std::string &name,
                           const std::vector<int> &_filter,
                           dlong fieldOffsetSum,
                           dlong fieldOffset,
                           const occa::memory &o_fld)
{
  field_t field;
  field.name = name;

  field.offsetSum = fieldOffsetSum;
  field.offset = fieldOffset;

  if (_filter.size() < 1) {
    auto nFields = (fieldOffset > 0) ? fieldOffsetSum / fieldOffset : 1;
    field.filter.resize(nFields);
    std::iota(field.filter.begin(), field.filter.end(), 0);
  } else {
    field.filter = _filter;
  }

  if (field.filter.size() > 1 && field.offset < mesh->Nlocal) {
    throw std::runtime_error("offset of field " + name + " has to be > mesh->Nlocal");
  }

  field.o_filter = platform->device.malloc<int>(field.filter.size());
  field.o_filter.copyFrom(field.filter.data());

  field.o_field = o_fld;

  fields_.push_back(field);
}
