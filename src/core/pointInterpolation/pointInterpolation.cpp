#include <inttypes.h>
#include "platform.hpp"
#include "findpts.hpp"
#include "pointInterpolation.hpp"

pointInterpolation_t::pointInterpolation_t(mesh_t *mesh_,
                                           MPI_Comm comm,
                                           bool mySession_,
                                           std::vector<int> bIntID,
                                           double bb_tol,
                                           double newton_tol_,
                                           dlong localHashSize,
                                           dlong globalHashSize)

    : mesh(mesh_), mySession(mySession_), nPoints(0)
{
  if (localHashSize == 0) {
    localHashSize = mesh->Nlocal;
  }
  if (globalHashSize == 0) {
    globalHashSize = mesh->Nlocal;
  }

  // communicator is implicitly required to be either platform->comm.mpiComm()()or
  // platform->comm.mpiComm()Parent due to other communicator synchronous calls, such as platform->timer.tic
  bool supported = false;
  for (auto &&supportedCommunicator : {platform->comm.mpiComm(), platform->comm.mpiCommParent()}) {
    int same = 0;
    MPI_Comm_compare(comm, supportedCommunicator, &same);
    supported |= (same != MPI_UNEQUAL);
  }
  nekrsCheck(!supported,
             comm,
             EXIT_FAILURE,
             "%s\n",
             "Communicator must be either platform->comm.mpiComm()()or platform->comm.mpiComm()Parent");

  newton_tol =
      (sizeof(dfloat) == sizeof(double)) ? std::max(5e-13, newton_tol_) : std::max(5e-5, newton_tol_);

  auto x = platform->device.mallocHost<dfloat>(mesh->Nlocal);
  auto y = platform->device.mallocHost<dfloat>(mesh->Nlocal);
  auto z = platform->device.mallocHost<dfloat>(mesh->Nlocal);

  if (mySession) {
    mesh->o_x.copyTo(x, mesh->Nlocal);
    mesh->o_y.copyTo(y, mesh->Nlocal);
    mesh->o_z.copyTo(z, mesh->Nlocal);
  }

  std::vector<dfloat> distanceINT;
  if (bIntID.size()) {
    auto o_bIntID = platform->deviceMemoryPool.reserve<int>(bIntID.size());
    o_bIntID.copyFrom(bIntID.data());
    _o_distanceINT = mesh->minDistance(bIntID.size(), o_bIntID, "cheap_dist");
    distanceINT.resize(mesh->Nlocal);
    _o_distanceINT.copyTo(distanceINT.data(), mesh->Nlocal);
  }

  // number of points to iterate on simultaneously
  const int npt_max = 1;

  int sessionID = 0;
  platform->options.getArgs("NEKNEK SESSION ID", sessionID);

  findpts_ = std::make_unique<findpts::findpts_t>(comm,
                                                  mySession ? x.ptr<dfloat>() : nullptr,
                                                  mySession ? y.ptr<dfloat>() : nullptr,
                                                  mySession ? z.ptr<dfloat>() : nullptr,
                                                  mesh->Nq,
                                                  mySession ? mesh->Nelements : 0,
                                                  2 * mesh->Nq,
                                                  bb_tol,
                                                  localHashSize,
                                                  globalHashSize,
                                                  npt_max,
                                                  newton_tol,
                                                  sessionID,
                                                  distanceINT.data());
}

occa::memory pointInterpolation_t::distanceINT()
{
  nekrsCheck(!_o_distanceINT.isInitialized(),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "No INT boundary IDs provided on setup!");

  return _o_distanceINT;
}

void pointInterpolation_t::find(pointInterpolation_t::VerbosityLevel verbosity, bool matchSession)
{
  if (timerLevel != TimerLevel::None) {
    platform->timer.tic("pointInterpolation_t::find");
  }

  int iErr = 0;
  iErr += !pointsAdded;
  nekrsCheck(iErr, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "find called without any points added!");

  const auto n = nPoints;
  const dlong sessionIDMatch = matchSession;

  if (useHostPoints) {
    findpts_->find(&data_, _x, _y, _z, _session, sessionIDMatch, n);
  } else {
    findpts_->find(&data_, _o_x, _o_y, _o_z, _o_session, sessionIDMatch, n);
  }

  if (verbosity != VerbosityLevel::None) {
    auto xPtr = _x;
    auto yPtr = _y;
    auto zPtr = _z;
    
    occa::memory h_x;
    occa::memory h_y;
    occa::memory h_z;
    if (!useHostPoints && verbosity == VerbosityLevel::Detailed) {
      h_x = platform->memoryPool.reserve<dfloat>(n);
      h_y = platform->memoryPool.reserve<dfloat>(n);
      h_z = platform->memoryPool.reserve<dfloat>(n);
      _o_x.copyTo(h_x);
      _o_y.copyTo(h_y);
      _o_z.copyTo(h_z);
      xPtr = h_x.ptr<dfloat>();
      yPtr = h_y.ptr<dfloat>();
      zPtr = h_z.ptr<dfloat>();
    }

    const auto maxVerbosePoints = 5;

    dlong nOutside = 0;
    dlong nBoundary = 0;
    dfloat maxDistNorm = 0;
    for (int in = 0; in < n; ++in) {
      if (data_.code_base[in] == findpts::CODE_BORDER) {
        if (data_.dist2_base[in] > 10 * newton_tol) {
          const auto distNorm = data_.dist2_base[in];
          maxDistNorm = std::max(maxDistNorm, distNorm);
          nBoundary++;
          if (nBoundary < maxVerbosePoints && verbosity == VerbosityLevel::Detailed) {
            std::cout << "pointInterpolation_t::find: WARNING point on boundary or outside the mesh"
                      << " xyz= " << xPtr[in] << " " << yPtr[in] << " " << zPtr[in]
                      << " distNorm= " << std::scientific << std::setprecision(3) << distNorm << std::endl;
          }
        }
      } else if (data_.code_base[in] == findpts::CODE_NOT_FOUND) {
        nOutside++;
        if (nOutside < maxVerbosePoints && verbosity == VerbosityLevel::Detailed) {
          std::cout << "pointInterpolation_t::find: WARNING point outside the mesh"
                    << " xyz= " << xPtr[in] << " " << yPtr[in] << " " << zPtr[in] << std::endl;
        }
      }
    }

    std::array<hlong, 3> counts = {n, nBoundary, nOutside};
    MPI_Allreduce(MPI_IN_PLACE, counts.data(), counts.size(), MPI_HLONG, MPI_SUM, platform->comm.mpiComm());
    MPI_Allreduce(MPI_IN_PLACE, &maxDistNorm, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm());

    if (platform->comm.mpiRank() == 0 && verbosity == VerbosityLevel::Detailed) {
      std::cout << "pointInterpolation_t::find:"
                << " total= " << counts[0] << " boundary= " << counts[1] << " (max distNorm=" << maxDistNorm
                << ")"
                << " outside= " << counts[2] << std::endl;
    }
  }

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc("pointInterpolation_t::find");
  }

  findCalled = true;
  data_.updateCache = true;
}

void pointInterpolation_t::eval(dlong nFields,
                                dlong inputFieldOffset,
                                const occa::memory &o_in,
                                dlong outputFieldOffset,
                                occa::memory &o_out,
                                dlong nPointsIn,
                                dlong offset)
{
  if (nFields == 1) {
    inputFieldOffset = mesh->Nlocal;
  }

  const auto nPoints_ = (nPointsIn > -1) ? nPointsIn : nPoints;
  nekrsCheck(nPointsIn > nPoints_, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "nPointsIn too large!");

  if (nFields == 1) {
    outputFieldOffset = nPoints_;
  }

  // enforce update as cache cannot be used (might have different size from a previous call)
  if (nPointsIn > -1) {
    data_.updateCache = true;
  }

  nekrsCheck(!findCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "find has not been called prior to eval!");

  nekrsCheck(nFields > 1 && mesh->Nlocal > inputFieldOffset,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "pointInterpolation_t::eval inputFieldOffset (%d) is less than mesh->Nlocal (%d)\n",
             inputFieldOffset,
             mesh->Nlocal);

  nekrsCheck(nFields > 1 && nPoints_ > outputFieldOffset,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "pointInterpolation_t::eval outputFieldOffset (%d) is less than nPoints (%d)\n",
             inputFieldOffset,
             nPoints_);

  nekrsCheck(o_in.byte_size() < nFields * inputFieldOffset * sizeof(dfloat),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "pointInterpolation_t::eval input size (%" PRId64 ") is smaller than expected\n",
             o_in.byte_size());

  nekrsCheck(o_out.byte_size() == 0 && nPoints_ ||
                 (o_out.byte_size() < nFields * outputFieldOffset * sizeof(dfloat)),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "pointInterpolation_t::eval output size (%" PRId64 ") is smaller than expected\n",
             o_out.byte_size());

  if (timerLevel != TimerLevel::None) {
    platform->timer.tic("pointInterpolation_t::eval");
  }

  findpts_->eval(nPoints_, offset, nFields, inputFieldOffset, outputFieldOffset, o_in, &data_, o_out);

  if (timerLevel != TimerLevel::None) {
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::setPoints(const std::vector<dfloat> &x,
                                     const std::vector<dfloat> &y,
                                     const std::vector<dfloat> &z)
{
  std::vector<dlong> session;
  this->setPoints(x, y, z, session);
}

void pointInterpolation_t::setPoints(const std::vector<dfloat> &x,
                                     const std::vector<dfloat> &y,
                                     const std::vector<dfloat> &z,
                                     const std::vector<dlong> &session)
{
  auto o_x = platform->device.malloc<dfloat>(x.size());
  o_x.copyFrom(x.data());
  auto o_y = platform->device.malloc<dfloat>(y.size());
  o_y.copyFrom(y.data());
  auto o_z = platform->device.malloc<dfloat>(z.size());
  o_z.copyFrom(z.data());

  occa::memory o_session;
  if (session.size()) {
    o_session = platform->device.malloc<dlong>(session.size());
    o_session.copyFrom(session.data());
  }
  this->setPoints(o_x, o_y, o_z, o_session);
}

void pointInterpolation_t::setPoints(const occa::memory &o_x,
                                     const occa::memory &o_y,
                                     const occa::memory &o_z)
{
  this->setPoints(o_x, o_y, o_z, o_NULL);
}

void pointInterpolation_t::setPoints(const occa::memory &o_x,
                                     const occa::memory &o_y,
                                     const occa::memory &o_z,
                                     const occa::memory &o_session)
{
  nPoints = o_x.size();

  pointsAdded = true;
  useHostPoints = false;
  useDevicePoints = true;

  if (data_.code.size() < nPoints) {
    data_.resize(nPoints);
  }

  for (int i = 0; i < nPoints; ++i) {
    data_.dist2_base[i] = 1e30;
    data_.code_base[i] = CODE_NOT_FOUND;
  }
  data_.updateCache = true;

  _o_session = o_session;
  _o_x = o_x;
  _o_y = o_y;
  _o_z = o_z;
}

void pointInterpolation_t::setTimerLevel(TimerLevel level)
{
  timerLevel = level;
  findpts_->setTimerLevel(level);
}

TimerLevel pointInterpolation_t::getTimerLevel() const
{
  return timerLevel;
}

void pointInterpolation_t::setTimerName(std::string name)
{
  timerName = name;
  findpts_->setTimerName(name);
}
