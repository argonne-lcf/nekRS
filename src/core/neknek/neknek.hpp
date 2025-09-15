#if !defined(neknek_hpp_)
#define neknek_hpp_

#include "nekrsSys.hpp"
#include "mesh.h"
#include "findpts.hpp"
#include "pointInterpolation.hpp"

class neknek_t
{
public:

class field_t {
public:
  std::string name;
  std::vector<int> filter;
  occa::memory o_filter;

  dlong offsetSum;
  dlong offset;
  occa::memory o_field;

  occa::memory o_intVal;

  int intValFieldIndex(int is) const
  {
    auto it = std::find(filter.begin(), filter.end(), is);

    if (it != filter.end()) {
        return std::distance(filter.begin(), it);
    } else {
        return -1; 
    }
  };
};
  neknek_t(mesh_t *mesh, int _nsessions, int _sessionID);

  void updateBoundary(int tstep, int stage, dfloat* dt, double time);
  void exchange(bool allTimeStates = false, bool lag = false);

  // multi-rate specific functions
  void exchangeTimes(const std::vector<dfloat>& dt, double time);
  void setCorrectorTime(double time);

  void extrapolateBoundary(int tstep, double time, bool predictor);

  // kludge: need to know whether we are in predictor or corrector part of step
  // for multi-rate timestepping
  void setPredictor(bool predictor)
  {
    predictorStep = predictor;
  }

  // provide partition of unity function
  // this may be used, e.g., to do global integrations across the domain
  // when collocated with the mass matrix
  occa::memory partitionOfUnity();

  void fixCoupledSurfaceFlux(const occa::memory& o_EToB, dlong fieldOffset, occa::memory& o_U);

  void setTimerLevel(const std::string& level);

  double adjustDt(double dt);

  dfloat tSync() const
  {
    return tSync_;
  }

  dfloat tExch() const
  {
    return tExch_;
  }

  dfloat ratio() const
  {
    return ratio_;
  }

  dlong fieldOffset() const
  {
    return intValOffset_;
  }

  dlong nEXT() const
  {
    return nEXT_;
  }

  dlong npt() const
  {
    return npt_;
  }

  dlong nPoints() const
  {
    return npt_;
  }

  dlong intValOffset() const
  {
    return intValOffset_;
  }

  dlong nsessions() const
  {
    return nsessions_;
  }

  dlong sessionID() const
  {
    return sessionID_;
  }

  const occa::memory &o_x() const
  {
    return o_x_;
  }

  const occa::memory &o_y() const
  {
    return o_y_;
  }

  const occa::memory &o_z() const
  {
    return o_z_;
  }

  const occa::memory &o_pointMap() const
  {
    return o_pointMap_;
  }

  const occa::memory &o_session() const
  {
    return o_session_;
  }

  const occa::memory &o_partition() const
  {
    return o_partition_;
  }

  bool multirate() const
  {
    return multirate_;
  }

  bool hasField(const std::string& name) const
  {
    auto it = std::find_if(fields_.begin(), fields_.end(),
                           [&name](const field_t& field) {
                               return field.name == name;
                           });

    return it != fields_.end();
  } 

  const field_t& getField(const std::string& name) const 
  {
    auto it = std::find_if(fields_.begin(), fields_.end(),
                           [&name](const field_t& field) {
                               return field.name == name;
                           });

    if (it == fields_.end()) {
      throw std::runtime_error("Field not found: " + name);
    }

    return *it;
  }

  // filter: field indices into o_fld (separated by fieldOffset)
  void addVariable(const std::string &name, const std::vector<int>& filter, dlong fieldOffsetSum, dlong fieldOffset, const occa::memory &fld);
  void addVariable(const std::string &name, dlong fieldOffset, const occa::memory &o_fld);
 
  void setup();

private:
  std::vector<field_t> fields_;

  void lag();
  void extrapolate(int tstep, dfloat *dt);
  void reserveAllocation();
  void updateInterpPoints();
  void findIntPoints();

  std::vector<dfloat> coeffEXT;
  occa::memory o_coeffEXT;
  bool globalMovingMesh;

  std::vector<int> intBIDs;

  mesh_t* mesh;

  occa::memory o_x_;
  occa::memory o_y_;
  occa::memory o_z_;

  occa::memory o_pointMap_;
  occa::memory o_session_;
  occa::memory o_partition_;
  occa::memory o_time_;

  dfloat tSync_;
  dfloat tExch_;
  dfloat ratio_;

  dlong nEXT_;

  dlong npt_;

  dlong intValOffset_;

  int nsessions_;
  int sessionID_;

  std::shared_ptr<pointInterpolation_t> interpolator;

  bool recomputePartition = true;

  bool multirate_ = false;

  bool predictorStep = false;

  TimerLevel findptsTimerLevel = TimerLevel::Basic;
};

#endif
