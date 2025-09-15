#if !defined(nrs_nekrs_hpp_)
#define nrs_nekrs_hpp_

#include "platform.hpp"
#include "linAlg.hpp"
#include "elliptic.hpp"
#include "scalarSolver.hpp"
#include "randomVector.hpp"
#include "aeroForce.hpp"
#include "iofldFactory.hpp"
#include "bdryBase.hpp"

#include "app.hpp"
#include "fluidSolver.hpp"
#include "geomSolver.hpp"
#include "tavg.hpp"

class nrs_t : public app_t
{

public:
  using preFluid_t = std::function<void(double, int)>;
  using postScalar_t = std::function<void(double, int)>;
  using userDivergence_t = std::function<void(double)>;

  void init() override;

  std::string id() const override
  {
    return "nrs";
  };

  void registerKernels(occa::properties kernelInfoBC) override;
  void setDefaultSettings(setupAide &options) override;

  void printSolutionMinMax() override;

  void initStep(double time, dfloat dt, int tstep) override;
  dfloat adjustDt(int tstep) override;
  bool runStep(userConvergenceCheck_t f, int stage) override;
  void finishStep() override;

  int setLastStep(double timeNew, int tstep, double elapsedTime) override;

  void writeToFile(const std::string &fileName_, 
                   double time, 
                   mesh_t *mesh_, 
                   const std::vector<std::tuple<std::string, std::vector<deviceMemory<dfloat>>>>& list,
                   bool enforceOutXYZ = true,
                   bool enforceFP64 = false,
                   int Nout = 0,
                   bool uniform = false);

  void writeCheckpoint(double t,
                       bool enforceOutXYZ = false,
                       bool enforceFP64 = false,
                       int Nout = 0,
                       bool uniform = false) override;

  void printRunStat(int step) override;
  void printStepInfo(double time, int tstep, bool printStepInfo, bool printVerboseInfo) override;

  void finalize() override;

  const std::vector<std::string>& fieldsToSolve() const override
  {
    auto& options = platform->options;

    static std::vector<std::string> fields;
    fields.clear();

    if (options.compareArgs("MOVING MESH", "TRUE") && !options.compareArgs("GEOM SOLVER", "NONE")) {
      fields.push_back("geom");
    }

    if (!options.compareArgs("FLUID", "FALSE") && !options.compareArgs("FLUID VELOCITY SOLVER", "NONE")) {
      fields.push_back("fluid velocity");
      fields.push_back("fluid pressure");
   }

    int Nscalar = 0;
    options.getArgs("NUMBER OF SCALARS", Nscalar);
    for (int i = 0; i < Nscalar; i++) {
      const auto sid = scalarDigitStr(i);
      if (!options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
        fields.push_back("scalar" + sid);
      }
    }
    return fields;
  };

  bool fieldsToSolveContains(const std::string& value) {
    return std::find(fieldsToSolve().begin(), fieldsToSolve().end(), value) != fieldsToSolve().end();
  };

  void saveSolutionState();
  void restoreSolutionState();

  void computeUrst();

  preFluid_t preFluid = nullptr;
  postScalar_t postScalar = nullptr;

  void addUserCheckpointField(const std::string &name, const std::vector<deviceMemory<dfloat>> &o_fld)
  {
    std::vector<occa::memory> o_fld_;
    for (const auto &entry : o_fld) {
      o_fld_.push_back(entry);
    }

    userCheckpointFields.push_back({name, o_fld_});
  };

  userDivergence_t userDivergence = nullptr;

  mesh_t *meshV = nullptr;
  mesh_t *meshT = nullptr;

  QQt *qqt = nullptr;
  QQt *qqtT = nullptr;

  int Nscalar = 0;

  dlong fieldOffset = -1;

  dfloat g0 = NAN;

  occa::memory o_coeffEXT, o_coeffBDF;

  dfloat p0th[3] = {0.0, 0.0, 0.0};
  dfloat dp0thdt = 0;
  dfloat alpha0Ref = 1;

  int outerCorrector = 1;
  int outputForceStep = 0;

  int advectionSubcycingSteps = 0;

  nrs_t();

  std::unique_ptr<fluidSolver_t> fluid = nullptr;
  std::unique_ptr<geomSolver_t> geom = nullptr;
  std::unique_ptr<scalar_t> scalar = nullptr;

  dfloat flowRateScaleFactor();

  std::unique_ptr<iofld> checkpointWriter = nullptr;

  void evaluateProperties(const double time);
  void evaluateDivergence(const double time);

  AeroForce *aeroForces(const occa::memory &o_bID, const occa::memory &o_Sij = o_NULL);

  occa::memory viscousShearStress(const occa::memory o_bID, occa::memory o_Sij_= o_NULL);

  // output in row-major order
  occa::memory strainRotationRate(dlong offset, const occa::memory &o_U, bool smooth = true);
  occa::memory strainRotationRate(bool smooth = true);
  occa::memory strainRate(dlong offset, const occa::memory &o_U, bool smooth = true);
  occa::memory strainRate(bool smooth = true);

  void Qcriterion(occa::memory &o_Q);
  void Qcriterion(dlong offset, const occa::memory &o_U, occa::memory &o_Q);
  occa::memory Qcriterion(dlong offset, const occa::memory &o_U);
  occa::memory Qcriterion();

  void restartFromFiles(const std::vector<std::string>& list);

  int lastStepLocalSession(double timeNew, int tstep, double elapsedTime);

  void copyToNek(double time, int tstep, bool updateMesh = false);
  void copyToNek(double time, bool updateMesh = false);

  void copyFromNek(double &time);
  void copyFromNek();
  void getICFromNek();

  class tavgLegacy_t 
  {
  public:
    tavgLegacy_t();

    void writeToFile(mesh_t *mesh);
    void reset();
    void run(double time);

    const deviceMemory<double> o_avg();
    const deviceMemory<double> o_rms();
    const deviceMemory<double> o_rm2();

  private:
    std::unique_ptr<tavg> _avg;
    std::unique_ptr<tavg> _rms;
    std::unique_ptr<tavg> _rm2;
  };
  friend class nrs_t::tavgLegacy_t; 
  std::unique_ptr<nrs_t::tavgLegacy_t> tavgLegacy = nullptr;

  class bdry : public bdryBase
  {
  public:
    bdry()
    {
      importFromNek = true;
    };

    void setup() override;
    bool useDerivedGeomBoundaryConditions();

  private:
    void deriveGeomBoundaryConditions(std::vector<std::string> velocityBCs);
    bool geomConditionsDerived = false;
  };

  bdry bc;

private:
  void initInnerStep(double time, dfloat dt, int tstep);
  bool runInnerStep(userConvergenceCheck_t f, int stage, bool outerConverged);
  void finishInnerStep();

  void initOuterStep(double time, dfloat dt, int tstep);
  void runOuterStep(userConvergenceCheck_t f, int stage);
  void finishOuterStep();

  int tStepOuterStart;
  double timeOuterStart;

  void flowRatePrintInfo(int tstep, bool verboseInfo);
  void adjustFlowRate(int tstep, double time);
  void computeHomogenousStokesSolution(double time);
  void computeBaseFlowRate(double time, int tstep);

  int numberActiveFields();

  void setIC();

  void setTimeIntegrationCoeffs(int tstep);

  double timePrevious;

  dfloat computeCFL();
  dfloat computeCFL(dfloat dt);
  dfloat computeCFL(mesh_t *mesh, const occa::memory &o_U, dfloat dt);

  void setupNeknek();

  std::vector<std::pair<std::string, std::vector<occa::memory>>> userCheckpointFields;

  std::vector<int> createEllipticEToB(std::string field, mesh_t *mesh, std::string fieldComponent = "");
};

#endif
