#if !defined(nekrs_solve_hpp_)
#define nekrs_solve_hpp_

#include "nekrsSys.hpp"
#include "bdryBase.hpp"
#include "neknek.hpp"

class app_t {
  public:
    using userProperties_t = std::function<void(double)>;
    using userConvergenceCheck_t = std::function<bool(int)>;
    using userSource_t = std::function<void(double)>;

    virtual std::string id() const { return ""; };

    virtual const std::vector<std::string>& fieldsToSolve() const = 0; 
    virtual void printRunStat(int step) = 0;

    virtual void init() = 0;
    virtual void finalize() = 0;

    virtual dfloat adjustDt(int tstep) = 0;

    int checkpointStep= 0;

    virtual void writeCheckpoint(double t,
                                 bool enforceOutXYZ = false,
                                 bool enforceFP64 = false,
                                 int Nout = 0,
                                 bool uniform = false) = 0;


    virtual void printStepInfo(double time, int tstep, bool printStepInfo, bool printVerboseInfo) = 0;

    virtual void initStep(double time, dfloat dt, int tstep) = 0;

    int lastStep = 0;
    virtual int setLastStep(double timeNew, int tstep, double elapsedTime) = 0;

    userConvergenceCheck_t userConvergenceCheck = nullptr;
    virtual bool runStep(userConvergenceCheck_t f, int stage) = 0;

    int tstep = 0;
    dfloat dt[3] = {0};

    virtual void finishStep() = 0;
    virtual void registerKernels(occa::properties kernelInfoBC) = 0;
  
    virtual void setDefaultSettings(setupAide& options) = 0;

    virtual void printSolutionMinMax() = 0;

    int timeStepConverged = 1;

    userProperties_t userProperties = nullptr;

    userSource_t userSource = nullptr;
 
    std::unique_ptr<neknek_t> neknek = nullptr;
    bdryBase* bc = nullptr;
};

#endif
