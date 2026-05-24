#if !defined(nrs_constflow_hpp_)
#define nrs_constflow_hpp_

#include "platform.hpp"
#include "linAlg.hpp"
#include "elliptic.hpp"
#include "fluidSolver.hpp"

class flowRate_t
{
public:
  flowRate_t(fluidSolver_t *fluidRef);
  
  void printInfo(int tstep, bool verboseInfo) const;
  void adjust();
  bool computationRequired(double time, int tstep);
  void rhsPressure(double time, int iter);
  void solvePressure(double time, int iter);
  void rhsVelocity(double time, int iter);
  void solveVelocity(double time, int iter);
  dfloat scaleFactor() const;

private:
  fluidSolver_t *fluid;
};

#endif
