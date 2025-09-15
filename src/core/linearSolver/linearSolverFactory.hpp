#if !defined(nrs_linearSolverFactory_hpp_)
#define nrs_linearSolverFactory_hpp_

#include "platform.hpp"
#include "linearSolver.hpp"

template <typename T = dfloat> class linearSolverFactory
{
public:
  static linearSolver*
  create(const std::string &_solver,
         const std::string &varName,
         dlong Nlocal,
         int Nfields,
         dlong fieldOffset,
         const occa::memory &o_weight,
         bool removeMean,
         std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> Ax,
         std::function<void(const occa::memory &o_r, occa::memory &o_z)> Pc = nullptr);
};

#endif
