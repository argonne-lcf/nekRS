#if !defined(nrs_linearSolver_hpp_)
#define nrs_linearSolver_hpp_

#include "platform.hpp"

struct CombinedPCGId {
  static constexpr int nReduction = 7;
  static constexpr int gamma = 0;
  static constexpr int a = 1;
  static constexpr int b = 2;
  static constexpr int c = 3;
  static constexpr int d = 4;
  static constexpr int e = 5;
  static constexpr int f = 6;
};

class linearSolver
{
public:
  virtual ~linearSolver() = default;

  // solve Ax = b (assumes zero initial guess, zeroing is done in solve) 
  virtual void 
  solve(const dfloat tol, const int maxIter, const occa::memory &o_b, occa::memory &o_x) = 0;

  occa::memory o_invDiagA;
  
  int nIter() const { return _nIter; }; 
  dfloat initialResidualNorm() const { return r0Norm; };
  dfloat finalResidualNorm() const { return rNorm; };

  void name(const std::string &val)
  {
    _name = val;
  };

protected:
  std::string _name;
  dlong Nlocal;
  int Nfields;
  dlong fieldOffset;
  std::string knlPrefix;
  double FPfactor;
  dfloat tiny;

  int _nIter;
  dfloat r0Norm;
  dfloat rNorm;
};

#endif
