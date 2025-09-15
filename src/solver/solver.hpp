#if !defined(nrs_solver_hpp_)
#define nrs_solver_hpp_

#include "platform.hpp"
#include "app.hpp"
#include "linAlg.hpp"
#include "elliptic.hpp"
#include "opSEM.hpp"

class solverCfg_t
{
public:
  std::string name;
  mesh_t *mesh;

  dfloat *g0;
  dfloat *dt;

  occa::memory o_coeffEXT;
  occa::memory o_coeffBDF;
};

class solver_t
{
public:
  virtual void solve(double time, int stage) = 0;
  virtual void lagSolution() = 0;
  virtual void extrapolateSolution() = 0;
  virtual void saveSolutionState() = 0;
  virtual void restoreSolutionState() = 0;
  virtual void applyDirichlet(double time) = 0;
  virtual void setupEllipticSolver() = 0;
  virtual void finalize() = 0;
  virtual void setTimeIntegrationCoeffs(int tstep) = 0;

  virtual deviceMemory<dfloat> o_solution(std::string key = "") = 0;
  virtual deviceMemory<dfloat> o_explicitTerms(std::string key = "") = 0;
  virtual deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") = 0;
  virtual deviceMemory<dfloat> o_transportCoeff(std::string key = "") = 0;

  std::string name;

  occa::memory o_prop;
  occa::memory o_EXT;
  occa::memory o_JwF;

  std::vector<elliptic *> ellipticSolver;

  occa::memory o_EToB;

  dfloat *g0 = nullptr;
  dfloat *dt = nullptr;
  occa::memory o_coeffEXT;
  occa::memory o_coeffBDF;

  dlong fieldOffsetSum;
  std::map<std::string, int> nameToIndex; 

private:

};

#endif
