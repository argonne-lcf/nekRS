#if !defined(nrs_motition_hpp_)
#define nrs_motition_hpp_

#include "solver.hpp"

struct geomSolverCfg_t : public solverCfg_t {
public:
  bool deriveBCFromVelocity;
  mesh_t *meshV;
  dlong fieldOffset;
};

class geomSolver_t : public solver_t
{
public:
  geomSolver_t(const geomSolverCfg_t &cfg);

  void lagSolution() override;
  void extrapolateSolution() override;

  void saveSolutionState() override;
  void restoreSolutionState() override;

  void applyDirichlet(double time) override;
  void setupEllipticSolver() override;

  void setTimeIntegrationCoeffs(int tstep) override;

  void solve(double time, int iter) override;

  void finalize() override
  {
    for (auto &entry : ellipticSolver) {
      delete entry;
      entry = nullptr;
    }
  };

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_U);

    auto it = nameToIndex.find(key);
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_U.slice(idx * fieldOffset, fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_prop);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_NULL);
  }

  void integrate(bool lag = true);

  void updateZeroNormalMask();

  void computeDiv();

  mesh_t *mesh = nullptr;
  mesh_t *meshV = nullptr;

  dlong fieldOffset = -1;

  occa::memory o_U;
  occa::memory o_Ue;

  occa::memory o_U0;
  occa::memory o_prop0;
  occa::memory o_invAJw0;
  occa::memory o_Jw0;
  occa::memory o_x0, o_y0, o_z0;

  occa::memory o_coeffAB;

  occa::memory o_Ufluid;

  occa::memory o_div;

private:
  occa::memory o_zeroNormalMask;
  bool deriveBCFromVelocity;
};

#endif
