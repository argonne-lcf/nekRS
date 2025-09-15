#ifndef ELLIPTIC_HPP
#define ELLIPTIC_HPP 1

#include "nekrsSys.hpp"
#include "mesh.h"
#include "ellipticBcTypes.h"

struct elliptic_t;

class elliptic 
{
public:
  elliptic(const std::string& name, mesh_t *mesh, dlong fieldOffset, const std::vector<int>& EToBIn, const occa::memory& o_lambda0, const occa::memory& o_lambda1);

  elliptic(const std::string& name, mesh_t *mesh, dlong fieldOffset, const occa::memory& o_lambda0, const occa::memory& o_lambda1);


  ~elliptic();  

  void updatePreconditioner(); // to accomodate mesh or coeff changes

  void solve(const occa::memory& o_lambda0, const occa::memory& o_lambda1, const occa::memory& RHS, occa::memory x);

  std::string& name() const;
  setupAide& options();

  int Niter() const;
  void Niter(int val);

  dlong fieldOffset() const;

  bool nullSpace() const;

  dfloat initialResidual() const;
  void initialResidual(dfloat val);
  dfloat initialGuessResidual() const;
  void initialGuessResidual(dfloat val);
  dfloat finalResidual() const;
  void finalResidual(dfloat val);

  dlong Nmasked();
  occa::memory o_maskIds() const;
  occa::memory o_EToB() const;
  std::vector<int> EToB() const; 

  int Nfields() const; 

  void op(const occa::memory &o_q, occa::memory &o_Aq, bool masked = true); 
  void Ax(const occa::memory &o_lambda0In, const occa::memory &o_lambda1In, const occa::memory &o_q, occa::memory &o_Aq); 

  void applyZeroNormalMask(const std::function<void(dlong Nelements, const occa::memory &o_elementList, occa::memory &o_x)>& f);
  void userPreconditioner(const std::function<void(const occa::memory &o_r, occa::memory &o_z)>& f);
  void userAx(const std::function<void(elliptic_t *elliptic, dlong NelementsList, const occa::memory &o_elementsList, const occa::memory &o_x, occa::memory &o_Ax)>& f);

  std::tuple<int, int> projectionCounters() const; 

private:
  elliptic_t *solver;
  std::vector<int> _EToB;

  void _solve(const occa::memory &o_lambda0,
              const occa::memory &o_lambda1,
              const occa::memory &o_r,
              occa::memory o_x);
  
  void _setup(const occa::memory &o_lambda0, const occa::memory &o_lambda1);
};

#endif
