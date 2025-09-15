#ifndef AERO_FORCE_HPP_
#define AERO_FORCE_HPP_ 

#include "platform.hpp"

class AeroForce
{
public:
  AeroForce() {};
  AeroForce(std::tuple< std::array<dfloat, 3>, std::array<dfloat, 3> > f) 
  {
    forceV = std::get<0>(f);
    forceP = std::get<1>(f); 
  };

  std::array<dfloat, 3> tangential() const { return forceV; };
  void tangential(std::array<dfloat, 3> f) { forceV = f; }

  // neglecting normal viscous forces
  std::array<dfloat, 3> normal() const { return forceP; };
  void normal(std::array<dfloat, 3> f) { forceP = f; }

  std::array<dfloat, 3> forceEff() const 
  { 
     return {forceV[0] + forceP[0], forceV[1] + forceP[1], forceV[2] + forceP[2]}; 
  };

  void rho(const occa::memory& o_in) { o_rho = o_in; }
  occa::memory rho() { return o_rho; }

  void p(const occa::memory& o_in) { o_P = o_in; }
  occa::memory p() { return o_P; }

private:
  occa::memory o_rho;
  occa::memory o_P;

  std::array<dfloat, 3> forceV;
  std::array<dfloat, 3> forceP;
};

#endif
