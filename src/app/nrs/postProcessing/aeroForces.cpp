#include "nrs.hpp"
#include "aeroForce.hpp"

AeroForce *nrs_t::aeroForces(const occa::memory &o_bID, const occa::memory &o_Sij_)
{
  auto mesh = meshV;
  auto af = new AeroForce();

  occa::memory o_Sij = o_Sij_;
  if (!o_Sij.isInitialized()) {
    o_Sij = this->strainRate();
  }
  auto o_tauT = viscousShearStress(o_bID, o_Sij);
  auto fT = mesh->surfaceAreaMultiplyIntegrate(mesh->dim, o_tauT.size() / mesh->dim, o_bID, o_tauT); 
  af->tangential({fT[0], fT[1], fT[2]});

  auto o_rhoP = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->axmyz(o_rhoP.size(), 
                          1.0, 
                          af->rho().isInitialized() ? af->rho() : this->fluid->o_rho, 
                          af->p().isInitialized() ? af->p() : this->fluid->o_P, 
                          o_rhoP);
  auto fN = mesh->surfaceAreaNormalMultiplyIntegrate(o_bID, o_rhoP);
  af->normal({fN[0], fN[1], fN[2]});

  return af;
}
