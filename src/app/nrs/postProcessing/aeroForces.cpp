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

  auto o_tangentialViscousTraction = viscousShearStress(o_bID, o_Sij); // tau dot n - ((tau dot n) dot n) * n
  auto o_normalViscousTraction = viscousNormalStress(o_bID, o_Sij); // ((tau dot n) dot n) * n

  const dlong Ntotal = o_tangentialViscousTraction.size() / mesh->dim;
  auto fvT = mesh->surfaceAreaMultiplyIntegrate(mesh->dim, Ntotal, o_bID, o_tangentialViscousTraction);
  auto fvN = mesh->surfaceAreaMultiplyIntegrate(mesh->dim, Ntotal, o_bID, o_normalViscousTraction);

  af->setViscousForce({fvT[0] + fvN[0], fvT[1] + fvN[1], fvT[2] + fvN[2]});
  af->setViscousForceNormal({fvN[0], fvN[1], fvN[2]});
  af->setViscousForceTangential({fvT[0], fvT[1], fvT[2]});

  auto o_P = af->p().isInitialized() ? af->p() : this->fluid->o_P.slice(0, mesh->Nlocal);
  auto fp = mesh->surfaceAreaNormalMultiplyIntegrate(o_bID, o_P);
  af->setPressureForce({fp[0], fp[1], fp[2]});

  return af;
}
