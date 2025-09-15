#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

void nrs_t::Qcriterion(dlong offset, const occa::memory &o_U, occa::memory &o_Q)
{
  auto o_SijOij = this->strainRotationRate(offset, o_U);

  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    kernel = platform->kernelRequests.load("nrs-Qcriterion");
  }
  kernel(meshV->Nlocal, offset, this->fluid->o_div, o_SijOij, o_Q);
}

void nrs_t::Qcriterion(occa::memory &o_Q)
{
  Qcriterion(fluid->fieldOffset, fluid->o_U, o_Q);
}

occa::memory nrs_t::Qcriterion(dlong offset, const occa::memory &o_U)
{
  auto o_Q = platform->deviceMemoryPool.reserve<dfloat>(meshV->Nlocal);
  Qcriterion(offset, o_U, o_Q);
  return o_Q;
}

occa::memory nrs_t::Qcriterion()
{
  auto o_Q = platform->deviceMemoryPool.reserve<dfloat>(meshV->Nlocal);
  Qcriterion(fluid->fieldOffset, fluid->o_U, o_Q);
  return o_Q;
}
