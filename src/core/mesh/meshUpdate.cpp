#include "mesh.h"
#include "linAlg.hpp"
#include "platform.hpp"

void mesh_t::update()
{
  {
    auto retVal = geometricFactors();
    nekrsCheck(retVal > 0,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "Invalid element Jacobian < 0 found!");
  }

  volume = platform->linAlg->sum(Nlocal, o_LMM, platform->comm.mpiComm());

  computeInvLMM();

  surfaceGeometricFactors();
}
