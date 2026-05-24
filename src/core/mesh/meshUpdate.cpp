#include "mesh.h"
#include "linAlg.hpp"
#include "platform.hpp"

void mesh_t::update()
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "updating mesh ...\n";
  }

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

  if (o_sgeo.isInitialized()) {
    surfaceGeometricFactors();
  }

  if (o_cubvgeo.isInitialized()) {
    cubatureGeometricFactors();
  }
}
