#include "neknek.hpp"
#include "nrs.hpp"
#include <array>

void neknek_t::fixCoupledSurfaceFlux(const occa::memory &o_EToB, dlong fieldOffsetU, occa::memory &o_U)
{
  constexpr int nReduction = 2; // flux + area
  auto o_reduction = platform->deviceMemoryPool.reserve<dfloat>(nReduction * mesh->Nelements);

  launchKernel("neknek::computeFlux",
               mesh->Nelements,
               fieldOffsetU,
               mesh->o_sgeo,
               mesh->o_vmapM,
               o_EToB,
               o_U,
               o_reduction);

  std::vector<dfloat> reduction(o_reduction.size());
  o_reduction.copyTo(reduction.data());

  std::array<dfloat, nReduction> res;
  for (int fld = 0; fld < nReduction; fld++) {
    res[fld] = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      res[fld] += reduction[e + fld * mesh->Nelements];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, res.data(), res.size(), MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm());

  auto [flux, area] = res;
  nekrsCheck(isnan(area) || area <= 0, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Invalid area!");
  nekrsCheck(isnan(flux), MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Invalid flux!");

  const auto gamma = -flux / area;

  if (platform->verbose() && platform->comm.mpiRank() == 0) {
    printf("neknek::fixCoupledSurfaceFlux flux = %11.4e, area = %11.4e, gamma = %11.4e\n", flux, area, gamma);
  }

  launchKernel("neknek::fixSurfaceFlux",
               mesh->Nelements,
               fieldOffsetU,
               mesh->o_sgeo,
               mesh->o_vmapM,
               o_EToB,
               gamma,
               o_U);
}
