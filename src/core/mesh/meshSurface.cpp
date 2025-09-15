#include "platform.hpp"
#include <mesh.h>

namespace
{

dfloat *sum;
dfloat *sumFace;
occa::memory o_sumFace;
occa::memory h_sumFace;

std::vector<dfloat> integral(mesh_t *mesh,
                             int Nfields,
                             int fieldOffset,
                             int mode,
                             const occa::memory o_bID,
                             const occa::memory &o_fld)
{
  if (o_sumFace.length() < Nfields * mesh->Nelements) {
    if (o_sumFace.byte_size()) {
      o_sumFace.free();
    }
    o_sumFace = platform->device.malloc<dfloat>(Nfields * mesh->Nelements);
    if (h_sumFace.length()) {
      h_sumFace.free();
    }
    h_sumFace = platform->device.mallocHost<dfloat>(Nfields * mesh->Nelements);
    sumFace = (dfloat *)h_sumFace.ptr();

    if (sum) {
      free(sum);
    }
    sum = (dfloat *)calloc(Nfields * mesh->Nelements, sizeof(dfloat));
  }

  auto kernel = [&]() {
    if (mode == 0) {
      return platform->kernelRequests.load("mesh-surfaceAreaMultiplyIntegrateHex3D");
    } else if (mode == 1) {
      return platform->kernelRequests.load("mesh-surfaceAreaNormalMultiplyVectorIntegrateHex3D");
    } else if (mode == 2) {
      nekrsCheck(Nfields != mesh->dim, MPI_COMM_SELF, EXIT_FAILURE, "%s", "invalid Nfields for mode2\n");
      return platform->kernelRequests.load("mesh-surfaceAreaNormalMultiplyIntegrateHex3D");
    } else {
      return occa::kernel();
    }
  }();

  platform->linAlg->fill(o_sumFace.size(), 0.0, o_sumFace);
  kernel(mesh->Nelements,
         Nfields,
         fieldOffset,
         static_cast<int>(o_bID.size()),
         o_bID,
         mesh->o_sgeo,
         mesh->o_vmapM,
         mesh->o_EToB,
         o_fld,
         o_sumFace);

  o_sumFace.copyTo(sumFace, Nfields * mesh->Nelements);

  for (int j = 0; j < Nfields; ++j) {
    sum[j] = 0;
    for (int i = 0; i < mesh->Nelements; ++i) {
      sum[j] += sumFace[i + j * mesh->Nelements];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, sum, Nfields, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm());

  std::vector<dfloat> out;
  for (int i = 0; i < Nfields; ++i) {
    out.push_back(sum[i]);
  }

  return out;
}

} // namespace

std::vector<dfloat> mesh_t::surfaceAreaMultiplyIntegrate(int Nfields,
                                                         dlong fieldOffset,
                                                         const occa::memory &o_bID,
                                                         const occa::memory &o_fld)
{
  nekrsCheck(o_fld.size() != (Nfields * ((Nfields > 1) ? fieldOffset : Nlocal)), 
             MPI_COMM_SELF, EXIT_FAILURE, "%s", "invalid input field size\n");
  return integral(this, Nfields, fieldOffset, 0, o_bID, o_fld);
}

dfloat mesh_t::surfaceAreaMultiplyIntegrate(const occa::memory &o_bID, const occa::memory &o_fld)
{
  return surfaceAreaMultiplyIntegrate(1, 0, o_bID, o_fld).at(0);
}

dfloat mesh_t::surfaceAreaNormalMultiplyVectorIntegrate(dlong fieldOffset,
                                                        const occa::memory &o_bID,
                                                        const occa::memory &o_fld)
{
  nekrsCheck(o_fld.size() != (dim * fieldOffset), MPI_COMM_SELF, EXIT_FAILURE, "%s", "invalid input field size\n");
  return integral(this, 1, fieldOffset, 1, o_bID, o_fld).at(0);
}

std::vector<dfloat> mesh_t::surfaceAreaNormalMultiplyIntegrate(const occa::memory &o_bID,
                                                               const occa::memory &o_fld)
{
  nekrsCheck(o_fld.size() != Nlocal, MPI_COMM_SELF, EXIT_FAILURE, "%s", "invalid input field size\n");
  return integral(this, dim, fieldOffset, 2, o_bID, o_fld);
}

occa::memory mesh_t::surfaceAreaMultiply(const occa::memory &o_bID, const occa::memory &o_fld)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(this->Nlocal);

  auto kernel = platform->kernelRequests.load("mesh-surfaceAreaMultiplyHex3D");
  nekrsCheck(o_fld.size() != Nlocal, MPI_COMM_SELF, EXIT_FAILURE, "%s", "invalid input field size\n");
  kernel(this->Nelements, static_cast<int>(o_bID.size()), o_bID, this->o_sgeo, this->o_vmapM, this->o_EToB, o_fld, o_out);

  return o_out;
}
