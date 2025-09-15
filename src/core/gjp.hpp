#if !defined(nekrs_gjp_hpp_)
#define nekrs_gjp_hpp_

#include "platform.hpp"
#include "mesh.h"
#include "opSEM.hpp"

static void addGJP(mesh_t *mesh,
                   const occa::memory &o_EToB,
                   const occa::memory &o_coef,
                   dlong fieldOffset,
                   const occa::memory &o_U,
                   const occa::memory &o_S,
                   occa::memory &o_out,
                   const dfloat NscalingFactor = 0.8)
{
  // (n * o_grad)
  auto o_grad = opSEM::strongGrad(mesh, fieldOffset, o_S, false);

  auto o_jump = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements * mesh->Nfaces * mesh->Nfp);
  auto o_h = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements * mesh->Nfaces);

  static occa::kernel gjpHelperKernel;
  if (!gjpHelperKernel.isInitialized()) {
    gjpHelperKernel = platform->kernelRequests.load("gjpHelperHex3D");
  }
  gjpHelperKernel(mesh->Nelements,
                  fieldOffset,
                  mesh->o_x,
                  mesh->o_y,
                  mesh->o_z,
                  mesh->o_vmapM,
                  mesh->o_sgeo,
                  o_grad,
                  o_h,
                  o_jump);

  static oogs_t *gshFace = nullptr;
  if (!gshFace) {
    gshFace = oogs::setup(o_jump.size(),
                          mesh->globalFaceIds,
                          1,
                          0,
                          ogsDfloat,
                          platform->comm.mpiComm(),
                          1,
                          platform->device.occaDevice(),
                          NULL,
                          OOGS_AUTO);
  }
  oogs::startFinish(o_jump, 1, 0, ogsDfloat, ogsAdd, gshFace);

  const dfloat tau = NscalingFactor / std::pow(mesh->Nq, 4);

  static occa::kernel gjpKernel;
  if (!gjpKernel.isInitialized()) {
    gjpKernel = platform->kernelRequests.load("gjpHex3D");
  }
  gjpKernel(mesh->Nelements,
            fieldOffset,
            tau,
            mesh->o_D,
            mesh->o_vmapM,
            mesh->o_vgeo,
            mesh->o_sgeo,
            o_EToB,
            o_coef,
            o_h,
            o_U,
            o_jump,
            o_out);
}

static void addGJP(mesh_t *mesh,
                   const occa::memory &o_EToB,
                   dlong fieldOffset,
                   const occa::memory &o_U,
                   const occa::memory &o_S,
                   occa::memory &o_out,
                   const dfloat NscalingFactor = 0.8)
{
  auto o_coef = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->fill(o_coef.size(), 1.0, o_coef);
  addGJP(mesh, o_EToB, o_coef, fieldOffset, o_U, o_S, o_out, NscalingFactor);
}

#endif
