/*

   The MIT License (MIT)

   Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

 */

#include "platform.hpp"
#include "elliptic.h"
#include "ellipticPrecon.h"
#include "maskedFaceIds.hpp"

namespace
{

std::string gen_suffix(const elliptic_t *elliptic, const char *floatString)
{
  const std::string precision = std::string(floatString);
  if (precision.find(pfloatString) != std::string::npos) {
    return std::string("_") + std::to_string(elliptic->mesh->N) + std::string("pfloat");
  } else {
    return std::string("_") + std::to_string(elliptic->mesh->N);
  }
}

} // namespace

elliptic_t *ellipticBuildMultigridLevel(elliptic_t *fineElliptic, int Nc, int Nf)
{
  auto elliptic = new elliptic_t(*fineElliptic);

  elliptic->mgLevel = true;
  elliptic->AxKernel = occa::kernel();

  auto mesh = createMeshMG(fineElliptic->mesh, Nc);
  elliptic->mesh = mesh;

  elliptic->fieldOffset = mesh->Nlocal; // assumes elliptic->Nfields == 1

  { // setup masked gs handle
    ogs_t *ogs = nullptr;
    const auto [Nmasked, o_maskIds, NmaskedLocal, o_maskIdsLocal, NmaskedGlobal, o_maskIdsGlobal] =
        maskedFaceIds(mesh,
                      mesh->Nlocal,
                      /* nFields */ 1,
                      /* offset */ 0,
                      elliptic->EToB,
                      ellipticBcType::DIRICHLET);

    elliptic->Nmasked = Nmasked;
    elliptic->o_maskIds = o_maskIds;
    elliptic->NmaskedLocal = NmaskedLocal;
    elliptic->o_maskIdsLocal = o_maskIdsLocal;
    elliptic->NmaskedGlobal = NmaskedGlobal;
    elliptic->o_maskIdsGlobal = o_maskIdsGlobal;

    if (!ogs) {
      nekrsCheck(elliptic->Nfields > 1,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s\n",
                 "Creating a masked gs handle for nFields > 1 is currently not supported!");

      std::vector<hlong> maskedGlobalIds(mesh->Nlocal);
      memcpy(maskedGlobalIds.data(), mesh->globalIds, mesh->Nlocal * sizeof(hlong));
      std::vector<dlong> maskIds(Nmasked);
      o_maskIds.copyTo(maskIds.data());
      for (dlong n = 0; n < Nmasked; n++) {
        maskedGlobalIds[maskIds[n]] = 0;
      }
      ogs = ogsSetup(mesh->Nlocal,
                     maskedGlobalIds.data(),
                     platform->comm.mpiComm(),
                     1,
                     platform->device.occaDevice());
    }
    elliptic->ogs = ogs;
    std::vector<pfloat> tmp(elliptic->mesh->Nlocal);
    for (int i = 0; i < elliptic->mesh->Nlocal; i++) {
      tmp[i] = (pfloat)elliptic->ogs->invDegree[i];
    }
    elliptic->o_invDegree = platform->device.malloc<pfloat>(elliptic->mesh->Nlocal);
    elliptic->o_invDegree.copyFrom(tmp.data());
  }

  const std::string suffix = "Hex3D";

  std::string kernelName;

  MPI_Barrier(platform->comm.mpiComm());
  double tStartLoadKernel = MPI_Wtime();

  ellipticBuildMultigridLevelKernels(elliptic);

  elliptic->precon = new precon_t();
  precon_t *precon = elliptic->precon;

  {
    const std::string kernelSuffix =
        std::string("_Nf_") + std::to_string(Nf) + std::string("_Nc_") + std::to_string(Nc);

    kernelName = "elliptic::coarsen" + suffix + kernelSuffix;
    precon->coarsenKernel = platform->kernelRequests.load(kernelName);

    kernelName = "elliptic::prolongate" + suffix + kernelSuffix;
    precon->prolongateKernel = platform->kernelRequests.load(kernelName);
  }

  elliptic->o_lambda0 = platform->device.malloc<pfloat>(mesh->Nlocal);
  if (fineElliptic->poisson) {
    elliptic->o_lambda1 = nullptr;
  } else {
    elliptic->o_lambda1 = platform->device.malloc<pfloat>(mesh->Nlocal);
  }

  const int Nfq = Nf + 1;
  const int Ncq = Nc + 1;
  auto fToCInterp = (dfloat *)calloc(Nfq * Ncq, sizeof(dfloat));
  InterpolationMatrix1D(Nf, Nfq, fineElliptic->mesh->r, Ncq, mesh->r, fToCInterp);

  occa::memory o_interp = platform->device.malloc<dfloat>(Nfq * Ncq, fToCInterp);
  elliptic->o_interp = platform->device.malloc<pfloat>(Nfq * Ncq);
  platform->copyDfloatToPfloatKernel(Nfq * Ncq, o_interp, elliptic->o_interp);

  precon->coarsenKernel(mesh->Nelements, elliptic->o_interp, fineElliptic->o_lambda0, elliptic->o_lambda0);
  if (!fineElliptic->poisson) {
    precon->coarsenKernel(mesh->Nelements, elliptic->o_interp, fineElliptic->o_lambda1, elliptic->o_lambda1);
  }

  free(fToCInterp);

  MPI_Barrier(platform->comm.mpiComm());
  if (platform->comm.mpiRank() == 0) {
    printf("done (%gs)\n", MPI_Wtime() - tStartLoadKernel);
  }
  fflush(stdout);

  return elliptic;
}
