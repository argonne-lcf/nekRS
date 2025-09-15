#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "planarCopy.hpp"

void planarCopy::_setup(mesh_t *mesh_,
                        const occa::memory &o_Uin_,
                        const int nFields_,
                        const dlong fieldOffset_,
                        const int bID_,
                        occa::memory &o_out_)
{
  mesh = mesh_;
  fieldOffset = fieldOffset_;
  nFields = nFields_;
  o_U = o_Uin_;
  bID = bID_;
  o_out = o_out_;

  nekrsCheck(o_U.length() < nFields * fieldOffset,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "input o_U too small!\n");

  nekrsCheck(o_out.length() < nFields * fieldOffset,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "input o_out too small!\n");

  std::vector<int> tmp{bID};
  o_bID = platform->device.malloc<int>(tmp.size());
  o_bID.copyFrom(tmp.data());
}

void planarCopy::buildKernel(occa::properties kernelInfo)
{
  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/core/mesh/");
    const auto fileName = path + "planarCopy.okl";
    const auto reqName = "planarCopy::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  buildKernel("setBCValue");
  buildKernel("maskCopy");
}

void planarCopy::execute()
{
  auto o_wrk = platform->deviceMemoryPool.reserve<dfloat>(nFields * fieldOffset);

  if (interp) {
    const dlong offset = o_Uint.length() / nFields;
    interp->eval(nFields, fieldOffset, o_U, offset, o_Uint);

    platform->linAlg->fill(o_wrk.size(), 0, o_wrk);

    static occa::kernel kernel;
    if (!kernel.isInitialized()) {
      kernel = platform->kernelRequests.load("planarCopy::", "maskCopy");
    }
    kernel(interp->numPoints(), offset, nFields, fieldOffset, o_maskIds, o_Uint, o_wrk);
  } else {
    o_wrk.copyFrom(o_U, nFields * fieldOffset);

    const dfloat zero = 0.0;
    static occa::kernel kernel;
    if (!kernel.isInitialized()) {
      kernel = platform->kernelRequests.load("planarCopy::", "setBCValue");
    }
    kernel(mesh->Nelements, zero, bID, nFields, fieldOffset, o_wrk, mesh->o_vmapM, mesh->o_EToB);
    oogs::startFinish(o_wrk, nFields, fieldOffset, ogsDfloat, ogsAdd, ogs);
  }

  o_wrk.copyTo(o_out);
}

planarCopy::planarCopy(mesh_t *mesh,
                       const occa::memory &o_Uin_,
                       const int nFields_,
                       const dlong fieldOffset_,
                       const hlong eOffset,
                       const int bID_,
                       occa::memory &o_out_)
{
  _setup(mesh, o_Uin_, nFields_, fieldOffset_, bID_, o_out_);

  const dlong Ntotal = mesh->Np * mesh->Nelements;

  // establish a unique numbering
  // relies on a special global element numbering (extruded mesh)
  std::vector<hlong> ids(Ntotal);
  for (int e = 0; e < mesh->Nelements; e++) {
    auto eg = nek::localElementIdToGlobal(e);

    for (int n = 0; n < mesh->Np; n++) {
      ids[e * mesh->Np + n] = eg * mesh->Np + (n + 1);
    }

    for (int n = 0; n < mesh->Nfp * mesh->Nfaces; n++) {
      const int f = n / mesh->Nfp;
      const int idM = mesh->vmapM[e * mesh->Nfp * mesh->Nfaces + n];
      if (mesh->EToB[f + e * mesh->Nfaces] == bID) {
        ids[idM] += eOffset * mesh->Np;
      }
    }
  }

  ogs = oogs::setup(Ntotal,
                    ids.data(),
                    nFields,
                    0,
                    ogsDfloat,
                    platform->comm.mpiComm(),
                    0,
                    platform->device.occaDevice(),
                    NULL,
                    OOGS_AUTO);
}

planarCopy::planarCopy(mesh_t *mesh,
                       const occa::memory &o_Uin,
                       const int nFields_,
                       const dlong fieldOffset_,
                       const dfloat xOffset,
                       const dfloat yOffset,
                       const dfloat zOffset,
                       const int bID_,
                       occa::memory &o_out_)
{
  _setup(mesh, o_Uin, nFields_, fieldOffset_, bID_, o_out_);

  const auto nPoints = [&]() {
    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int n = 0; n < mesh->Nfp * mesh->Nfaces; n++) {
        const int f = n / mesh->Nfp;
        if (mesh->EToB[f + e * mesh->Nfaces] == bID) {
          cnt++;
        }
      }
    }
    return cnt;
  }();

  o_Uint = platform->device.malloc<dfloat>(nFields * alignStride<dfloat>(nPoints));
  o_maskIds = platform->device.malloc<dlong>(nPoints);

  std::vector<dlong> maskIds(nPoints);
  std::vector<dfloat> xBid(nPoints);
  std::vector<dfloat> yBid(nPoints);
  std::vector<dfloat> zBid(nPoints);

  {
    const auto [x, y, z] = mesh->xyzHost();

    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int n = 0; n < mesh->Nfp * mesh->Nfaces; n++) {
        const auto f = n / mesh->Nfp;
        const auto idM = mesh->vmapM[e * mesh->Nfp * mesh->Nfaces + n];
        if (mesh->EToB[f + e * mesh->Nfaces] == bID) {
          maskIds[cnt] = idM;
          xBid[cnt] = x[idM] + xOffset;
          yBid[cnt] = y[idM] + yOffset;
          zBid[cnt] = z[idM] + zOffset;
          cnt++;
        }
      }
    }
  }
  o_maskIds.copyFrom(maskIds.data());
  auto o_xBid = platform->device.malloc<dfloat>(nPoints, xBid.data());
  auto o_yBid = platform->device.malloc<dfloat>(nPoints, yBid.data());
  auto o_zBid = platform->device.malloc<dfloat>(nPoints, zBid.data());

  interp = new pointInterpolation_t(mesh, platform->comm.mpiComm());
  interp->setPoints(o_xBid, o_yBid, o_zBid);
  const auto verbosity = platform->verbose() ? pointInterpolation_t::VerbosityLevel::Detailed
                                             : pointInterpolation_t::VerbosityLevel::Basic;
  interp->find(verbosity);
}
