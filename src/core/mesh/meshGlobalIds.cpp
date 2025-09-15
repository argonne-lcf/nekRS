#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "platform.hpp"
#include "mesh.h"
#include "nekInterfaceAdapter.hpp"

struct parallelNode_t {
  int baseRank;
  hlong baseId;
};

// uniquely label each node with a global index, used for gatherScatter
static void meshNekParallelConnectNodes(mesh_t *mesh, bool numberInterior)
{
  const auto Nlocal = std::pow(mesh->N + 1, mesh->dim) * mesh->Nelements;

  mesh->globalIds = (hlong *)calloc(Nlocal, sizeof(hlong));
  hlong ngv = nek::set_glo_num(mesh->globalIds, mesh->N + 1, mesh->solid, numberInterior);
}

void meshGlobalIds(mesh_t *mesh, bool numberInterior)
{
  meshNekParallelConnectNodes(mesh, numberInterior);
}

void meshGlobalFaceIds(mesh_t *mesh)
{
  auto meshExt = new mesh_t();
  meshExt->Nelements = mesh->Nelements;
  meshExt->N = mesh->N + 2;
  meshExt->Nq = meshExt->N + 1;
  meshExt->Np = meshExt->Nq * meshExt->Nq * meshExt->Nq;
  meshExt->Nfp = meshExt->Nq * meshExt->Nq;
  meshExt->Nfaces = mesh->Nfaces;
  meshExt->solid = mesh->solid;

  meshGlobalIds(meshExt);

  const dlong localNodeCount = mesh->Nelements * mesh->Nfp * mesh->Nfaces;
  mesh->globalFaceIds = (hlong *)calloc(localNodeCount, sizeof(hlong));

  for (int e = 0; e < mesh->Nelements; e++) {
    for (int j = 0; j < mesh->Nq; j++) {
      for (int i = 0; i < mesh->Nq; i++) {
        const dlong sk0 = e * mesh->Nfp * mesh->Nfaces + 0 * mesh->Nfp + i + j * mesh->Nq;
        const dlong sk5 = e * mesh->Nfp * mesh->Nfaces + 5 * mesh->Nfp + i + j * mesh->Nq;

        const dlong idx0 = e * meshExt->Np + 0 * meshExt->Nfp + (j + 1) * meshExt->Nq + (i + 1);
        const dlong idx5 =
            e * meshExt->Np + (meshExt->Nq - 1) * meshExt->Nfp + (j + 1) * meshExt->Nq + (i + 1);

        mesh->globalFaceIds[sk0] = meshExt->globalIds[idx0];
        mesh->globalFaceIds[sk5] = meshExt->globalIds[idx5];
      }
    }

    for (int k = 0; k < mesh->Nq; k++) {
      for (int i = 0; i < mesh->Nq; i++) {
        const dlong sk1 = e * mesh->Nfp * mesh->Nfaces + 1 * mesh->Nfp + i + k * mesh->Nq;
        const dlong sk3 = e * mesh->Nfp * mesh->Nfaces + 3 * mesh->Nfp + i + k * mesh->Nq;

        const dlong idx1 = e * meshExt->Np + (k + 1) * meshExt->Nfp + 0 * meshExt->Nq + (i + 1);
        const dlong idx3 =
            e * meshExt->Np + (k + 1) * meshExt->Nfp + (meshExt->Nq - 1) * meshExt->Nq + (i + 1);

        mesh->globalFaceIds[sk1] = meshExt->globalIds[idx1];
        mesh->globalFaceIds[sk3] = meshExt->globalIds[idx3];
      }
    }

    for (int k = 0; k < mesh->Nq; k++) {
      for (int j = 0; j < mesh->Nq; j++) {
        const dlong sk2 = e * mesh->Nfp * mesh->Nfaces + 2 * mesh->Nfp + j + k * mesh->Nq;
        const dlong sk4 = e * mesh->Nfp * mesh->Nfaces + 4 * mesh->Nfp + j + k * mesh->Nq;

        const dlong idx2 =
            e * meshExt->Np + (k + 1) * meshExt->Nfp + (j + 1) * meshExt->Nq + (meshExt->Nq - 1);
        const dlong idx4 = e * meshExt->Np + (k + 1) * meshExt->Nfp + (j + 1) * meshExt->Nq + 0;

        mesh->globalFaceIds[sk2] = meshExt->globalIds[idx2];
        mesh->globalFaceIds[sk4] = meshExt->globalIds[idx4];
      }
    }
  }

  if (meshExt->globalIds) free(meshExt->globalIds);
  delete meshExt;
}
