#if !defined(nekrs_subcycle_hpp_)
#define nekrs_subcycle_hpp_

#include "nekrsSys.hpp"
#include "mesh.h"

void advectionSubcyclingRK(mesh_t *_mesh, mesh_t *meshV,
                           double time, dfloat *dt, int Nsubsteps, const occa::memory& o_coeffBDF, int nEXT,
                           int nFields, const occa::kernel& kernel,oogs_t *_gsh,
                           dlong _meshOffset, dlong _fieldOffset, dlong cubatureOffset, dlong fieldOffsetSum,
                           const occa::memory& o_divUMesh, const occa::memory& o_Urst, const occa::memory& o_U,
                           occa::memory& o_out);

#endif
