#if !defined(nekrs_opSEM_hpp_)
#define nekrs_opSEM_hpp_

#include "platform.hpp"
#include "mesh.h"

/*
  unless noted otherwise, operators are based on weak formulation
  strong operators are smoothed or weighted by Jw 
*/

namespace opSEM 
{

void grad(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory& o_out);
occa::memory grad(mesh_t *mesh, dlong offset, const occa::memory &o_in);

void strongGrad(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory& o_out, bool avg = true);
occa::memory strongGrad(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg = true);

// output in row-major order
void strongGradVec(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory& o_out, bool avg = true);
occa::memory strongGradVec(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg = true);

void divergence(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory& o_out);
occa::memory divergence(mesh_t *mesh, dlong offset, const occa::memory &o_in);

void strongDivergence(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory& o_out, bool avg = true);
occa::memory strongDivergence(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg = true);

void laplacian(mesh_t *mesh, dlong offset, const occa::memory &o_lambda, const occa::memory &o_in, occa::memory& o_out);
occa::memory laplacian(mesh_t *mesh, dlong offset, const occa::memory &o_lambda, const occa::memory &o_in);

void strongLaplacian(mesh_t *mesh, dlong offset, const occa::memory &o_lambda, const occa::memory &o_in, occa::memory& o_out, bool avg = true);
occa::memory strongLaplacian(mesh_t *mesh, dlong offset, const occa::memory &o_lambda, const occa::memory &o_in, bool avg = true);

void strongCurl(mesh_t *mesh, dlong offset, const occa::memory& o_in, occa::memory& o_out, bool avg = true);
occa::memory strongCurl(mesh_t *mesh, dlong offset, const occa::memory& o_in, bool avg = true);

}

#endif
