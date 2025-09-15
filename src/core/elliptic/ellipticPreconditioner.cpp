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

#include "elliptic.h"
#include "ellipticPrecon.h"
#include "timer.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

void ellipticPreconditioner(elliptic_t *elliptic, const occa::memory &o_r, occa::memory &o_z)
{
  auto mesh = elliptic->mesh;
  auto precon = elliptic->precon;
  auto &options = elliptic->options;

  if (options.compareArgs("PRECONDITIONER", "JACOBI")) {
    platform->linAlg
        ->axmyzMany(mesh->Nlocal, elliptic->Nfields, elliptic->fieldOffset, 1.0, o_r, precon->o_invDiagA, o_z);
    platform->flopCounter->add("jacobiPrecon", static_cast<double>(mesh->Nlocal) * elliptic->Nfields);
  } else if (options.compareArgs("PRECONDITIONER", "MULTIGRID")) {
    platform->linAlg->fill<pfloat>(elliptic->fieldOffset * elliptic->Nfields, 0.0, elliptic->o_zPfloat);
    platform->copyDfloatToPfloatKernel(elliptic->fieldOffset * elliptic->Nfields, o_r, elliptic->o_rPfloat);
    precon->MGSolver->Run(elliptic->o_rPfloat, elliptic->o_zPfloat);
    platform->copyPfloatToDfloatKernel(elliptic->fieldOffset * elliptic->Nfields, elliptic->o_zPfloat, o_z);
  } else if (options.compareArgs("PRECONDITIONER", "SEMFEM")) {
    platform->linAlg->fill<pfloat>(elliptic->fieldOffset * elliptic->Nfields, 0.0, elliptic->o_zPfloat);
    platform->copyDfloatToPfloatKernel(elliptic->fieldOffset * elliptic->Nfields, o_r, elliptic->o_rPfloat);
    precon->SEMFEMSolver->run(elliptic->o_rPfloat, elliptic->o_zPfloat);
    platform->copyPfloatToDfloatKernel(elliptic->fieldOffset * elliptic->Nfields, elliptic->o_zPfloat, o_z);
  } else if (options.compareArgs("PRECONDITIONER", "NONE")) {
    o_z.copyFrom(o_r, elliptic->fieldOffset * elliptic->Nfields);
  } else {
    if (platform->comm.mpiRank() == 0) {
      printf("ERROR: Unknown preconditioner\n");
    }
    MPI_Abort(platform->comm.mpiComm(), 1);
  }
}
