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

void ellipticAx(elliptic_t *elliptic,
                dlong NelementsList,
                const occa::memory &o_elementsList,
                const occa::memory &o_q,
                occa::memory &o_Aq)
{
  if (NelementsList == 0) {
    return;
  }

  if (elliptic->userAx) {
    elliptic->userAx(elliptic, NelementsList, o_elementsList, o_q, o_Aq);
    return;
  }

  auto& mesh = elliptic->mesh;

  auto& o_geom_factors = elliptic->stressForm ? mesh->o_vgeo : mesh->o_ggeo;
  auto& o_D = mesh->o_D;
  auto& o_DT = mesh->o_DT;
  auto& o_lambda0 = elliptic->o_lambda0;
  auto o_lambda1 = (elliptic->poisson) ? o_NULL : elliptic->o_lambda1;

  auto loadKernel = [&]() {
    std::string kernelNamePrefix = (elliptic->poisson) ? "poisson-" : "";
    kernelNamePrefix += "elliptic";
    if (elliptic->Nfields > 1) {
      kernelNamePrefix += (elliptic->stressForm) ? "Stress" : "Block";
    }
    std::string kernelName = "Ax";
    if (elliptic->mgLevel) {
      if (elliptic->options.compareArgs("ELLIPTIC PRECO COEFF FIELD", "TRUE")) {
        kernelName += "Var";
      }
    } else {
       if (elliptic->options.compareArgs("ELLIPTIC COEFF FIELD", "TRUE")) {
         kernelName += "Var";
       }
    }
    kernelName += "Coeff";
    if (elliptic->options.compareArgs("ELEMENT MAP", "TRILINEAR")) {
      kernelName += "Trilinear";
    }

    auto gen_suffix = [&](const int N) 
    {      
      std::string dataType;
      if (o_Aq.dtype() == occa::dtype::get<double>()) {
         dataType = "double";
      } else if (o_Aq.dtype() == occa::dtype::get<float>()) {  
        dataType = "float";
      }
 
      return std::string("_") + std::to_string(N) + dataType;
    };

    kernelName += "Hex3D" + gen_suffix(elliptic->mesh->N);

#if 0
    if (platform->comm.mpiRank() == 0 && platform->verbose()) {
      std::cout << kernelNamePrefix + "Partial" + kernelName << std::endl;
    }
#endif 
    return platform->kernelRequests.load(kernelNamePrefix + "Partial" + kernelName);
  };

  if (!elliptic->AxKernel.isInitialized()) elliptic->AxKernel = loadKernel();
  elliptic->AxKernel(NelementsList,
                     elliptic->fieldOffset,
                     elliptic->loffset,
                     o_elementsList,
                     o_geom_factors,
                     o_D,
                     o_DT,
                     o_lambda0,
                     o_lambda1,
                     o_q,
                     o_Aq);

  double flopCount = mesh->Np * 12 * mesh->Nq + 15 * mesh->Np;

#if 0
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "AxKernel: " << elliptic->AxKernel.binaryFilename() << std::endl;
  }
#endif

  if (elliptic->options.compareArgs("ELLIPTIC COEFF FIELD", "TRUE")) {
    flopCount += 3 * mesh->Np;
  } else {
    flopCount += 1 * mesh->Np;
  }

  if (!elliptic->poisson) {
    flopCount += (2 + 1) * mesh->Np;
  }

  if (elliptic->stressForm) {
    flopCount += (15 + 6) * mesh->Np;
  }

  flopCount *= elliptic->Nfields * static_cast<double>(NelementsList);

  const double FPfactor = (o_Aq.dtype() == occa::dtype::get<float>()) ? 0.5 : 1.0;
  platform->flopCounter->add("ellipticAx", FPfactor * flopCount);
}

void ellipticOperator(elliptic_t *elliptic,
                      const occa::memory &o_q,
                      occa::memory &o_Aq,
                      bool masked)
{
  auto& mesh = elliptic->mesh;
  auto& oogs = elliptic->oogsAx;

  const auto ogsDataType = [&]()
  {
    if (o_Aq.dtype() == occa::dtype::get<float>()) {
      return ogsFloat;
    } else {
      return ogsDouble;
    }
  }();

  const auto overlap = (oogs != elliptic->oogs);
  if (overlap) {

    ellipticAx(elliptic, mesh->NglobalGatherElements, mesh->o_globalGatherElementList, o_q, o_Aq);
    if (masked) {
      ellipticApplyMask(elliptic,
                        mesh->NglobalGatherElements,
                        elliptic->NmaskedGlobal,
                        mesh->o_globalGatherElementList,
                        elliptic->o_maskIdsGlobal,
                        o_Aq);
    }

    oogs::start(o_Aq, elliptic->Nfields, elliptic->fieldOffset, ogsDataType, ogsAdd, oogs);
    ellipticAx(elliptic, mesh->NlocalGatherElements, mesh->o_localGatherElementList, o_q, o_Aq);

    if (masked) {
      ellipticApplyMask(elliptic,
                        mesh->NlocalGatherElements,
                        elliptic->NmaskedLocal,
                        mesh->o_localGatherElementList,
                        elliptic->o_maskIdsLocal,
                        o_Aq);
    }

    oogs::finish(o_Aq, elliptic->Nfields, elliptic->fieldOffset, ogsDataType, ogsAdd, oogs);

  } else {

    ellipticAx(elliptic, mesh->Nelements, mesh->o_elementList, o_q, o_Aq);
    if (masked) {
      ellipticApplyMask(elliptic,
                        mesh->Nelements,
                        elliptic->Nmasked,
                        mesh->o_elementList,
                        elliptic->o_maskIds,
                        o_Aq);
    }
    oogs::startFinish(o_Aq, elliptic->Nfields, elliptic->fieldOffset, ogsDataType, ogsAdd, oogs);
  }
}
