#include <elliptic.h>
#include <ellipticApplyMask.hpp>

void ellipticApplyMask(elliptic_t *solver, occa::memory &o_x)
{
  auto mesh = solver->mesh;
  ellipticApplyMask(solver,
                    mesh->Nelements,
                    solver->Nmasked,
                    mesh->o_elementList,
                    solver->o_maskIds,
                    o_x);
}

void ellipticApplyMask(elliptic_t *solver,
                       dlong Nelements,
                       dlong Nmasked,
                       const occa::memory &o_elementList,
                       const occa::memory &o_maskIds,
                       occa::memory &o_x)
{
  auto mesh = solver->mesh;

  if (solver->applyZeroNormalMask) {
    solver->applyZeroNormalMask(Nelements, o_elementList, o_x);
  }

  if (o_x.dtype() == occa::dtype::get<double>()) {
    platform->linAlg->mask<double>(Nmasked, o_maskIds, o_x);
  } else if (o_x.dtype() == occa::dtype::get<float>()) {
    platform->linAlg->mask<float>(Nmasked, o_maskIds, o_x);
  }
}
