#include "elliptic.h"
#include "ellipticPrecon.h"

void ellipticAllocateWorkspace(elliptic_t *elliptic)
{
  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      elliptic->precon->MGSolver->allocateWorkStorage();
    }
  }
}

void ellipticFreeWorkspace(elliptic_t *elliptic)
{
  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      elliptic->precon->MGSolver->freeWorkStorage();
    }
  }

  elliptic->o_rPfloat.free();
  elliptic->o_zPfloat.free();

  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      MGSolver_t::multigridLevel **levels = elliptic->precon->MGSolver->levels;
      for (int lev = 0; lev < elliptic->precon->MGSolver->numLevels; lev++) {
        auto level = dynamic_cast<pMGLevel *>(levels[lev]);
        level->o_work1.free();
        level->o_work2.free();
      }
    }
  }
}
