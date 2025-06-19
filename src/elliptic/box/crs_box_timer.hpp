#if !defined(_CRS_BOX_TIMER_HPP_)
#define _CRS_BOX_TIMER_HPP_

#include <gslib.h>

typedef enum {
  INV_MUL,
  COPY_RHS,
  CRS_DSAVG1,
  U2C,
  ASM1,
  ZERO,
  C2U,
  CRS_DSAVG2,
  MULT_RHS_UPDATE,
  COPY_TO_NEK5000,
  MAP_VTX_TO_BOX,
  ASM2,
  MAP_BOX_TO_VTX,
  COPY_FROM_NEK5000,
  CRS_DSAVG3,
  COPY_SOLUTION,
  NONE,
} BoxMetric;

void timer_init();
void timer_tic(const struct comm *c);
void timer_toc(const BoxMetric m);
void timer_print(MPI_Comm comm);

#endif
