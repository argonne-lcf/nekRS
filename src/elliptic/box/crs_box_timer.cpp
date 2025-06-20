#include <stdio.h>

#include "nekrs_crs.hpp"
#include "crs_box_timer.hpp"

#define MAX_METRICS 32
static unsigned timer_on = 0;
static double start_time;
static double time_box[MAX_METRICS];

void timer_init() {
  for (unsigned i = 0; i < MAX_METRICS; i++)
    time_box[i] = 0;
  timer_on = 1;
}

void timer_tic(const struct comm *c) {
  if (!timer_on) return;
  comm_barrier(c);
  start_time = comm_time();
}

void timer_toc(BoxMetric m) {
  if (!timer_on || m == NONE) return;
  platform->device.finish();
  time_box[m] += (comm_time() - start_time);
}

void timer_print(MPI_Comm comm) {
  if (!timer_on) return;

  struct comm c;
  comm_init(&c, comm);

  double max[MAX_METRICS], wrk[2 * MAX_METRICS];
  for (unsigned i = 0; i < MAX_METRICS; i++)
    max[i] = time_box[i];
  comm_allreduce(&c, gs_double, gs_max, max, MAX_METRICS, wrk);

  if (c.id == 0) {
    printf("box copy_rhs          : %e\n", time_box[COPY_RHS]);
    printf("box u2c               : %e\n", time_box[U2C]);
    printf("box asm1              : %e\n", time_box[ASM1]);
    printf("box c2u               : %e\n", time_box[C2U]);
    printf("box mult_rhs_update   : %e\n", time_box[MULT_RHS_UPDATE]);
    printf("box map_vtx_to_box    : %e\n", time_box[MAP_VTX_TO_BOX]);
    printf("box asm2              : %e\n", time_box[ASM2]);
    printf("box map_box_to_vtx    : %e\n", time_box[MAP_BOX_TO_VTX]);
    printf("box copy_solution     : %e\n", time_box[COPY_SOLUTION]);
    printf("box zero              : %e\n", time_box[ZERO]);
    printf("box inv_mul           : %e\n", time_box[INV_MUL]);
    printf("box crs_dsavg         : %e\n", time_box[CRS_DSAVG1]);
  }
  fflush(stdout);

  comm_free(&c);
}

#undef MAX_METRICS
