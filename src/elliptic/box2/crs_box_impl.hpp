#if !defined(_CRS_BOX_IMPL_HPP_)
#define _CRS_BOX_IMPL_HPP_

#include "crs_box.hpp"

#define BOX_DOMAIN_SWITCH(dom, macro)                                          \
  {                                                                            \
    switch (dom) {                                                             \
    case gs_double:                                                            \
      macro(double);                                                           \
      break;                                                                   \
    case gs_float:                                                             \
      macro(float);                                                            \
      break;                                                                   \
    }                                                                          \
  }

struct box {
  jl_opts opts; /* User configurable options */
  uint un, cn, sn, ncr; /* User size, compressed size, Schwarz size and 1D dofs */
  void *sx, *srhs; /* Schwarz/XXT work arrays on CPU */
  void *asm1; /* Pointer to the asm1 solver */
  void *asm2; /* Pointer to the asm2 solver */
  struct gs_data *gsh; /* dssum */
  struct comm global; /* communicators */
  buffer bfr; /* Buffers for gslib */
};

// Fetch neighbors API.
void fetch_nbrs_v3(unsigned *nei, slong *vids, double *mat, sint *wids,
                          unsigned nv, unsigned ndim, unsigned nw,
                          unsigned max_ne, MPI_Comm comm);

void box_debug(const int verbose, const char *fmt, ...);

// ASM1: CHOLMOD, redundant API interface.
struct cholmod_csr;
struct cholmod_csr *sparse_cholmod_factor(uint n, const uint *Arp,
                                          const uint *Aj, const void *A,
                                          gs_dom dom, buffer *bfr);
void sparse_cholmod_solve(void *x, struct cholmod_csr *factor, const void *r);
void sparse_cholmod_free(struct cholmod_csr *factor);

// ASM1: CHOLMOD API interface.
void asm1_cholmod_setup(struct csr *A, unsigned null_space, struct box *box);
void asm1_cholmod_solve(void *x, struct box *box, const void *r);
void asm1_cholmod_free(struct box *box);

// ASM1: GPU BLAS interface.
template <typename T>
void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box);
void asm1_gpu_solve(occa::memory &o_x, struct box *box, occa::memory &o_r);
void asm1_gpu_free(struct box *box);

#endif
