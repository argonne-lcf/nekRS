#ifndef XXT_H
#define XXT_H

#include <mpi.h>
#include "gslib.h"


#ifdef __cplusplus
extern "C" {
#endif

/* factors: L is in CSR format
            D is a diagonal matrix stored as a vector
   actual factorization is:

                  -1      T
     A   = (I-L) D   (I-L)

      -1        -T        -1
     A   = (I-L)   D (I-L)

   (triangular factor is unit diagonal; the diagonal is not stored)
*/
struct sparse_cholesky {
  uint n, *Lrp, *Lj;
  double *L, *D;
};

struct csr_mat {
  uint n, *Arp, *Aj; double *A;
};


struct xxt {

  /* communication */

  struct comm comm;
  uint pcoord;   /* coordinate in communication tree */ 
  unsigned plevels; /* # of stages of communication */
  sint *pother;     /* let p = pother[i], then during stage i of fan-in,
                           if p>=0, receive from p
                           if p< 0, send to (-p-1)
                       fan-out is just the reverse ...
                       on proc 0, pother is never negative
                       on others, pother is negative for the last stage only */
  comm_req *req;
  
  /* separators */

  unsigned nsep;  /* number of separators */
  uint *sep_size; /* # of dofs on each separator,
                     ordered from the bottom to the top of the tree:
                     separator 0 is the bottom-most one (dofs not shared)
                     separator nsep-1 is the root of the tree */

  unsigned null_space;
  double *share_weight;

  /* vector sizes */

  uint un;        /* user's vector size */
  
  /* xxt_solve works with "condensed" vectors;
     same dofs as in user's vectors, but no duplicates and no Dirichlet nodes,
     and also ordered topologically (children before parents) according to the
     separator tree */
  
  uint cn;        /* size of condensed vectors */
  sint *perm_u2c; /* permutation from user vector to condensed vector,
                     p=perm_u2c[i]; xu[i] = p=-1 ? 0 : xc[p];          */
  uint ln, sn;    /* xc[0 ... ln-1] are not shared   (ln=sep_size[0])
                     xc[ln ... ln+sn-1] are shared
                     ln+sn = cn                    */
  
  uint xn;        /* # of columns of x = sum_i(sep_size[i]) - sep_size[0] */

  /* data */
  struct sparse_cholesky fac_A_ll;
  struct csr_mat             A_sl;
  uint *Xp; double *X;   /* column i of X starts at X[Xp[i]] */
  
  /* execution buffers */
  double *vl, *vc, *vx, *combuf;
};

struct xxt *xxt_setup(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  uint null_space, const struct comm *comm);
void xxt_solve(double *x, struct xxt *data, const double *b);
void xxt_stats(struct xxt *data);
void xxt_free(struct xxt *data);

#ifdef __cplusplus
}
#endif

#endif
