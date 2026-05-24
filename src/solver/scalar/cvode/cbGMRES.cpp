/*
 * -----------------------------------------------------------------
 * Clone of CVODE's SPGMR solver with some modifications;
 * -----------------------------------------------------------------
 */

#ifdef ENABLE_CVODE

#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_math.h>

//#include "sundials_context_impl.h"
#include "sundials_logger_impl.h"

#include "nekrsSys.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

#include "cbGMRES.hpp"

#define ZERO SUN_RCONST(0.0)
#define ONE SUN_RCONST(1.0)

#define SPGMR_CONTENT(S) ((SUNLinearSolverContent_SPGMR)(S->content))
#define LASTFLAG(S) (SPGMR_CONTENT(S)->last_flag)

namespace
{

static const std::string timerName = "cvode_t::";

static occa::kernel axmyzKernel;
static occa::kernel axpbyKernel;
static occa::kernel linearCombinationKernel;
static occa::kernel innerProdMultiKernel;

static dlong N;

static occa::memory o_yg;
static occa::memory h_yg;

static occa::memory o_stemp;
static occa::memory h_stemp;

// o_z[n] = alpha*o_x[n]*o_y[n]
static void axmyz(const dfloat alpha,
                  const occa::memory &o_x,
                  const occa::memory &o_y,
                  occa::memory &o_z,
                  const dlong xOffset = 0)
{
  axmyzKernel(N, xOffset * N, alpha, o_x, o_y, o_z);
}

// o_y[n] = beta*o_y[n] + alpha*o_x[n]
static void axpby(const dfloat alpha,
                  const occa::memory &o_x,
                  const dfloat beta,
                  occa::memory &o_y,
                  const dlong xOffset = 0,
                  const dlong yOffset = 0)
{
  axpbyKernel(N, xOffset * N, yOffset * N, alpha, o_x, beta, o_y);
}

// o_z[n] = y_{Nfields} * c_{Nfields} + \sum_{i=0}^{Nfields-1} c_i * x_i
static void linearCombination(const dlong Nfields,
                              const occa::memory &o_coeff,
                              const occa::memory &o_x,
                              const occa::memory &o_y,
                              occa::memory &o_z)
{
  linearCombinationKernel(N, Nfields, N, o_coeff, o_x, o_y, o_z);
}

static void innerProdMulti(const dlong NVec,
                           const occa::memory &o_x,
                           const occa::memory &o_y,
                           MPI_Comm _comm,
                           dfloat *result,
                           const dlong yOffset = 0)
{
  const int Nblock = (N + BLOCKSIZE - 1) / BLOCKSIZE;

  auto o_scratch = platform->deviceMemoryPool.reserve(NVec * Nblock * sizeof(dfloat));
  auto h_scratch = platform->memoryPool.reserve(o_scratch.byte_size());

  innerProdMultiKernel(Nblock,
                       N,
                       N, // fieldOffset
                       NVec,
                       yOffset,
                       o_x,
                       o_y,
                       o_scratch);

  auto scratch = (sunrealtype *)h_scratch.ptr();
  o_scratch.copyTo(scratch, NVec * Nblock * sizeof(dfloat));

  for (int field = 0; field < NVec; ++field) {
    dfloat dot = 0;
    for (dlong n = 0; n < Nblock; ++n) {
      dot += scratch[n + field * Nblock];
    }
    result[field] = dot;
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, result, NVec, MPI_DFLOAT, MPI_SUM, _comm);
  }
}

/* iterative classical Gram-Schmidt */
static int CGSI(sunrealtype **h, int k, int p, sunrealtype *new_vk_norm, const occa::memory& o_V, occa::memory &o_omega)
{
  auto stemp = (sunrealtype *)h_stemp.ptr();

  const int k_minus_1 = k - 1;
  const int nIterMax = 2;
  const sunrealtype Kthres = 100;

  sunrealtype vk_norm0;

  for (int iter = 0; iter < nIterMax; iter++) {
    if (((*new_vk_norm) * Kthres < vk_norm0) || iter == 0) {
      platform->timer.tic(timerName + "solve::cvode::linearSolve::cgsi", 0);
      innerProdMulti(k + 1, o_V, o_omega, platform->comm.mpiComm(), stemp);
      const dfloat omegaDotp = stemp[k];
      vk_norm0 = SUNRsqrt(omegaDotp);

      dfloat sTempSqrSum = 0;
      for (int i = 0; i < k; i++) {
        sTempSqrSum += stemp[i] * stemp[i];
        h[i][k_minus_1] = (iter > 0) ? h[i][k_minus_1] + stemp[i] : stemp[i];
        stemp[i] = -stemp[i];
      }

      // o_omega = stemp_[k] * o_omega + sum (o_stemp_[i] * o_v[i]) for i = 1 ... k-1
      stemp[k] = ONE;
      o_stemp.copyFrom(stemp, k + 1);
      linearCombination(k + 1, o_stemp, o_V, o_omega, o_omega);

#if 1
      const auto arg = omegaDotp - sTempSqrSum;
      if (arg <= 0) {
        return 1;
      }
      *new_vk_norm = SUNRsqrt(arg);
#else
      *new_vk_norm = SUNRsqrt(platform->linAlg->innerProd(N, o_omega, o_omega, platform->comm.mpiComm()));
#endif
      platform->timer.toc(timerName + "solve::cvode::linearSolve::cgsi");

#if 0
      if(platform->comm.mpiRank() == 0 && platform->verbose()) {
        std::cout << "CGSI: vk_norm0/vk_norm= " << vk_norm0/(*new_vk_norm) << ", iter=" << iter << std::endl;
      }
#endif
    }
  }

  return 0;
}

#define cbGMRESFinish(lastFlag)                                                                              \
  {                                                                                                          \
    return lastFlag;                                                                                         \
  }

} // namespace

void cbGMRESSetup(SUNLinearSolver S)
{
  const auto l_max = SPGMR_CONTENT(S)->maxl;

  static_assert(sizeof(sunrealtype) == sizeof(dfloat), "sunrealtype has to match dfloat!");

  auto &kernelRequests = platform->kernelRequests;
  const std::string prefix = "cvode_t::";
  axpbyKernel = kernelRequests.load(prefix + "axpby");
  axmyzKernel = kernelRequests.load(prefix + "axmyz");
  linearCombinationKernel = kernelRequests.load(prefix + "linearCombination");
  innerProdMultiKernel = kernelRequests.load(prefix + "innerProdMulti");

  N = N_VGetLocalLength(SPGMR_CONTENT(S)->xcor);

  o_yg = platform->device.malloc<sunrealtype>(l_max + 1);
  h_yg = platform->device.mallocHost(o_yg.byte_size());

  o_stemp = platform->device.malloc<sunrealtype>(l_max + 1);
  h_stemp = platform->device.mallocHost(o_stemp.byte_size());

  N_VDestroyVectorArray(SPGMR_CONTENT(S)->V, l_max + 1);
}

int cbGMRESSolve(SUNLinearSolver S, N_Vector x, N_Vector b, sunrealtype delta)
{
  sunrealtype beta, r_norm, rho;
  sunbooleantype converged;
  int i, j, k, l, l_plus_1, krydim, ier;

  l_plus_1 = 0;
  krydim = 0;

  /* Make local shorcuts to solver variables. */
  if (S == NULL) {
    return (SUN_ERR_ARG_CORRUPT);
  }

  static int firstTime = 1;
  if (firstTime) {
    cbGMRESSetup(S);
    firstTime = 0;
  }

  auto l_max = SPGMR_CONTENT(S)->maxl;
  auto Hes = SPGMR_CONTENT(S)->Hes;
  auto givens = SPGMR_CONTENT(S)->givens;

  auto A_data = SPGMR_CONTENT(S)->ATData;
  auto P_data = SPGMR_CONTENT(S)->PData;
  auto atimes = SPGMR_CONTENT(S)->ATimes;
  auto zeroguess = &(SPGMR_CONTENT(S)->zeroguess);
  auto nli = &(SPGMR_CONTENT(S)->numiters);
  auto res_norm = &(SPGMR_CONTENT(S)->resnorm);

  auto o_b = getN_VectorMemory(sunrealtype, b);
  auto o_x = getN_VectorMemory(sunrealtype, x);

  auto xcor = SPGMR_CONTENT(S)->xcor;
  auto o_xcor = getN_VectorMemory(sunrealtype, xcor);
  auto vtemp = SPGMR_CONTENT(S)->vtemp;
  auto o_vtemp = getN_VectorMemory(sunrealtype, vtemp);

  auto s1 = SPGMR_CONTENT(S)->s1;
  auto s2 = SPGMR_CONTENT(S)->s2;

  auto o_s1 = getN_VectorMemory(sunrealtype, s1);
  auto o_s2 = getN_VectorMemory(sunrealtype, s2);

  auto o_s2Inv = platform->deviceMemoryPool.reserve<sunrealtype>(N);
  auto o_V = platform->deviceMemoryPool.reserve<pfloat>((l_max + 1) * static_cast<size_t>(N));

  /* Initialize counters and convergence flag */
  *nli = 0;
  converged = SUNFALSE;

  /* Check if Atimes function has been set */
  if (atimes == NULL) {
    *zeroguess = SUNFALSE;
    LASTFLAG(S) = SUNLS_ATIMES_NULL;
    cbGMRESFinish(LASTFLAG(S));
  }

  /* cache s2Inv */
  platform->linAlg->adyz(N, ONE, o_s2, o_s2Inv);

  /* Set V[0] to initial (unscaled) residual r_0 = b - A*x_0 */
  if (*zeroguess) {
    o_vtemp.copyFrom(o_b, o_b.length());
  } else {
    ier = atimes(A_data, x, vtemp);
    if (ier != 0) {
      *zeroguess = SUNFALSE;
      LASTFLAG(S) = (ier < 0) ? SUNLS_ATIMES_FAIL_UNREC : SUNLS_ATIMES_FAIL_REC;
      cbGMRESFinish(LASTFLAG(S));
    }

    platform->linAlg->axpbyz(N, ONE, o_b, -ONE, o_vtemp, o_vtemp);
  }

  /* Set r_norm = beta to L2 norm of V[0] = s1 r_0, and return if small */
  platform->linAlg->axmy(N, ONE, o_s1, o_vtemp);

  sunrealtype rdotr = platform->linAlg->innerProd(N, o_vtemp, o_vtemp, platform->comm.mpiComm());
  *res_norm = r_norm = beta = SUNRsqrt(rdotr);

  if (r_norm <= delta) {
    *zeroguess = SUNFALSE;
    LASTFLAG(S) = SUN_SUCCESS;
    cbGMRESFinish(LASTFLAG(S));
  }

  /* Initialize rho to avoid compiler warning message */
  rho = beta;

  /* Initialize the Hessenberg matrix Hes and Givens rotation product */
  for (i = 0; i <= l_max; i++) {
    for (j = 0; j < l_max; j++) {
      Hes[i][j] = ZERO;
    }
  }
  sunrealtype rotation_product = ONE;

  /* Set V[0] and normalize */
  axpby(ONE / r_norm, o_vtemp, ZERO, o_V);

  /* Inner loop: generate Krylov sequence and Arnoldi basis */
  for (l = 0; l < l_max; l++) {
    (*nli)++;
    krydim = l_plus_1 = l + 1;

    /* Apply right scaling: vtemp = s2_inv V[l] */
    axmyz(ONE, o_V, o_s2Inv, o_vtemp, l);

    /* Apply A: V[l+1] = A s2_inv V[l] */
    ier = atimes(A_data, vtemp, xcor);
    if (ier != 0) {
      *zeroguess = SUNFALSE;
      LASTFLAG(S) = (ier < 0) ? SUNLS_ATIMES_FAIL_UNREC : SUNLS_ATIMES_FAIL_REC;
      cbGMRESFinish(LASTFLAG(S));
    }

    /* Apply left scaling: V[l+1] = s1 A s2_inv V[l] */
    platform->linAlg->axmy(N, ONE, o_s1, o_xcor);

    /* Orthogonalize vtemp2 (V[l+1]) against previous V[i] */
    if (CGSI(Hes, l_plus_1, l_max, &(Hes[l_plus_1][l]), o_V, o_xcor) != 0) {
      *zeroguess = SUNFALSE;
      LASTFLAG(S) = SUNLS_CONV_FAIL;
      cbGMRESFinish(LASTFLAG(S));
    }

    /* Update the QR factorization of Hes */
    if (SUNQRfact(krydim, Hes, givens, l) != 0) {
      *zeroguess = SUNFALSE;
      LASTFLAG(S) = SUNLS_QRFACT_FAIL;
      cbGMRESFinish(LASTFLAG(S));
    }

    /* Update residual norm estimate; break if convergence test passes */
    rotation_product *= givens[2 * l + 1];
    *res_norm = rho = SUNRabs(rotation_product * r_norm);
    if (rho <= delta) {
      converged = SUNTRUE;
      break;
    }

    /* Set V[l+1] and normalize with norm value from the Gram-Schmidt routine */
    axpby(ONE / Hes[l_plus_1][l], o_xcor, ZERO, o_V, ZERO, l_plus_1);
  }
  /* Inner loop is done.  Compute the new correction vector xcor */

  /* Construct g, then solve for y */
  auto yg = (sunrealtype *)h_yg.ptr();
  yg[0] = r_norm;
  for (i = 1; i <= krydim; i++) {
    yg[i] = ZERO;
  }
  if (SUNQRsol(krydim, Hes, givens, yg) != 0) {
    *zeroguess = SUNFALSE;
    LASTFLAG(S) = SUNLS_QRSOL_FAIL;
    cbGMRESFinish(LASTFLAG(S));
  }
  yg[krydim] = ZERO;

  /* Set xcor to correction vector V_l y */
  o_yg.copyFrom(yg);
  linearCombination(krydim + 1, o_yg, o_V, o_xcor, o_xcor);

  /* Add unscaled xcor to initial x to get final solution x, and return */
  if (converged || rho < beta) {
    if (*zeroguess) {
      platform->linAlg->axmyz(N, ONE, o_s2Inv, o_xcor, o_x);
    } else {
      platform->linAlg->axmy(N, ONE, o_s2Inv, o_xcor);
      platform->linAlg->axpby(N, ONE, o_xcor, ONE, o_x);
    }

    *zeroguess = SUNFALSE;
    LASTFLAG(S) = (converged) ? SUN_SUCCESS : SUNLS_RES_REDUCED;
    cbGMRESFinish(LASTFLAG(S));
  }

  *zeroguess = SUNFALSE;
  LASTFLAG(S) = SUNLS_CONV_FAIL;
  cbGMRESFinish(LASTFLAG(S));
}

#endif
