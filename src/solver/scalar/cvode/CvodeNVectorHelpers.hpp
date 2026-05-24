#ifndef Cvode_NVector_Helpers_HPP_
#define Cvode_NVector_Helpers_HPP_ 

#ifdef ENABLE_CVODE

#include "nvector/nvector_serial.h"
#include "nvector/nvector_mpiplusx.h"

#ifdef ENABLE_CUDA
#include "nvector/nvector_cuda.h"
#endif

#ifdef ENABLE_HIP
#include "nvector/nvector_hip.h"
#endif

static void check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && returnvalue == NULL) {
    nekrsAbort(MPI_COMM_SELF,
               EXIT_FAILURE,
               "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
               funcname);
  }
  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *)returnvalue;
    if (*retval < 0) {
      nekrsAbort(MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                 funcname,
                 *retval);
    }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    nekrsAbort(MPI_COMM_SELF,
               EXIT_FAILURE,
               "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
               funcname);
  }
}

static N_Vector __N_VCloneEmpty(N_Vector w)
{
  N_Vector y;
  if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
    y = N_VCloneEmpty_Cuda(w);
#endif
  } else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
    y = N_VCloneEmpty_Hip(w);
#endif
  } else if (platform->device.mode() == "Serial") {
    y = N_VCloneEmpty_Serial(w);
  }
  check_retval((void *)y, "__N_VCloneEmpty", 0);
  return y;
}

static void __N_VDestroy(N_Vector w) 
{
  //N_VDestroy(N_VGetLocalVector_MPIPlusX(w));
  N_VDestroy(w);
}

static N_Vector __N_VNew(size_t N, SUNContext sunctx)
{
  N_Vector y;
  if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
    y = N_VNew_Cuda(N, sunctx);
#endif
  } else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
    y = N_VNew_Hip(N, sunctx);
#endif
  } else if (platform->device.mode() == "Serial") {
    y = N_VNew_Serial(N, sunctx);
  }
  check_retval((void *)y, "__N_VNew", 0);
  return y;
}

static sunrealtype *__N_VGetDeviceArrayPointer(N_Vector u)
{
  bool useDevice = false;
  useDevice |= platform->device.mode() == "CUDA";
  useDevice |= platform->device.mode() == "HIP";
  useDevice |= platform->device.mode() == "OPENCL";

  if (useDevice) {
    return N_VGetDeviceArrayPointer(u);
  } else {
    return N_VGetArrayPointer_Serial(u);
  }
}

#define getN_VectorMemory(T,S) (  platform->device.occaDevice().wrapMemory<T>( __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(S)), N_VGetLocalLength(S)) )

#endif

#endif
