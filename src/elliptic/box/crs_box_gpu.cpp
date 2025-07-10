#include <cstdio>
#include <cstdlib>

#include <platform.hpp>

#include "crs_box_impl.hpp"

static const double one = 1.0, zero = 0.0;
static const float one_f32 = 1.0f, zero_f32 = 0.0f;

static int initialized = 0;
static gs_dom dom;
static int nr = 0;
static void *d_A_inv = NULL;

template <typename T>
static void setup_inverse(T *A_inv, const struct csr *A) {
  assert(sizeof(dfloat) == sizeof(double));

  const int N = A->nr;
  std::vector<dfloat> B(N * N);
  for (uint i = 0; i < A->nr; i++) {
    for (uint j = A->offs[i]; j < A->offs[i + 1]; j++)
      B[(A->cols[j] - A->base) * A->nr + i] = A->vals[j];
  }

  auto invA = platform->linAlg->matrixInverse(N, B);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      A_inv[i * N + j] = (T)invA[j * N + i];
  }
}

#if defined(ENABLE_HIPBLAS)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define check_hip_runtime(call)                                                \
  {                                                                            \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP runtime error: %s\n", hipGetErrorString(err));      \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  }

static hipblasHandle_t handle = NULL;

void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box) {
  assert(null_space == 0);

  double *A_inv = 0;
  float *A_inv_f32 = 0;
  setup_inverse(&A_inv, &A_inf_f32, A);

  check_hip_runtime(hipMalloc(&d_A_inv, A->nr * A->nr * sizeof(double)));
  check_hip_runtime(hipMemcpy(d_A_inv, A_inv, A->nr * A->nr * sizeof(double),
                              hipMemcpyHostToDevice));

  check_hip_runtime(hipMalloc(&d_A_inv_f32, A->nr * A->nr * sizeof(float)));
  check_hip_runtime(hipMemcpy(d_A_inv_f32, A_inv_f32,
                              A->nr * A->nr * sizeof(float),
                              hipMemcpyHostToDevice));
  free(A_inv), free(A_inv_f32);

  h_r = calloc(A->nr, sizeof(double));
  h_x = calloc(A->nr, sizeof(double));
  check_hip_runtime(hipMalloc(&d_r, A->nr * sizeof(double)));
  check_hip_runtime(hipMalloc(&d_x, A->nr * sizeof(double)));

  o_cx = platform->device.malloc(A->nr * sizeof(float));
  hipblasCreate(&handle);

  initialized = 1;
}

void asm1_gpu_solve_float(float *x, struct box *box, const float *r) {
  float *h_r_ = (float *)h_r;
  for (uint i = 0; i < nr; i++)
    h_r_[i] = 0;
  for (uint i = 0; i < box->sn; i++) {
    if (box->u2c[i] >= 0)
      h_r_[box->u2c[i]] += r[i];
  }

  check_hip_runtime(
      hipMemcpy(d_r, h_r_, nr * sizeof(float), hipMemcpyHostToDevice));

  hipblasSgemv(handle, HIPBLAS_OP_T, nr, nr, &one_f32, d_A_inv_f32, nr,
               (float *)d_r, 1, &zero_f32, (float *)d_x, 1);

  check_hip_runtime(
      hipMemcpy(h_x, d_x, nr * sizeof(float), hipMemcpyDeviceToHost));

  float *h_x_ = (float *)h_x;
  for (uint i = 0; i < box->sn; i++) {
    if (box->u2c[i] >= 0)
      x[i] = h_x_[box->u2c[i]];
    else
      x[i] = 0;
  }
}

void asm1_gpu_solve_double(double *x, struct box *box, const double *r) {
  double *h_r_ = (double *)h_r;
  for (uint i = 0; i < nr; i++)
    h_r_[i] = 0;
  for (uint i = 0; i < box->sn; i++) {
    if (box->u2c[i] >= 0)
      h_r_[box->u2c[i]] += r[i];
  }

  check_hip_runtime(
      hipMemcpy(d_r, h_r_, nr * sizeof(double), hipMemcpyHostToDevice));

  hipblasDgemv(handle, HIPBLAS_OP_T, nr, nr, &one, d_A_inv, nr, (double *)d_r,
               1, &zero, (double *)d_x, 1);

  check_hip_runtime(
      hipMemcpy(h_x, d_x, nr * sizeof(double), hipMemcpyDeviceToHost));

  double *h_x_ = (double *)h_x;
  for (uint i = 0; i < box->sn; i++) {
    if (box->u2c[i] >= 0)
      x[i] = h_x_[box->u2c[i]];
    else
      x[i] = 0;
  }
}

void asm1_gpu_solve(void *x, struct box *box, const void *r) {
  if (!initialized) {
    fprintf(stderr, "GPU BLAS not initialized.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (box->dom == gs_float) {
    asm1_gpu_solve_float((float *)x, box, (const float *)r);
    return;
  }

  if (box->dom == gs_double) {
    asm1_gpu_solve_double((double *)x, box, (const double *)r);
    return;
  }
}

void asm1_gpu_free(struct box *box) {
  check_hip_runtime(hipFree(d_A_inv));
  check_hip_runtime(hipFree(d_A_inv_f32));
  check_hip_runtime(hipFree(d_r));
  check_hip_runtime(hipFree(d_x));

  hipblasDestroy(handle);

  free(h_r), h_r = NULL;
  free(h_x), h_x = NULL;
  nr = 0;
  initialized = 0;
}

#elif defined(ENABLE_ONEMKL)
#include "crs_box_gpu_onemkl.hpp"

template <typename T>
void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box) {
  if (initialized) return;

  assert(null_space == 0);

  const size_t size = A->nr * A->nr;
  T *A_inv = tcalloc(T, size);
  setup_inverse(A_inv, A);

  d_A_inv = static_cast<void *>(box_onemkl_device_malloc<T>(size));
  box_onemkl_device_copyto<T>(static_cast<T *>(d_A_inv), A_inv, size);
  free(A_inv);

  initialized = 1;
  dom = box->opts.dom;
  nr = A->nr;
}

void asm1_gpu_solve(occa::memory &o_x, struct box *box, occa::memory &o_r) {
  if (!initialized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

  if (box->opts.dom == gs_double)
    box_onemkl_device_gemv<double>((double *)o_x.ptr(), nr, (double *)d_A_inv, (double *)o_r.ptr());
  else
    box_onemkl_device_gemv<float>((float *)o_x.ptr(), nr, (float *)d_A_inv, (float *)o_r.ptr());
}

void asm1_gpu_free(struct box *box) {
  box_onemkl_free(static_cast<void *>(d_A_inv));
}

template void asm1_gpu_setup<float>(struct csr *A, unsigned null_space, struct box *box);
template void asm1_gpu_setup<double>(struct csr *A, unsigned null_space, struct box *box);

#else

void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box) {
  fprintf(stderr, "GPU BLAS not enabled.\n");
  exit(EXIT_FAILURE);
}

void asm1_gpu_solve(void *x, struct box *box, const void *r) {
  fprintf(stderr, "GPU BLAS not enabled.\n");
  exit(EXIT_FAILURE);
}

void asm1_gpu_free(struct box *box) {
  fprintf(stderr, "GPU BLAS not enabled.\n");
  exit(EXIT_FAILURE);
}
#endif
