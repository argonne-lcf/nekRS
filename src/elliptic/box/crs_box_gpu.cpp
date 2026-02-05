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

#if defined(ENABLE_BOX_HIPBLAS)
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
void *d_r, *d_x;

template <typename T>
void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box) {
  assert(null_space == 0);

  const size_t size = A->nr * A->nr;
  T *A_inv = tcalloc(T, size);
  setup_inverse(A_inv, A);

  check_hip_runtime(hipMemcpy(d_A_inv, A_inv, A->nr * A->nr * sizeof(T),
                              hipMemcpyHostToDevice));
  free(A_inv);

  check_hip_runtime(hipMalloc(&d_r, A->nr * sizeof(T)));
  check_hip_runtime(hipMalloc(&d_x, A->nr * sizeof(T)));

  hipblasCreate(&handle);

  dom = box->opts.dom;
  nr = A->nr;
  initialized = 1;
}

template <typename T>
void box_hipblas(T *x, struct box *box, const T *r) {
  if (sizeof(T) == sizeof(float)) {
    hipblasSgemv(handle, HIPBLAS_OP_T, nr, nr, &one_f32, (float *)d_A_inv, nr,
        (float *)d_r, 1, &zero_f32, (float *)d_x, 1);
  } else if (sizeof(T) == sizeof(double)) {
    hipblasDgemv(handle, HIPBLAS_OP_T, nr, nr, &one, (double *)d_A_inv, nr,
        (double *)d_r, 1, &zero, (double *)d_x, 1);
  }
}

void asm1_gpu_solve(occa::memory &o_x, struct box *box, occa::memory &o_r) {
  if (!initialized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

  if (box->opts.dom == gs_double)
    box_hipblas<double>((double *)o_x.ptr(), box, (double *)o_r.ptr());
  else
    box_hipblas<float>((float *)o_x.ptr(), box, (float *)o_r.ptr());
}

void asm1_gpu_free(struct box *box) {
  hipblasDestroy(handle);
  check_hip_runtime(hipFree(d_A_inv));
  check_hip_runtime(hipFree(d_r));
  check_hip_runtime(hipFree(d_x));
  nr = 0, initialized = 0;
}

#elif defined(ENABLE_BOX_ONEMKL)
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

  dom = box->opts.dom;
  nr = A->nr;
  initialized = 1;
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

#else

template <typename T>
void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box) {
  fprintf(stderr, "GPU BLAS not enabled.\n");
  exit(EXIT_FAILURE);
}

void asm1_gpu_solve(occa::memory &o_x, struct box *box, occa::memory &o_r) {
  fprintf(stderr, "GPU BLAS not enabled.\n");
  exit(EXIT_FAILURE);
}

void asm1_gpu_free(struct box *box) {
  fprintf(stderr, "GPU BLAS not enabled.\n");
  exit(EXIT_FAILURE);
}
#endif

template void asm1_gpu_setup<float>(struct csr *A, unsigned null_space, struct box *box);
template void asm1_gpu_setup<double>(struct csr *A, unsigned null_space, struct box *box);
