#include <cstdio>
#include <cstdlib>

#include <platform.hpp>

#include "crs_box_impl.hpp"

static const double one = 1.0, zero = 0.0;
static const float one_f32 = 1.0f, zero_f32 = 0.0f;

static int initialized = 0;
static int nr = 0;
static void *h_r = NULL, *h_x = NULL;

static double *d_A_inv = NULL;
static float *d_A_inv_f32 = NULL;
static void *d_r = NULL, *d_x = NULL;

static uint gs_n;
static occa::memory o_gs_off, o_gs_idx;
static occa::memory o_cx;

#define FNAME(x) TOKEN_PASTE(x, _)
#define FDGETRF FNAME(dgetrf)
#define FDGETRI FNAME(dgetri)

extern "C" {
  void FDGETRF(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);
  void FDGETRI(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
               int *INFO);
}

static void setup_core(uint nr_) {
  nr = nr_;
  h_r = calloc(nr, sizeof(double));
  h_x = calloc(nr, sizeof(double));
}

static void finalize_core(void) {
  free(h_r), h_r = NULL;
  free(h_x), h_x = NULL;
  nr = 0;
  initialized = 0;
}

static void setup_inverse(double **A_inv, float **A_inv_f32, const struct csr *A) {
  double *B = tcalloc(double, A->nr * A->nr);
  for (uint i = 0; i < A->nr; i++) {
    for (uint j = A->offs[i]; j < A->offs[i + 1]; j++)
      B[i * A->nr + A->cols[j] - A->base] = A->vals[j];
  }

  int N = A->nr;
  *A_inv = tcalloc(double, A->nr * A->nr);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      (*A_inv)[i * N + j] = B[j * N + i];
  }

  int *ipiv = tcalloc(int, A->nr);
  int info;
  FDGETRF(&N, &N, *A_inv, &N, ipiv, &info);
  if (info != 0) {
    fprintf(stderr, "dgetrf failed !\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int size = N * N;
  double *work = (double *)calloc(size, sizeof(double));
  FDGETRI(&N, *A_inv, &N, ipiv, work, &size, &info);
  if (info != 0) {
    fprintf(stderr, "dgetri failed !\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  free(ipiv), free(work);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      B[i * N + j] = (*A_inv)[j * N + i];
  }

  for (uint i = 0; i < A->nr; i++)
    for (uint j = 0; j < A->nr; j++)
      (*A_inv)[i * N + j] = B[i * N + j];

  *A_inv_f32 = tcalloc(float, A->nr * A->nr);
  for (uint i = 0; i < A->nr; i++)
    for (uint j = 0; j < A->nr; j++)
      (*A_inv_f32)[i * N + j] = (float)B[i * N + j];

  free(B);
}

static void setup_u2c(const int un, const int *u2c) {
  struct map_t {
    uint u, c;
  };

  struct array map;
  array_init(struct map_t, &map, un);

  struct map_t m;
  for (uint i = 0; i < un; i++) {
    if (u2c[i] < 0)
      continue;
    m.u = i, m.c = u2c[i];
    array_cat(struct map_t, &map, &m, 1);
  }

  if (map.n == 0) {
    fprintf(stderr, "RHS is empty.\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  buffer bfr;
  buffer_init(&bfr, 1024);
  sarray_sort_2(struct map_t, map.ptr, map.n, c, 0, u, 0, &bfr);
  buffer_free(&bfr);

  gs_n = 1;
  struct map_t *pm = (struct map_t *)map.ptr;
  uint c = pm[0].c;
  for (uint i = 1; i < map.n; i++) {
    if (pm[i].c != c) {
      gs_n++;
      c = pm[i].c;
    }
  }

  dlong *gs_off = tcalloc(dlong, gs_n + 1);
  dlong *gs_idx = tcalloc(dlong, map.n);
  gs_off[0] = 0;
  gs_idx[0] = pm[0].u;
  uint gs_n_ = 0;
  for (uint i = 1; i < map.n; i++) {
    if (pm[i].c != pm[i - 1].c) {
      gs_n_++;
      gs_off[gs_n_] = i;
    }
    gs_idx[i] = pm[i].u;
  }
  gs_n_++;

  if (gs_n_ != gs_n) {
    fprintf(stderr, "gs_n_ != gs_n\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  o_gs_off = platform->device.malloc((gs_n + 1) * sizeof(dlong));
  o_gs_off.copyFrom(gs_off, sizeof(dlong) * (gs_n + 1), 0);
  o_gs_idx = platform->device.malloc(map.n * sizeof(dlong));
  o_gs_idx.copyFrom(gs_idx, sizeof(dlong) * map.n, 0);
  o_cx = platform->device.malloc(nr * sizeof(float));

  free(gs_off), free(gs_idx), array_free(&map);
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
  setup_u2c(box->sn, box->u2c);

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

void asm1_gpu_setup(struct csr *A, unsigned null_space, struct box *box) {
  assert(null_space == 0);

  if (initialized) return;

  double *A_inv = 0;
  float *A_inv_f32 = 0;
  setup_inverse(&A_inv, &A_inv_f32, A);
  setup_core(A->nr);
  setup_u2c(box->sn, box->u2c);

  const size_t size = nr * nr;
  d_A_inv = box_onemkl_device_malloc<double>(size);
  box_onemkl_device_copyto<double>(d_A_inv, A_inv, size);

  d_A_inv_f32 = box_onemkl_device_malloc<float>(size);
  box_onemkl_device_copyto<float>(d_A_inv_f32, A_inv_f32, size);

  free(A_inv), free(A_inv_f32);

  d_r = box_onemkl_device_malloc<double>(nr);
  d_x = box_onemkl_device_malloc<double>(nr);

  initialized = 1;
}

template <typename T>
static void asm1_gpu_solve_aux(T *x, struct box *box, const T *r) {
  T *h_r_T = static_cast<T *>(h_r);
  for (uint i = 0; i < nr; i++)
    h_r_T[i] = 0;
  for (uint i = 0; i < box->sn; i++) {
    if (box->u2c[i] >= 0)
      h_r_T[box->u2c[i]] += r[i];
  }

  box_onemkl_device_copyto<T>(static_cast<T *>(d_r), h_r_T, nr);

  T *h_x_T = static_cast<T *>(h_x);
  box_onemkl_device_copyfrom<T>(static_cast<T *>(d_x), h_x_T, nr);

  for (uint i = 0; i < box->sn; i++) {
    if (box->u2c[i] >= 0)
      x[i] = h_x_T[box->u2c[i]];
    else
      x[i] = 0;
  }
}

void asm1_gpu_solve(void *x, struct box *box, const void *r) {
  if (!initialized) {
    fprintf(stderr, "oneMKL is not initialized.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  switch(box->dom) {
    case gs_double:
      asm1_gpu_solve_aux<double>((double *)x, box, (double *)r);
      break;
    case gs_float:
      asm1_gpu_solve_aux<float>((float *)x, box, (float *)r);
      break;
    default:
      break;
  }
}

void asm1_gpu_free(struct box *box) {
  box_onemkl_free(static_cast<void *>(d_A_inv));
  box_onemkl_free(static_cast<void *>(d_A_inv_f32));
  box_onemkl_free(static_cast<void *>(d_r));
  box_onemkl_free(static_cast<void *>(d_r));
  finalize_core();
}
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
