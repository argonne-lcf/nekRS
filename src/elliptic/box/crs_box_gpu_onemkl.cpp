#include <sycl.hpp>
#include "oneapi/mkl/blas.hpp"

static sycl::device D;
static sycl::context C;
static sycl::queue Q;

void box_onemkl_setup(int device_id) {
  sycl::platform P{sycl::gpu_selector_v};
  D = P.get_devices(sycl::info::device_type::gpu)[device_id];
  C = sycl::context{D};
  Q = sycl::queue{C, D};
}

template <typename T>
T *box_onemkl_device_malloc(size_t size) {
  return sycl::malloc_device<T>(size, Q);
}

template <typename T>
void box_onemkl_device_copyto(T *device, const T *host, size_t size) {
  Q.copy(host, device, size).wait();
}

template <typename T>
void box_onemkl_device_copyfrom(const T *device, T *host, size_t size) {
  Q.copy(device, host, size).wait();
}

void box_onemkl_free(void *ptr) {
  sycl::free(ptr, Q);
}

template <typename T>
void box_onemkl_device_gemv(T *y, size_t n, const T * A, const T *x) {
  oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;

  T alpha = 1.0;
  T beta = 0.0;
  try {
    oneapi::mkl::blas::gemv(Q, transA, n, n, alpha, A, n, x, 1, beta, y, 1);
  }
  catch(sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMV:\n"
              << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;
  }
}

template float *box_onemkl_device_malloc<float>(size_t size);
template double *box_onemkl_device_malloc<double>(size_t size);

template void box_onemkl_device_copyto(float *device, const float *host, size_t size);
template void box_onemkl_device_copyto(double *device, const double *host, size_t size);

template void box_onemkl_device_copyfrom(const double *device, double *host, size_t size);
template void box_onemkl_device_copyfrom(const float *device, float *host, size_t size);

template void box_onemkl_device_gemv(float *y, size_t n, const float * A, const float *x);
template void box_onemkl_device_gemv(double *y, size_t n, const double * A, const double *x);
