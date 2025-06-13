#if !defined(CRS_BOX_GPU_ONEMKL)
#define CRS_BOX_GPU_ONEMKL

void box_onemkl_setup(int device_id);

template <typename T>
T *box_onemkl_device_malloc(size_t size);

template <typename T>
void box_onemkl_device_copyto(T *device, const T *host, size_t size);

template <typename T>
void box_onemkl_device_copyfrom(const T *device, T *host, size_t size);

template <typename T>
void box_onemkl_device_gemv(T *y, size_t n, const T * A, const T *x);

void box_onemkl_free(void *ptr);

#endif // CRS_BOX_GPU_ONEMKL
