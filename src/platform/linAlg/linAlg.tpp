template <typename T = dfloat> void mask(const dlong N, const occa::memory &o_maskIds, occa::memory &o_a)
{
  if (N) {
    linAlgLaunchKernel(getKnlPrefix<T>() + "mask", N, o_maskIds, o_a);
  }
}

// o_a[n] = alpha
template <typename T = dfloat> void fill(const dlong N, const double alpha, occa::memory &o_a)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "fill", N, static_cast<T>(alpha), o_a);
}

// o_a[n] = abs(o_a[n])
template <typename T = dfloat> void abs(const dlong N, occa::memory &o_a)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "vabs", N, o_a);
}

// o_a[n] += alpha
template <typename T = dfloat> void add(const dlong N, const double alpha, occa::memory &o_a, const dlong offset = 0)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "add", N, offset, static_cast<T>(alpha), o_a);
}

// o_a[n] *= alpha
template <typename T = dfloat> void scale(const dlong N, const double alpha, occa::memory &o_a)
{
  scaleMany(N, 1, 0, alpha, o_a, 0);
}

template <typename T = dfloat>
void scaleMany(const dlong N,
               const int Nfields,
               const dlong fieldOffset,
               const double alpha,
               occa::memory &o_a,
               const dlong offset = 0)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "scaleMany", N, Nfields, fieldOffset, offset, static_cast<T>(alpha), o_a);
}

// o_y[n] = beta*o_y[n] + alpha*o_x[n]
template <typename T = dfloat>
void axpby(const dlong N,
           const double alpha,
           const occa::memory &o_x,
           const double beta,
           occa::memory &o_y,
           const dlong xOffset = 0,
           const dlong yOffset = 0)
{
  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  linAlgLaunchKernel(knlPrefix + "axpby", N, xOffset, yOffset, static_cast<T>(alpha), o_x, static_cast<T>(beta), o_y);
  platform->flopCounter->add("axpby", FPfactor + 3 * static_cast<double>(N));
}

template <typename T = dfloat>
void axpbyMany(const dlong N,
               const int Nfields,
               const dlong offset,
               const double alpha,
               const occa::memory &o_x,
               const double beta,
               occa::memory &o_y)
{
  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  linAlgLaunchKernel(knlPrefix + "axpbyMany", N, Nfields, offset, static_cast<T>(alpha), o_x, static_cast<T>(beta), o_y);
  platform->flopCounter->add("axpbyMany", FPfactor * 3 * static_cast<double>(N) * Nfields);
}

// o_z[n] = beta*o_y[n] + alpha*o_x[n]
template <typename T = dfloat>
void axpbyz(const dlong N,
            const double alpha,
            const occa::memory &o_x,
            const double beta,
            const occa::memory &o_y,
            occa::memory &o_z)
{
  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  linAlgLaunchKernel(knlPrefix + "axpbyz", N, static_cast<T>(alpha), o_x, static_cast<T>(beta), o_y, o_z);
  platform->flopCounter->add("axpbyz", FPfactor * 3 * static_cast<double>(N));
}

template <typename T = dfloat>
void axpbyzMany(const dlong N,
                const int Nfields,
                const dlong fieldOffset,
                const double alpha,
                const occa::memory &o_x,
                const double beta,
                const occa::memory &o_y,
                occa::memory &o_z)
{
  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  linAlgLaunchKernel(knlPrefix + "axpbyzMany", N, Nfields, fieldOffset, static_cast<T>(alpha), o_x, static_cast<T>(beta), o_y, o_z);
  platform->flopCounter->add("axpbyzMany", FPfactor * 3 * static_cast<double>(N) * Nfields);
}

// o_y[n] = alpha*o_x[n]*o_y[n]
template <typename T = dfloat>
void axmy(const dlong N, const double alpha, const occa::memory &o_x, occa::memory &o_y)
{
  const auto knlPrefix = getKnlPrefix<T>();
  linAlgLaunchKernel(knlPrefix + "axmy", N, static_cast<T>(alpha), o_x, o_y);
}

// mode 1:
// o_y[n,fld] = alpha*o_x[n,fld]*o_y[n,fld]
// mode 0:
// o_y[n,fld] = alpha*o_x[n]*o_y[n,fld]
template <typename T = dfloat>
void axmyMany(const dlong N,
              const int Nfields,
              const dlong offset,
              const int mode,
              const double alpha,
              const occa::memory &o_x,
              occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "axmyMany", N, Nfields, offset, mode, static_cast<T>(alpha), o_x, o_y);
}

template <typename T = dfloat>
void axmyVector(const dlong N,
                const dlong offset,
                const int mode,
                const double alpha,
                const occa::memory &o_x,
                occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "axmyVector", N, offset, mode, static_cast<T>(alpha), o_x, o_y);
}

// o_z[n] = alpha*o_x[n]*o_y[n]
template <typename T = dfloat>
void axmyz(const dlong N,
           const dfloat alpha,
           const occa::memory &o_x,
           const occa::memory &o_y,
           occa::memory &o_z)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "axmyz", N, static_cast<T>(alpha), o_x, o_y, o_z);
}

template <typename T = dfloat>
void axmyzMany(const dlong N,
               const int Nfields,
               const dlong offset,
               const double alpha,
               const occa::memory &o_x,
               const occa::memory &o_y,
               occa::memory &o_z)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "axmyzMany", N, Nfields, offset, static_cast<T>(alpha), o_x, o_y, o_z);
}

// o_y[n] = alpha*o_x[n]/o_y[n]
template <typename T = dfloat>
void axdy(const dlong N, const double alpha, const occa::memory &o_x, occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "axdy", N, static_cast<T>(alpha), o_x, o_y);
}

template <typename T = dfloat>
void aydx(const dlong N, const double alpha, const occa::memory &o_x, occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "aydx", N, static_cast<T>(alpha), o_x, o_y);
}

template <typename T = dfloat>
void aydxMany(const dlong N,
              const int Nfields,
              const dlong fieldOffset,
              const int mode,
              const double alpha,
              const occa::memory &o_x,
              occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "aydxMany", N, Nfields, fieldOffset, mode, static_cast<T>(alpha), o_x, o_y);
}

// o_y[n] = alpha/o_y[n]
template <typename T = dfloat> void ady(const dlong N, const double alpha, occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "ady", N, static_cast<T>(alpha), o_y);
}

// o_z[n] = alpha/o_y[n]
template <typename T = dfloat>
void adyz(const dlong N, const double alpha, const occa::memory &o_y, occa::memory &o_z)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "adyz", N, static_cast<T>(alpha), o_y, o_z);
}

template <typename T = dfloat>
void adyMany(const dlong N, const int Nfields, const dlong offset, const double alpha, occa::memory &o_y)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "adyMany", N, Nfields, offset, static_cast<T>(alpha), o_y);
}

// o_z[n] = alpha*o_x[n]/o_y[n]
template <typename T = dfloat>
void axdyz(const dlong N, const double alpha, const occa::memory &o_x, const occa::memory &o_y, occa::memory &o_z)
{
  linAlgLaunchKernel(getKnlPrefix<T>() + "axdyz", N, static_cast<T>(alpha), o_x, o_y, o_z);
}

// \sum o_a
template <typename T = dfloat>
T sum(const dlong N, const occa::memory &o_a, MPI_Comm _comm, const dlong offset = 0)
{ 
  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  
  if (N > 1) {
    linAlgLaunchKernel(getKnlPrefix<T>() + "sum", Nblock, N, offset, o_a, o_scratch);
    o_scratch.copyTo(scratch);
  } else { 
    o_a.copyTo(scratch, N); 
  } 
    
  double sum = 0;
  for (dlong n = 0; n < Nblock; ++n) {
    sum += scratch[n];
  } 
    
  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, _comm);
  } 
      
  return sum;
}

template <typename T = dfloat>
T sumMany(const dlong N, const int Nfields, const dlong fieldOffset, const occa::memory &o_a, MPI_Comm _comm)
{
  int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  if (N > 1 || Nfields > 1) {
    linAlgLaunchKernel(getKnlPrefix<T>() + "sumMany", Nblock, N, Nfields, fieldOffset, o_a, o_scratch);
    o_scratch.copyTo(scratch);
  } else {
    o_a.copyTo(scratch, N);
  }

  double sum = 0;
  for (dlong n = 0; n < Nblock; ++n) {
    sum += scratch[n];
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  return sum;
}

template <typename T = dfloat>
std::vector<std::pair<T, T>> minMax(dlong Nlocal, const std::vector<occa::memory> &o_fldList, MPI_Comm comm)
{
  std::vector<std::pair<T, T>> out;
  for (const auto &o_entry : o_fldList) {
    const auto min = this->min<T>(Nlocal, o_entry, comm);
    const auto max = this->max<T>(Nlocal, o_entry, comm);
    out.push_back({min, max});
  }
  return out;
}

// \min o_a
template <typename T = dfloat>
std::vector<T> min(dlong Nlocal, const std::vector<occa::memory> &o_fldList, MPI_Comm comm)
{
  std::vector<T> out;
  for (const auto &o_entry : o_fldList) {
    out.push_back(this->min<T>(Nlocal, o_entry, comm));
  }
  return out;
}

template <typename T = dfloat> T min(const dlong N, const occa::memory &o_a, MPI_Comm _comm)
{
  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  if (N > 1) {
    linAlgLaunchKernel(getKnlPrefix<T>() + "min", Nblock, N, o_a, o_scratch);
    o_scratch.copyTo(scratch);
  } else {
    o_a.copyTo(scratch, N);
  }

  double val = scratch[0];
  for (dlong n = 1; n < Nblock; ++n) {
    val = (scratch[n] < val) ? static_cast<dfloat>(scratch[n]) : val;
  }

  MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_MIN, _comm);

  return val;
}

// \max o_a
template <typename T = dfloat>
std::vector<T> max(const dlong Nlocal, const std::vector<occa::memory> &o_fldList, MPI_Comm comm)
{
  std::vector<T> out;
  for (const auto &o_entry : o_fldList) {
    out.push_back(this->max<T>(Nlocal, o_entry, comm));
  }
  return out;
}

template <typename T = dfloat> T max(const dlong N, const occa::memory &o_a, MPI_Comm _comm)
{
  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  if (N > 1) {
    linAlgLaunchKernel(getKnlPrefix<T>() + "max", Nblock, N, o_a, o_scratch);
    o_scratch.copyTo(scratch);
  } else {
    o_a.copyTo(scratch, N);
  }

  double val = scratch[0];
  for (dlong n = 1; n < Nblock; ++n) {
    val = (scratch[n] > val) ? static_cast<dfloat>(scratch[n]) : val;
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_MAX, _comm);
  }

  return val;
}

// ||o_a||_\infty
template <typename T = dfloat> T amax(const dlong N, const occa::memory &o_a, MPI_Comm _comm)
{
  return amaxMany<T>(N, 1, 0, o_a, _comm);
}

template <typename T = dfloat>
T amaxMany(const dlong N, const int Nfields, const dlong fieldOffset, const occa::memory &o_x, MPI_Comm _comm)
{
  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  if (N > 1) {
    linAlgLaunchKernel(getKnlPrefix<T>() + "amaxMany", Nblock, N, Nfields, fieldOffset, o_x, o_scratch);
    o_scratch.copyTo(scratch);
  } else {
    o_x.copyTo(scratch, N);
  }

  double val = scratch[0];
  for (dlong n = 1; n < Nblock; ++n) {
    val = std::max(val, static_cast<double>(scratch[n]));
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_MAX, _comm);
  }

  return val;
}

// ||o_a||_2
template <typename T = dfloat> T norm2(const dlong N, const occa::memory &o_x, MPI_Comm _comm)
{
  return norm2Many<T>(N, 1, 0, o_x, _comm);
}

template <typename T = dfloat>
T norm2Many(const dlong N,
            const int Nfields,
            const dlong fieldOffset,
            const occa::memory &o_x,
            MPI_Comm _comm)
{
  if (timer) {
    platform->timer.tic("dotp");
  }

  int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  double norm = 0;
  linAlgLaunchKernel(getKnlPrefix<T>() + "norm2Many", Nblock, N, Nfields, fieldOffset, o_x, o_scratch);
  if (serial) {
    norm = *((T *)o_scratch.ptr());
  } else {
    o_scratch.copyTo(scratch);
    for (dlong n = 0; n < Nblock; ++n) {
      norm += scratch[n];
    }
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  if (timer) {
    platform->timer.toc("dotp");
  }

  return std::sqrt(norm);
}

// ||o_a||_1
template <typename T = dfloat> T norm1(const dlong N, const occa::memory &o_x, MPI_Comm _comm)
{
  return norm1Many<T>(N, 1, 0, o_x, _comm);
}

template <typename T = dfloat>
T norm1Many(const dlong N,
            const int Nfields,
            const dlong fieldOffset,
            const occa::memory &o_x,
            MPI_Comm _comm)
{
  if (timer) {
    platform->timer.tic("dotp");
  }

  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  linAlgLaunchKernel(getKnlPrefix<T>() + "norm1Many", Nblock, N, Nfields, fieldOffset, o_x, o_scratch);

  double norm = 0;
  if (serial) {
    norm = *((T *)o_scratch.ptr());
  } else {
    o_scratch.copyTo(scratch);
    for (dlong n = 0; n < Nblock; ++n) {
      norm += scratch[n];
    }
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  if (timer) {
    platform->timer.toc("dotp");
  }

  return norm;
}

// o_x.o_y
template <typename T = dfloat>
T innerProd(const dlong N,
            const occa::memory &o_x,
            const occa::memory &o_y,
            MPI_Comm _comm,
            const dlong offset = 0)
{

  if (timer) {
    platform->timer.tic("dotp");
  }

  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  linAlgLaunchKernel(getKnlPrefix<T>() + "innerProd", Nblock, N, offset, o_x, o_y, o_scratch);

  double dot = 0;
  if (serial) {
    dot = *((T *)o_scratch.ptr());
  } else {
    o_scratch.copyTo(scratch);

    for (dlong n = 0; n < Nblock; ++n) {
      dot += scratch[n];
    }
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &dot, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  if (timer) {
    platform->timer.toc("dotp");
  }

  return dot;
}

template <typename T = dfloat>
void innerProdMulti(const dlong N,
                    const int NVec,
                    const int Nfields,
                    const dlong fieldOffset,
                    const occa::memory &o_x,
                    const occa::memory &o_y,
                    MPI_Comm _comm,
                    T *result,
                    const dlong yOffset = 0)
{
  weightedInnerProdMulti<T>(N, NVec, Nfields, fieldOffset, o_NULL, o_x, o_y, _comm, result, yOffset, 0);
}

template <typename T = dfloat>
void weightedInnerProdMulti(const dlong N,
                            const int NVec,
                            const int Nfields,
                            const dlong fieldOffset,
                            const occa::memory &o_w,
                            const occa::memory &o_x,
                            const occa::memory &o_y,
                            MPI_Comm _comm,
                            T *result,
                            const dlong yOffset = 0,
                            const int weight = 1)
{
  if (timer) {
    platform->timer.tic("dotpMulti");
  }

  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(NVec * Nblock);
  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  linAlgLaunchKernel(knlPrefix + "weightedInnerProdMulti",
               Nblock,
               N,
               Nfields,
               fieldOffset,
               NVec,
               yOffset,
               weight,
               o_w,
               o_x,
               o_y,
               o_scratch);

  o_scratch.copyTo(scratch);

  for (int field = 0; field < NVec; ++field) {
    T dot = 0;
    for (dlong n = 0; n < o_scratch.size() / NVec; ++n) {
      dot += scratch[n + field * Nblock];
    }
    result[field] = dot;
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE,
                  result,
                  NVec,
                  (std::is_same<T, double>::value) ? MPI_DOUBLE : MPI_FLOAT,
                  MPI_SUM,
                  _comm);
  }

  if (timer) {
    platform->timer.toc("dotpMulti");
  }

  platform->flopCounter->add("weightedInnerProdMulti",
                             FPfactor * NVec * static_cast<double>(N) * (2 * Nfields + 1));
}

template <typename T = dfloat>
void weightedInnerProdMulti(const dlong N,
                            const int NVec,
                            const int Nfields,
                            const dlong fieldOffset,
                            const occa::memory &o_w,
                            const occa::memory &o_x,
                            const occa::memory &o_y,
                            MPI_Comm _comm,
                            occa::memory &o_result,
                            const dlong yOffset = 0,
                            const int weight = 1)
{
  if (timer) {
    platform->timer.tic("dotpMulti");
  }

  const auto Nblock = (N + blocksize - 1) / blocksize;

  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  nekrsCheck(!platform->device.deviceAtomic,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s",
             "requires support of floating point atomics!\n");

  linAlgLaunchKernel(knlPrefix + "weightedInnerProdMultiDevice",
               Nblock,
               N,
               Nfields,
               fieldOffset,
               NVec,
               yOffset,
               weight,
               o_w,
               o_x,
               o_y,
               o_result);

  if (_comm != MPI_COMM_SELF) {
    platform->device.finish();
    MPI_Allreduce(MPI_IN_PLACE,
                  (void *)o_result.ptr(),
                  NVec,
                  (std::is_same<T, double>::value) ? MPI_DOUBLE : MPI_FLOAT,
                  MPI_SUM,
                  _comm);
  }

  if (timer) {
    platform->timer.toc("dotpMulti");
  }

  platform->flopCounter->add("weightedInnerProdMulti",
                             FPfactor * NVec * static_cast<double>(N) * (2 * Nfields + 1));
}

template <typename T = dfloat>
T weightedInnerProd(const dlong N,
                    const occa::memory &o_w,
                    const occa::memory &o_x,
                    const occa::memory &o_y,
                    MPI_Comm _comm)
{
  return weightedInnerProdMany<T>(N, 1, 0, o_w, o_x, o_y, _comm);
}

template <typename T = dfloat>
T weightedInnerProdMany(const dlong N,
                        const int Nfields,
                        const dlong fieldOffset,
                        const occa::memory &o_w,
                        const occa::memory &o_x,
                        const occa::memory &o_y,
                        MPI_Comm _comm)
{
  if (timer) {
    platform->timer.tic("dotp");
  }

  const auto knlPrefix = getKnlPrefix<T>();
  const auto FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;

  const auto Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  linAlgLaunchKernel(knlPrefix + "weightedInnerProdMany",
               Nblock,
               N,
               Nfields,
               fieldOffset,
               o_w,
               o_x,
               o_y,
               o_scratch);

  double dot = 0;
  if (serial) {
    dot = *((T *)o_scratch.ptr());
  } else {
    o_scratch.copyTo(scratch);
    for (dlong n = 0; n < o_scratch.size(); ++n) {
      dot += scratch[n];
    }
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &dot, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  if (timer) {
    platform->timer.toc("dotp");
  }

  platform->flopCounter->add("weightedInnerProdMany", FPfactor * 3 * static_cast<double>(N) * Nfields);

  return dot;
}

// ||o_a||_w2
template <typename T = dfloat>
T weightedNorm2(const dlong N, const occa::memory &o_w, const occa::memory &o_a, MPI_Comm _comm)
{
  return weightedNorm2Many<T>(N, 1, 0, o_w, o_a, _comm);
}

template <typename T = dfloat>
T weightedNorm2Many(const dlong N,
                    const int Nfields,
                    const dlong fieldOffset,
                    const occa::memory &o_w,
                    const occa::memory &o_a,
                    MPI_Comm _comm)
{
  return std::sqrt(weightedInnerProdMany<T>(N, Nfields, fieldOffset, o_w, o_a, o_a, _comm));
}

// ||o_a||_w1
template <typename T = dfloat>
T weightedNorm1(const dlong N, const occa::memory &o_w, const occa::memory &o_a, MPI_Comm _comm)
{
  return weightedNorm1Many<T>(N, 1, 0, o_w, o_a, _comm);
}

template <typename T = dfloat>
T weightedNorm1Many(const dlong N,
                    const int Nfields,
                    const dlong fieldOffset,
                    const occa::memory &o_w,
                    const occa::memory &o_a,
                    MPI_Comm _comm)
{
  if (timer) {
    platform->timer.tic("dotp");
  }
  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<T>(Nblock);

  auto h_scratch = getScratch<T>(o_scratch.size(), true);
  auto scratch = h_scratch.template ptr<T>();

  linAlgLaunchKernel(getKnlPrefix<T>() + "weightedNorm1Many", Nblock, N, Nfields, fieldOffset, o_w, o_a, o_scratch);

  double norm = 0;
  if (serial) {
    norm = *((T *)o_scratch.ptr());
  } else {
    o_scratch.copyTo(scratch);
    for (dlong n = 0; n < Nblock; ++n) {
      norm += scratch[n];
    }
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  if (timer) {
    platform->timer.toc("dotp");
  }

  return norm;
}

dfloat weightedSqrSum(const dlong N, const occa::memory &o_w, const occa::memory &o_a, MPI_Comm _comm)
{
  const int Nblock = (N + blocksize - 1) / blocksize;

  auto o_scratch = getScratch<dfloat>(Nblock);

  auto h_scratch = getScratch<dfloat>(o_scratch.size(), true);
  auto scratch = h_scratch.ptr<dfloat>();

  double sum = 0;
  if (N > 1) {
    linAlgLaunchKernel(getKnlPrefix<dfloat>() + "weightedSqrSum", Nblock, N, o_w, o_a, o_scratch);

    if (serial) {
      sum = *((dfloat *)o_scratch.ptr());
    } else {
      o_scratch.copyTo(scratch);
      for (dlong n = 0; n < Nblock; ++n) {
        sum += scratch[n];
      }
    }
  } else {
    dfloat w, a;
    o_w.copyTo(&w, N);
    o_a.copyTo(&a, N);
    sum = (w * a) * (w * a) / N;
  }

  if (_comm != MPI_COMM_SELF) {
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, _comm);
  }

  return sum;
}

template <typename T = dfloat> void rescale(const double newMin, const double newMax, occa::memory &o_a, MPI_Comm _comm)
{
  double mn = this->min<T>(o_a.size(), o_a, _comm);
  double mx = this->max<T>(o_a.size(), o_a, _comm);

  MPI_Allreduce(MPI_IN_PLACE, &mn, 1, MPI_DOUBLE, MPI_MIN, _comm);
  MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, _comm);
  const auto fac = (newMax - newMin)/(mx - mn);

  this->add<T>(o_a.size(), (newMin - fac*mn)/fac, o_a);
  this->scale<T>(o_a.size(), fac, o_a);
}; 
