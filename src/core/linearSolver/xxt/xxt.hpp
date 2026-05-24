#ifndef XXT_HPP
#define XXT_HPP

#include "platform.hpp"

struct xxt;
struct comm;

class xxt_t {

public:
  ~xxt_t();

  xxt_t(const std::vector<hlong>& globalRowIds,
        const std::vector<dlong>& Ai,
        const std::vector<dlong>& Aj,
        const std::vector<dfloat>& Av,
        bool null_space,
        MPI_Comm ce_,
        bool verbose);


  void statistics();

  template <typename T = dfloat>
  void solve(const occa::memory& o_b, occa::memory& o_x)
  {
    if (!bHost.isInitialized()) {
      bHost = platform->device.mallocHost<T>(o_b.size());
    }
    o_b.copyTo(bHost);

    if (!xHost.isInitialized()) {
      xHost = platform->device.mallocHost<T>(o_x.size());
    }

    if (o_x.dtype() == occa::dtype::get<float>()) {
      if (!bHostDouble.isInitialized()) {
        bHostDouble = platform->device.mallocHost<double>(bHost.size());
      }
      auto b = bHostDouble.ptr<double>();
      auto bHostPtr = bHost.ptr<T>();
      const auto bHostSize = bHost.size();
      for(int i = 0; i < bHostSize; i++) {
        b[i] = bHostPtr[i];
      }

      if (!xHostDouble.isInitialized()) {
        xHostDouble = platform->device.mallocHost<double>(xHost.size());
      }
      auto x = xHostDouble.ptr<double>();
      auto xHostPtr = xHost.ptr<T>();

       // x is not read on input
      _solve(b, x);
 
      const auto xHostSize = xHost.size();
      for(int i = 0; i < xHostSize; i++) {
        xHostPtr[i] = x[i];
      }
    } else {
      _solve(bHost.ptr<double>(), xHost.ptr<double>());
    }

    o_x.copyFrom(xHost);
  };
  
private:
  struct comm *c;
  struct xxt *data = nullptr;

  occa::memory bHost;
  occa::memory bHostDouble;

  occa::memory xHost;
  occa::memory xHostDouble;

  void _solve(double *b, double *x);
};

#endif
