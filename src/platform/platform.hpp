#ifndef platform_hpp_
#define platform_hpp_
#include <set>
#include "nekrsSys.hpp"
#include "QQt.hpp"
#include "flopCounter.hpp"
#include "timer.hpp"
#include "comm.hpp"
#include "par.hpp"
#include "device.hpp"
#include "device.tpp"
#include "kernelManager.hpp"

class app_t;

const occa::memory o_NULL;

class setupAide;
class linAlg_t;
class flopCounter_t;

struct platform_t {
public:
  platform_t(setupAide &_options, MPI_Comm _commg, MPI_Comm _comm);
  void bcastJITKernelSourceFiles();

  static platform_t *getInstance(setupAide &_options, MPI_Comm _commg, MPI_Comm _comm)
  {
    if (!singleton) {
      singleton = new platform_t(_options, _commg, _comm);
    }
    return singleton;
  }

  static platform_t *getInstance()
  {
    return singleton;
  }

  bool multiSession() const
  {
    int result = 0;
    MPI_Comm_compare(comm.mpiCommParent(), comm.mpiComm(), &result);
    return (result != 0) ? true : false;
  }

  bool serial() const { return _serial; };

private:
  static platform_t *singleton;

public:
  setupAide &options;
  int warpSize;
  comm_t comm;
  device_t device;
  occa::properties kernelInfo;
  timer::timer_t timer;
  occa::memoryPool deviceMemoryPool;
  occa::memoryPool memoryPool;
  kernelManager_t kernelRequests;
  Par *par;
  app_t *app;
  bool _serial;
  linAlg_t *linAlg;
  std::unique_ptr<flopCounter_t> flopCounter;
  int exitValue;
  std::string tmpDir;
  bool cacheLocal;
  bool cacheBcast;
  bool buildOnly;

  std::string callerScope;

  occa::kernel copyDfloatToPfloatKernel;
  occa::kernel copyPfloatToDfloatKernel;
  occa::kernel copyDfloatToDoubleKernel;
  occa::kernel copyDfloatToFloatKernel;
  occa::kernel copyDoubleToDfloatKernel;
  occa::kernel copyFloatToDfloatKernel;
  occa::kernel copyFloatToDoubleKernel;
  occa::kernel copyDoubleToFloatKernel;

  bool verbose() const
  {
    return options.compareArgs("VERBOSE", "TRUE");
  }
};

template <typename T = dfloat>
static bool o_isfinite(const occa::memory& o_u) 
{
  auto h_tmp = platform->memoryPool.template reserve<T>(o_u.size());
  h_tmp.copyFrom(o_u);
  auto tmp = h_tmp.template ptr<T>();
  for (int i = 0; i < h_tmp.size(); i++) {
    if (std::isfinite(tmp[i])) return true;
  }
  return false;
};

#endif

#include "occaWrapper.hpp"
