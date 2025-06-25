#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/modes/dpcpp/streamTag.hpp>

namespace occa {
  namespace dpcpp {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   ::sycl::queue commandQueue_) :
      modeStream_t(modeDevice_, properties_),
      commandQueue(commandQueue_) {}

    void stream::finish()
    {
      OCCA_DPCPP_ERROR("stream::finish",
                       commandQueue.wait_and_throw());
    }

    // TODO: Add compile-time and run-time checks for extension support
    occa::streamTag stream::tag()
    {
      ::sycl::event e;
    #if SYCL_EXT_ONEAPI_PROFILING_TAG
      OCCA_DPCPP_ERROR("stream::tag",
        e = sycl::ext::oneapi::experimental::submit_profiling_tag(commandQueue));
    #endif
      return new dpcpp::streamTag(modeDevice, e);
    }

    occa::dpcpp::streamTag stream::memcpy(void * dest,const void* src, occa::udim_t num_bytes)
    {
      ::sycl::event e{commandQueue.memcpy(dest, src, num_bytes)};
      return dpcpp::streamTag(modeDevice, e);
    }

    void* stream::unwrap() {
      return static_cast<void*>(&commandQueue);
    }
  }
}
