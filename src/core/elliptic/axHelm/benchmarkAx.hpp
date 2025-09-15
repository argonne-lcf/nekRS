#include "occa.hpp"

template <typename T, typename TGeo>
occa::kernel benchmarkAx(int Nelements,
                         int Nq,
                         int Ng,
                         bool constCoeff,
                         bool poisson,
                         bool computeGeom,
                         int Ndim,
                         bool stressForm,
                         int verbosity,
                         double targetTime,
                         bool requiresBenchmark,
                         std::string suffix = "");
