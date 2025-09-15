#include <registerKernels.hpp>
#include <tuple>

void registerLinAlgKernels()
{
  occa::properties kernelInfo = platform->kernelInfo;

  const std::string oklDir = getenv("NEKRS_KERNEL_DIR") + std::string("/platform/linAlg/");
  const bool serial = platform->serial();

  const std::string extension = serial ? ".c" : ".okl";
  const std::vector<std::pair<std::string, bool>> allKernels{
      {"fill", false},
      {"vabs", false},
      {"add", false},
      {"scale", false},
      {"scaleMany", false},
      {"axpby", true},
      {"axpbyMany", true},
      {"axpbyz", false},
      {"axpbyzMany", false},
      {"axmy", true},
      {"axmyMany", true},
      {"axmyVector", true},
      {"axmyz", true},
      {"axmyzMany", false},
      {"ady", false},
      {"adyz", false},
      {"adyMany", false},
      {"axdy", false},
      {"aydx", false},
      {"aydxMany", false},
      {"axdyz", false},
      {"sum", false},
      {"sumMany", false},
      {"min", false},
      {"max", false},
      {"amax", false},
      {"amaxMany", false},
      {"norm1", true},
      {"norm1Many", true},
      {"norm2", true},
      {"norm2Many", true},
      {"weightedNorm1", true},
      {"weightedNorm1Many", true},
      {"weightedSqrSum", true},
      {"innerProd", true},
      {"weightedInnerProd", true},
      {"weightedInnerProdMany", true},
      {"weightedInnerProdMulti", false},
      {"weightedInnerProdMultiDevice", false},
      {"crossProduct", false},
      {"dotProduct", false},
      {"dotConstProduct", false},
      {"unitVector", false},
      {"entrywiseMag", false},
      {"linearCombination", false},
      {"relativeError", false},
      {"absoluteError", false},
      {"magSqrVector", false},
      {"magVector", false},
      {"magSqrSymTensor", false},
      {"magSqrSymTensorDiag", false},
      {"magSqrTensor", false},
      {"mask", false},
  };

  std::string kernelName;
  std::string fileName;
  bool nativeSerialImplementation;
  const std::string prefix = "linAlg::";

  for (bool useFloat : {true, false}) {
    for (auto &&nameAndSerialImpl : allKernels) {
      std::tie(kernelName, nativeSerialImplementation) = nameAndSerialImpl;

      fileName = kernelName;
      occa::properties props = kernelInfo;
      if (useFloat) {
        kernelName = "f_" + kernelName;
        props["defines/dfloat"] = "float";
      } else {
        kernelName = "d_" + kernelName;
        props["defines/dfloat"] = "double";
      }

      const std::string extension = (serial && nativeSerialImplementation) ? ".c" : ".okl";

      platform->kernelRequests.add(prefix + kernelName, oklDir + fileName + extension, props);
    }
  }

  {
    auto props = kernelInfo;
    props["defines/dfloat"] = hlongString;
    kernelName = "sum";
    fileName = oklDir + kernelName + ".okl";
    platform->kernelRequests.add(prefix + "hlong-" + kernelName, fileName, props);
  }
}
