#if !defined(compile_kernels_hpp_)
#define compile_kernels_hpp_

#include "platform.hpp"
#include "ellipticBcTypes.h"
#include "mesh.h"
#include "re2Reader.hpp"

occa::properties registerUDFKernels();
void loadUDFKernels();

void registerCoreKernels(occa::properties kernelInfoBC);
void registerLinAlgKernels();
void registerLinearSolverKernels();
void registerPostProcessingKernels();
void registerPointInterpolationKernels();
void registerNekNekKernels();
void registerCvodeKernels();
void registerMeshKernels(occa::properties kernelInfoBC);
void registerScalarKernels(occa::properties kernelInfoBC);
void registerEllipticKernels(std::string section, bool stressForm = false);
void registerEllipticPreconditionerKernels(std::string section);
void registerFluidSolverKernels(occa::properties kernelInfoBC);
void registerGeomSolverKernels(occa::properties kernelInfoBC);

std::string createOptionsPrefix(std::string section);

#endif
