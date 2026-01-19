//.. NOTE: this modified version has been copied from:
//..       https://github.com/yslan/nekrsExamples/blob/master/tavg/tavg.hpp
#if !defined(nekrs_tavg_hpp_)
#define nekrs_tavg_hpp_

/*
     Statistics can be obtained from runtime averages:

     <X>    := AVG(X)
     <X'Y'> := AVG(X*Y) - AVG(X)*AVG(Y)
*/

#include "nekrsSys.hpp"
#include "mesh.h"
#include "nrs.hpp" // FIXME

namespace tavg
{
void buildKernel(occa::properties kernelInfo);
void run(double time);
void setup(dlong fieldOffset, const std::vector< std::vector<deviceMemory<dfloat>> >& fields);
void setup(nrs_t* nrs);
void outfld(mesh_t *mesh, bool FP64 = true, bool reset_ = true);
void reset();
void free();
deviceMemory<double> o_avg();
}

#endif
