#if !defined(nekrs_tavg_hpp_)
#define nekrs_tavg_hpp_

/*
     Statistics can be obtained from runtime averages:

     <X>    := AVG(X)
     <X'Y'> := AVG(X*Y) - AVG(X)*AVG(Y)
*/

#include "nekrsSys.hpp"
#include "mesh.h"
#include "iofldFactory.hpp"

class tavg
{
public:

using field = std::tuple< std::string, std::vector<deviceMemory<dfloat>> >;

static void registerKernels(occa::properties &kernelInfo);
tavg(dlong fieldOffset, const std::vector<tavg::field>& fields, std::string ioEngine = "");
~tavg();

void run(double time);
void writeToFile(mesh_t *mesh, bool resetAvergingTime = true);
void reset();
void free();
const double& time() const { return atime; }
dlong fieldOffset() const { return fieldOffset_; };
const deviceMemory<double> o_data() const { return deviceMemory<double>(o_AVG); };


private:
dlong fieldOffset_;

std::vector<field> userFieldList;
occa::memory o_AVG;

static occa::kernel E1Kernel;
static occa::kernel E2Kernel;
static occa::kernel E3Kernel;
static occa::kernel E4Kernel;

void E1(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EX);
void E2(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_y, occa::memory o_EXY);
void E3(dlong N,
               dfloat a,
               dfloat b,
               int nflds,
               occa::memory o_x,
               occa::memory o_y,
               occa::memory o_z,
               occa::memory &o_EXYZ);
void E4(dlong N,
               dfloat a,
               dfloat b,
               int nflds,
               occa::memory o_1,
               occa::memory o_2, 
               occa::memory o_3,
               occa::memory o_4,
               occa::memory &o_E4);

std::unique_ptr<iofld> fldWriter = nullptr;

static bool buildKernelCalled;

int counter = 0;
double atime = 0;
double timel = 0;

int outfldCounter = 0;
};

#endif
