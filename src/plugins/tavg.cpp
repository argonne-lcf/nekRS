//.. NOTE: this modified version has been copied from:
//..       https://github.com/yslan/nekrsExamples/blob/master/tavg/tavg.cpp
#include "platform.hpp"
#include "tavg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "iofldFactory.hpp"

// private members
namespace
{
dlong fieldOffset;
int Nscalar; // avg all

std::vector< std::vector<deviceMemory<dfloat>> > userFieldList;
occa::memory o_AVG;

occa::kernel E1Kernel;
occa::kernel E2Kernel;
occa::kernel E3Kernel;
occa::kernel E4Kernel;

std::unique_ptr<iofld> fldWriter;
std::unique_ptr<iofld> fldWriter_avg, fldWriter_rms, fldWriter_rm2;

bool buildKernelCalled = false;
bool setupCalled = false;
bool avg_all = false; // nek5000's avg_all mode

int counter = 0;

double atime;
double timel;

int outfldCounter = 0;
} // namespace

static void E1(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EX)
{
  E1Kernel(N, fieldOffset, nflds, a, b, o_x, o_EX);
}

static void E1(dlong N, dlong fieldOffset, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EX)
{
  E1Kernel(N, fieldOffset, nflds, a, b, o_x, o_EX);
}

static void
E2(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_y, occa::memory o_EXY)
{
  E2Kernel(N, fieldOffset, nflds, a, b, o_x, o_y, o_EXY);
}

static void E3(dlong N,
               dfloat a,
               dfloat b,
               int nflds,
               occa::memory o_x,
               occa::memory o_y,
               occa::memory o_z,
               occa::memory &o_EXYZ)
{
  E3Kernel(N, fieldOffset, nflds, a, b, o_x, o_y, o_z, o_EXYZ);
}

static void E4(dlong N,
               dfloat a,
               dfloat b,
               int nflds,
               occa::memory o_1,
               occa::memory o_2,
               occa::memory o_3,
               occa::memory o_4,
               occa::memory &o_E4)
{
  E4Kernel(N, fieldOffset, nflds, a, b, o_1, o_2, o_3, o_4, o_E4);
}

void tavg::buildKernel(occa::properties kernelInfo)
{
  auto buildKernel = [&kernelInfo](const std::string& kernelName)
  {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/plugins/");
    const auto fileName = path + "E.okl";
    const auto reqName = "tavg::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  E1Kernel = buildKernel("E1");
  E2Kernel = buildKernel("E2");
  E3Kernel = buildKernel("E3");
  E4Kernel = buildKernel("E4");
}

void tavg::reset()
{
  counter = 0;
  atime = 0;
}

void tavg::run(double time)
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  if (!counter) {
    atime = 0;
    timel = time;
  }
  counter++;

  const double dtime = time - timel;
  atime += dtime;

  if (atime == 0 || dtime == 0) {
    return;
  }

  const dfloat b = dtime / atime;
  const dfloat a = 1 - b;

  if (userFieldList.size()) {
    int cnt = 0;
    for (auto &entry : userFieldList) {
      auto o_avg = o_AVG.slice(cnt * fieldOffset, fieldOffset);
      const auto N = fieldOffset;

      if (entry.size() == 1) {
        E1(N, a, b, 1, entry.at(0), o_avg);
      } else if (entry.size() == 2) {
        E2(N, a, b, 1, entry.at(0), entry.at(1), o_avg);
      } else if (entry.size() == 3) {
        E3(N, a, b, 1, entry.at(0), entry.at(1), entry.at(2), o_avg);
      } else if (entry.size() == 4) {
        E4(N, a, b, 1, entry.at(0), entry.at(1), entry.at(2), entry.at(3), o_avg);
      }
      cnt++;
    }
  }

  timel = time;
}

void tavg::setup(dlong _fieldOffset, const std::vector< std::vector<deviceMemory<dfloat>> >& flds)
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  nekrsCheck(!buildKernelCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "called prior tavg::buildKernel()!");

  userFieldList = flds;

  for (auto &entry : userFieldList) {
    nekrsCheck(entry.size() < 1 || entry.size() > 4,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "invalid number of vectors in one of the user list entries!");

    for (auto &entry_i : entry) {
      nekrsCheck(entry_i.length() < _fieldOffset,
                 platform->comm.mpiComm,
                 EXIT_FAILURE,
                 "%s\n",
                 "vector size in one of the user list entries smaller than fieldOffset");
    }
  }

  fieldOffset = _fieldOffset;
  o_AVG = platform->device.malloc<double>(userFieldList.size() * fieldOffset);

  setupCalled = true;
}

void tavg::setup(nrs_t *nrs) // avg_all
{
  avg_all = true;
  Nscalar = nrs->Nscalar;

  std::vector< std::vector<deviceMemory<dfloat>> > tavgFields;
  deviceMemory<dfloat> o_u(nrs->o_U.slice(0 * nrs->fieldOffset , nrs->fieldOffset));
  deviceMemory<dfloat> o_v(nrs->o_U.slice(1 * nrs->fieldOffset , nrs->fieldOffset));
  deviceMemory<dfloat> o_w(nrs->o_U.slice(2 * nrs->fieldOffset , nrs->fieldOffset));
  deviceMemory<dfloat> o_p(nrs->o_P.slice(0 * nrs->fieldOffset , nrs->fieldOffset));

  // E[X]
  tavgFields.push_back({o_u});
  tavgFields.push_back({o_v});
  tavgFields.push_back({o_w});
  tavgFields.push_back({o_p});
  for (int is=0; is<nrs->Nscalar; is++) {
    deviceMemory<dfloat> o_s(nrs->cds->o_S.slice(nrs->cds->fieldOffsetScan[is], nrs->cds->fieldOffset[is]));
    tavgFields.push_back({o_s});
  }

  // E[X^2]
  tavgFields.push_back({o_u, o_u});
  tavgFields.push_back({o_v, o_v});
  tavgFields.push_back({o_w, o_w});
  tavgFields.push_back({o_p, o_p});
  for (int is=0; is<nrs->Nscalar; is++) {
    deviceMemory<dfloat> o_s(nrs->cds->o_S.slice(nrs->cds->fieldOffsetScan[is], nrs->cds->fieldOffset[is]));
    tavgFields.push_back({o_s, o_s});
  }

  // E[X,Y]
  tavgFields.push_back({o_u, o_v});
  tavgFields.push_back({o_v, o_w});
  tavgFields.push_back({o_w, o_u});

  tavg::setup(nrs->fieldOffset, tavgFields);
}

void tavg::outfld(mesh_t *mesh, bool FP64, bool reset_)
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  //.. instantiate the casename variable
  const std::string casename = platform->options.getArgs("CASENAME");

  if (userFieldList.size() == 0) return;

  const bool outXYZ = mesh && outfldCounter == 0;

  if (avg_all) {

    auto iofldWrapper = [&](std::unique_ptr<iofld> &iofld, int &idx, std::string fileName, bool hasScalar)
    {
      if (!iofld) {
        iofld = iofldFactory::create();
        if (platform->comm.mpiRank == 0) {
          printf("create a new iofldFactory... %s\n", fileName.c_str());
        }
      }

      if (!iofld->isInitialized()) {
        iofld->open(mesh, iofld::mode::write, fileName);
        iofld->writeAttribute("precision", (FP64) ? "64" : "32");

        std::vector<occa::memory> o_V;
        o_V.push_back(o_AVG.slice((idx+0)*fieldOffset, mesh->Nlocal));
        o_V.push_back(o_AVG.slice((idx+1)*fieldOffset, mesh->Nlocal));
        o_V.push_back(o_AVG.slice((idx+2)*fieldOffset, mesh->Nlocal));
        iofld->addVariable("velocity", o_V);
        idx += 3;
      
        if (hasScalar) {
          auto o_p = std::vector<occa::memory>{o_AVG.slice((idx+0)*fieldOffset, mesh->Nlocal)};
          iofld->addVariable("pressure", o_p);
          idx += 1;
      
          for (int i = 0; i < Nscalar; i++) {
            if (platform->options.compareArgs("SCALAR" + scalarDigitStr(i) + " CHECKPOINTING", "TRUE")) {
              const auto temperatureExists = platform->options.compareArgs("SCALAR00 IS TEMPERATURE", "TRUE");
              std::vector<occa::memory> o_Si = {o_AVG.slice((idx+i)*fieldOffset, mesh->Nlocal)};
              if (i == 0 && temperatureExists) {
                iofld->addVariable("temperature", o_Si);
              } else {
                const auto is = (temperatureExists) ? i - 1 : i;
                iofld->addVariable("scalar" + scalarDigitStr(is), o_Si);
              }
            }
          }
          idx += Nscalar;
        }
      }

      iofld->addVariable("time", atime);
      iofld->writeAttribute("outputmesh", (outXYZ) ? "true" : "false");
      iofld->process();
    }; // iofldWrapper

    int idx = 0;
    //.. add the casename to the name of the avg/rms/rm2 checkpoint files
    iofldWrapper(fldWriter_avg, idx, "avg"+casename, true);
    iofldWrapper(fldWriter_rms, idx, "rms"+casename, true);
    iofldWrapper(fldWriter_rm2, idx, "rm2"+casename, false);

  } else {
  
    fldWriter = iofldFactory::create();
  
    if (!fldWriter->isInitialized()) {
      fldWriter->open(mesh, iofld::mode::write, "tavg");
  
      fldWriter->writeAttribute("precision", (FP64) ? "64" : "32");
   
      fldWriter->addVariable("time", atime);
   
      for(int i = 0; i < userFieldList.size(); i++) {
        fldWriter->addVariable("scalar" + scalarDigitStr(i), std::vector<occa::memory>{o_AVG.slice(i * fieldOffset, mesh->Nlocal)});
      }  
    }
  
    fldWriter->writeAttribute("outputmesh", (outXYZ) ? "true" : "false");
    fldWriter->process();
  }

  if (reset_) {
    atime = 0; // reset
  }
  outfldCounter++;
}

deviceMemory<double> tavg::o_avg()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");
  deviceMemory<double> d_AVG(o_AVG);
  return d_AVG;
}

void tavg::free()
{
  userFieldList.clear();
  o_AVG.free();
  fldWriter.reset();
}
