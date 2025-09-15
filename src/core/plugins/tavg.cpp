#include "platform.hpp"
#include "tavg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "iofldFactory.hpp"
#include "iofldNek.hpp"

bool tavg::buildKernelCalled = false;
occa::kernel tavg::E1Kernel;
occa::kernel tavg::E2Kernel;
occa::kernel tavg::E3Kernel;
occa::kernel tavg::E4Kernel;

// private members
void tavg::E1(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EX)
{
  E1Kernel(N, fieldOffset_, nflds, a, b, o_x, o_EX);
}

void tavg::E2(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_y, occa::memory o_EXY)
{
  E2Kernel(N, fieldOffset_, nflds, a, b, o_x, o_y, o_EXY);
}

void tavg::E3(dlong N,
              dfloat a,
              dfloat b,
              int nflds,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_z,
              occa::memory &o_EXYZ)
{
  E3Kernel(N, fieldOffset_, nflds, a, b, o_x, o_y, o_z, o_EXYZ);
}

void tavg::E4(dlong N,
              dfloat a,
              dfloat b,
              int nflds,
              occa::memory o_1,
              occa::memory o_2,
              occa::memory o_3,
              occa::memory o_4,
              occa::memory &o_E4)
{
  E4Kernel(N, fieldOffset_, nflds, a, b, o_1, o_2, o_3, o_4, o_E4);
}

void tavg::registerKernels(occa::properties &kernelInfo)
{
  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/core/plugins/");
    const auto fileName = path + "E.okl";
    const auto reqName = "tavg::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  buildKernelCalled = false;

  E1Kernel = buildKernel("E1");
  E2Kernel = buildKernel("E2");
  E3Kernel = buildKernel("E3");
  E4Kernel = buildKernel("E4");

  buildKernelCalled = 1;
}

void tavg::reset()
{
  counter = 0;
  atime = 0;
}

void tavg::run(double time)
{
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
    for (auto [name, entry] : userFieldList) {
      auto o_avg = o_AVG.slice(cnt * fieldOffset_, fieldOffset_);
      const auto N = fieldOffset_;

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

tavg::tavg(dlong fieldOffsetIn, const std::vector<tavg::field> &flds, std::string ioEngine)
{
  nekrsCheck(!buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior tavg::registerKernels()!");

  userFieldList = flds;

  for (auto [name, entry] : userFieldList) {
    nekrsCheck(entry.size() < 1 || entry.size() > 4,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "invalid number of vectors in one of the user list entries!");

    for (auto &entry_i : entry) {
      nekrsCheck(entry_i.length() < fieldOffsetIn,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s\n",
                 "vector size in one of the user list entries smaller than fieldOffset");
    }
  }

  fieldOffset_ = fieldOffsetIn;
  o_AVG = platform->device.malloc<double>(userFieldList.size() * fieldOffset_);

  fldWriter = iofldFactory::create(ioEngine);
}

void tavg::writeToFile(mesh_t *mesh, bool reset)
{
  if (userFieldList.size() == 0) {
    return;
  }

  const bool outXYZ = mesh && outfldCounter == 0;

  if (!fldWriter->isInitialized()) {
    fldWriter->open(mesh, iofld::mode::write, "tavg");

    if (platform->options.compareArgs("TAVG OUTPUT PRECISION", "FP32")) {
      fldWriter->writeAttribute("precision", "32");
    } else {
      fldWriter->writeAttribute("precision", "64");
    }

    fldWriter->addVariable("time", atime);

    const auto engineTypeIsNek = (dynamic_cast<iofldNek *>(fldWriter.get())) ? true : false;

    for (int i = 0; i < userFieldList.size(); i++) {
      const auto name = engineTypeIsNek ? "scalar" + scalarDigitStr(i) : std::get<0>(userFieldList.at(i));
      fldWriter->addVariable(name, std::vector<occa::memory>{o_AVG.slice(i * fieldOffset_, mesh->Nlocal)});
    }
  }

  fldWriter->writeAttribute("outputmesh", (outXYZ) ? "true" : "false");
  fldWriter->process();

  if (reset) {
    atime = 0;
  }
  outfldCounter++;
}

tavg::~tavg()
{
  userFieldList.clear();
  o_AVG.free();
  fldWriter.reset();
}
