#include <math.h>

static int ciMode = 0;

#define EPS_SST 8.71e-2
#define EPS_DDES 1.74e-1
#define EPS_IDDES 2.54e-1

dfloat hill_step(dfloat x, dfloat w, dfloat h);
dfloat hill_height(dfloat x, dfloat Lx, dfloat w, dfloat h);

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CONSTANT FLOW RATE", "TRUE");
  options.setArgs("CONSTANT FLOW DIRECTION", "X");
  if (ciMode == 1) {
    options.setArgs("RESTART FILE NAME", std::string("sst.fld+time=0"));
  }
  if (ciMode == 2) {
    options.setArgs("RESTART FILE NAME", std::string("ddes.fld+time=0"));
  }
  if (ciMode == 3) {
    options.setArgs("RESTART FILE NAME", std::string("iddes.fld+time=0"));
  }
  options.setArgs("CHECKPOINT INTERVAL", "0");
  options.setArgs("FLUID VISCOSITY", std::to_string(1.0 / 10595.0));
  options.setArgs("FLUID DENSITY", std::string("1.0"));
  options.setArgs("END TIME", std::string("0.5"));
  options.setArgs("ADVECTION TYPE", "CUBATURE+CONVECTIVE");

  options.setArgs("DT", std::string("5e-3"));
  options.setArgs("VARIABLE DT", std::string("TRUE"));
  options.setArgs("TARGET CFL", std::string("1.5"));
  options.setArgs("SUBCYCLING STEPS", std::string("1"));

  options.setArgs("FLUID PRESSURE SOLVER TOLERANCE", std::string("1e-4"));

  options.setArgs("FLUID VELOCITY SOLVER TOLERANCE", std::string("1e-6"));

  options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-6"));

  options.setArgs("SCALAR01 SOLVER TOLERANCE", std::string("1e-6"));
}

void ciTestErrors(nrs_t *nrs, double time, int tstep, const occa::memory& o_avgU)
{
  if (!nrs->lastStep) {
    return;
  }

  auto mesh = nrs->meshV;
  auto &scalar = nrs->scalar;

  auto o_Sij = nrs->strainRate(nrs->fluid->fieldOffset, o_avgU);

  std::vector<int> bidWall = {1};
  auto o_bidWall = platform->device.malloc<int>(bidWall.size(), bidWall.data());

  auto o_tauw = nrs->viscousShearStress(o_bidWall, o_Sij);

  const auto np = (platform->comm.mpiRank() == 0) ? 201 : 0;
  const auto offset = np;

  static pointInterpolation_t *interpolator = nullptr;
  static std::vector<dfloat> xp, yp, zp;
  static deviceMemory<dfloat> o_tauwpx;

  if (!interpolator) {
    auto mesh = nrs->meshV;
    const auto xmin = platform->linAlg->min(mesh->Nlocal, mesh->o_x, platform->comm.mpiComm());
    const auto xmax = platform->linAlg->max(mesh->Nlocal, mesh->o_x, platform->comm.mpiComm());

    if (np) {
      const auto deltax = (xmax - xmin) / (np - 1.0);
      for (int i = 0; i < np; i++) {
        const auto x = xmin + i * deltax;
        const auto y = hill_height(x, 9.0, 1.929, 1.0);
        xp.push_back(x);
        yp.push_back(y);
        zp.push_back(0.0);
      }
      o_tauwpx.resize(offset);
    }

    interpolator = new pointInterpolation_t(nrs->meshV, platform->comm.mpiComm());
    interpolator->setPoints(xp, yp, zp);
    interpolator->find();
  }

  interpolator->eval(1, nrs->fluid->fieldOffset, o_tauw, offset, o_tauwpx);

  if (platform->comm.mpiRank() == 0) {
    // https://turbmodels.larc.nasa.gov/Other_LES_Data/2Dhill_periodic/hill_LES_cf_digitized.dat
    std::ifstream file("refLes.dat");
    std::string line;
    std::vector<dfloat> lesData;
    while (file >> line) {
      lesData.push_back(std::stod(line));
    }
    file.close();

    deviceMemory<dfloat> o_err(lesData);
    platform->linAlg->axpby(lesData.size(), 2.0, o_tauwpx, -1.0, o_err);
    const auto errNorm = platform->linAlg->norm2(lesData.size(), o_err, MPI_COMM_SELF);

#if 0
    std::vector<dfloat> tauwpx(np);
    o_tauwpx.copyTo(tauwpx);

    std::ofstream f("Cf.dat", std::ios::out);
    for (int i = 0; i < np; i++) {
      f << std::scientific << xp[i] << " " << 2.0 * tauwpx[i] << " " << lesData[i] << std::endl;
    }
    f.close();
#endif

    auto eps = EPS_SST;
    if (ciMode == 2) {
      eps = EPS_DDES;
    }
    if (ciMode == 3) {
      eps = EPS_IDDES;
    }
    CiEvalTest("Cf norm Err: " + to_string_f(errNorm), errNorm < eps);
  }
}
