#include <math.h>
#include <vector>
#include "pointInterpolation.hpp"

static int ciMode = 0;
double numCVODERHS = 0;

#define EPS 5e-3

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("POLYNOMIAL DEGREE", std::string("7"));
  options.setArgs("SOLUTION OUTPUT INTERVAL", "0");
  options.setArgs("END TIME", std::string("2"));
  options.setArgs("DT", std::string("2e-2"));
  options.setArgs("SUBCYCLING STEPS", std::string("0"));
  if (ciMode == 2) {
    options.setArgs("SUBCYCLING STEPS", std::string("1"));
  }
  options.setArgs("BDF ORDER", "2");
  options.setArgs("ADVECTION TYPE", "CONVECTIVE+CUBATURE");
  options.setArgs("FLUID VELOCITY SOLVER TOLERANCE", std::string("1e-06"));
  options.setArgs("FLUID PRESSURE SOLVER TOLERANCE", std::string("1e-04"));
  options.setArgs("SCALAR01 SOLVER TOLERANCE", std::string("1e-06"));
  options.setArgs("VARIABLEPROPERTIES", "TRUE");

  // set up CVODE solver details
  if (ciMode == 3) {
    options.setArgs("CVODE", "TRUE");
    options.setArgs("CVODE INTEGRATOR", "BDF");
    options.setArgs("CVODE MAX TIMESTEPPER ORDER", "3");
    options.setArgs("CVODE EPS LIN", "0.05");
    options.setArgs("CVODE STOP TIME", "TRUE");

    const auto hmax = 1;
    options.setArgs("CVODE HMAX", to_string_f(hmax));

    const auto maxSteps = 10000;
    options.setArgs("CVODE MAX STEPS", std::to_string(maxSteps));

    options.setArgs("CVODE RELATIVE TOLERANCE", "1e-06");
    options.setArgs("CVODE ABSOLUTE TOLERANCE", "1e-08");
    options.setArgs("CVODE GMR VECTORS", "20");

    options.setArgs("SCALAR00 SOLVER", "CVODE");
  }
}

void ciTestPointInterpolation(nrs_t *nrs, double time, int tstep)
{
  if (ciMode != 1) {
    return;
  }

  auto& scalar = nrs->scalar;
  auto interp = pointInterpolation_t(scalar->mesh(1), platform->comm.mpiComm());

  std::vector<dfloat> x = {4, 4, 4};
  std::vector<dfloat> y = {-0.25, 0.25, 1.0};
  std::vector<dfloat> z = {0.5, 0.5, 0.5};

  int nPoints = x.size();
  interp.setPoints(x, y, z);
  const auto verbosity = pointInterpolation_t::VerbosityLevel::Basic;
  interp.find(verbosity);

  auto o_Tint = platform->device.malloc<dfloat>(nPoints);
  interp.eval(1, scalar->fieldOffset(), scalar->o_solution("temperature"), nPoints, o_Tint);
  std::vector<dfloat> Tint(nPoints);
  o_Tint.copyTo(Tint.data(), nPoints);

  std::vector<dfloat> Texpected = {13.921, 0.00133, 9.62969};

  dfloat relError = 0;
  for (int i = 0; i < nPoints; ++i) {
    relError = std::max(relError, std::abs(Tint[i] - Texpected[i]));
  }
  if (platform->comm.mpiRank() == 0) {
    std::cout << "Point interpolation relErr=" << relError << std::endl;
  } 

  const auto tol = 5e-4;
  if (relError > tol) {
    if (platform->comm.mpiRank() == 0) {
      std::cout << "Point interpolation test failed" << std::endl;
    }

    CiFailTest();
    nekrsCheck(platform->exitValue, platform->comm.mpiComm(), EXIT_FAILURE, "%s", "");
  }
}

void ciTestErrors(nrs_t *nrs, double time, int tstep)
{
  auto& scalar = nrs->scalar;

  if (scalar->cvode) {
    numCVODERHS += scalar->cvode->numRHSEvals();
    numCVODERHS += scalar->cvode->numLinIters();
  }

  if (!nrs->lastStep) {
    return;
  }

  numCVODERHS /= tstep;

  ciTestPointInterpolation(nrs, time, tstep);

  const int rank = platform->comm.mpiRank();

  nrs->copyToNek(time, tstep);
  nek::userchk();

  auto err = nek::ptr<double>("errors");
  double vxErr, sErr;

  vxErr = abs((err[0] - 2.06559) / err[0]);
  sErr = abs((err[1] - 5.16746) / err[1]);

  if (rank == 0) {
    printf("relative error to target: vx=%g s=%g\n", vxErr, sErr);
  }

  (vxErr < EPS && sErr < EPS) ? (CiPassTest()) : (CiFailTest());

  if (scalar->cvode) {
    dfloat expectedCVODERHS = 13.11;
    if (rank == 0) {
      printf("CVODE RHS evaluations per timestep: expected = %g, actual = %g\n",
             expectedCVODERHS,
             numCVODERHS);
    }
    numCVODERHS > 1.1 * expectedCVODERHS ? (CiFailTest()) : (CiPassTest());
  }
}
