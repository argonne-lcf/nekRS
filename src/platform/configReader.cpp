#include "nekrsSys.hpp"
#include "setEnvVars.hpp"

#include "inipp.hpp"

#if 0
#define UPPER(a)                                                                                             \
{                                                                                                            \
transform(a.begin(), a.end(), a.begin(), std::ptr_fun<int, int>(std::toupper));                              \
}
#define LOWER(a)                                                                                             \
{                                                                                                            \
transform(a.begin(), a.end(), a.begin(), std::ptr_fun<int, int>(std::tolower));                              \
}
#endif

void configRead(MPI_Comm comm)
{
  std::string installDir{getenv("NEKRS_HOME")};
  nekrsCheck(installDir.empty(),
             comm,
             EXIT_FAILURE,
             "\n%s",
             "nERROR: The environment variable NEKRS_HOME is not defined!\n");

  // read config file
  const std::string configFile = installDir + "/nekrs.conf";
  const char *ptr = realpath(configFile.c_str(), NULL);
  nekrsCheck(!ptr, comm, EXIT_FAILURE, "\nCannot find %s\n", configFile.c_str());

  int rank;
  MPI_Comm_rank(comm, &rank);

  std::stringstream is;
  {
    char *rbuf;
    long fsize;
    if (rank == 0) {
      FILE *f = fopen(configFile.c_str(), "rb");
      fseek(f, 0, SEEK_END);
      fsize = ftell(f);
      fseek(f, 0, SEEK_SET);
      rbuf = new char[fsize];
      const auto readCnt = fread(rbuf, 1, fsize, f);
      fclose(f);
    }
    MPI_Bcast(&fsize, sizeof(fsize), MPI_BYTE, 0, comm);
    if (rank != 0) {
      rbuf = new char[fsize];
    }
    MPI_Bcast(rbuf, fsize, MPI_CHAR, 0, comm);
    is.write(rbuf, fsize);
    delete[] rbuf;
  }
  inipp::Ini ini;
  ini.parse(is, false);
  ini.interpolate();

  setEnvVars(installDir, ini);
}
