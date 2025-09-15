#include "re2Reader.hpp"
#include <cstring>

void re2::nelg(const std::string &meshFile, int &nelgt, int &nelgv, MPI_Comm comm)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  if (rank == 0) {
    const auto hdr = [&]() {
      auto fp = fopen(meshFile.c_str(), "r");
      nekrsCheck(!fp, MPI_COMM_SELF, EXIT_FAILURE, "Cannot find %s!\n", meshFile.c_str());

      const auto re2HeaderSize = 80;
      std::vector<char> buf(re2HeaderSize + 1); // leave space for '\0'
      nekrsCheck(fgets(buf.data(), static_cast<int>(buf.size()), fp) == nullptr, 
                 MPI_COMM_SELF, EXIT_FAILURE, "failed to read header of %s!\n", meshFile.c_str());

      fclose(fp);
      return buf;
    }();

    char ver[6];
    sscanf(hdr.data(), "%5s", ver);

    int ndim;
    if (strcmp(ver, "#v004") == 0) {
      sscanf(hdr.data(), "%5s %d %d %d", ver, &nelgt, &ndim, &nelgv);
    } else if (strcmp(ver, "#v001") == 0 || strcmp(ver, "#v002") == 0 || strcmp(ver, "#v003") == 0) {
      sscanf(hdr.data(), "%5s %9d %1d %9d", ver, &nelgt, &ndim, &nelgv);
    } else {
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "Unsupported re2 version %5s!\n", ver);
    }

    nekrsCheck(ndim != 3, MPI_COMM_SELF, EXIT_FAILURE, "\nUnsupported ndim=%d read from re2 header!\n", ndim);

    nekrsCheck(nelgt <= 0 || nelgv <= 0 || nelgv > nelgt,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "\nInvalid nelgt=%d / nelgv=%d read from re2 header!\n",
               nelgt,
               nelgv);
  }

  MPI_Bcast(&nelgt, 1, MPI_INT, 0, comm);
  MPI_Bcast(&nelgv, 1, MPI_INT, 0, comm);
}
