#if !defined(nekrs_bcast_hpp_)
#define nekrs_bcast_hpp_

#include "nekrsSys.hpp"

void fileBcast(const fs::path &srcPath, const fs::path &dstPath,
               MPI_Comm comm, int verbose);

#endif
