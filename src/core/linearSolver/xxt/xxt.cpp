#include "platform.hpp"
#include "xxt.hpp"
#include "xxt.h"

xxt_t::~xxt_t()
{
  xxt_free(data);
  comm_free(c);
}

xxt_t::xxt_t(const std::vector<hlong>& ids,
             const std::vector<dlong>& Ai,
             const std::vector<dlong>& Aj,
             const std::vector<dfloat>& Av,
             bool null_space,
             MPI_Comm ce_,
             bool verbose)
{
  const double tStart = MPI_Wtime();
  const auto nnz = Av.size();
 
  std::vector<uint> _Ai(nnz);
  std::vector<uint> _Aj(nnz);
  std::vector<double> _Av(nnz);

  for (int i=0; i < nnz; i++) {
    _Ai[i] = Ai[i];
    _Aj[i] = Aj[i];
    _Av[i] = Av[i];
  }

  std::vector<ulong> _ids(ids.size());
  for (int i=0; i < _ids.size(); i++) {
    _ids[i] = ids[i];
  }

  if (platform->comm.mpiRank() == 0) {
    std::cout << "\nsetup XXT ...\n" << std::flush;
  }
 
  c = new comm();
  comm_init(c, ce_);

  data = xxt_setup(_ids.size(), 
                   _ids.data(), 
                   nnz, 
                   _Ai.data(), 
                   _Aj.data(), 
                   _Av.data(), 
                   null_space, 
                   c); 

  statistics();

  if (platform->comm.mpiRank() == 0) {
    printf("\ndone (%gs)\n", MPI_Wtime() - tStart);
    fflush(stdout);
  }
}

void xxt_t::_solve(double *b, double *x)
{
  xxt_solve(x, data, b);
};

void xxt_t::statistics()
{
  xxt_stats(data);
};
