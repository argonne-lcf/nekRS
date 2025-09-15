#if !defined(nekrs_recycling_hpp_)
#define nekrs_recycling_hpp_


#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

class planarCopy   
{

public:

// assumes elements are arranged such that the z-dimension varies the least
planarCopy(mesh_t *mesh, const occa::memory& o_U, const int nFields, const dlong fieldOffset,
          const hlong nElementsSrcPlane, const int bIDtarget, occa::memory &o_out); 

// interpolation-based
planarCopy(mesh_t *mesh, const occa::memory& o_U, const int nFields, const dlong fieldOffset,
           const dfloat xOffsetSrcPlane, const dfloat yOffsetSrcPlane, const dfloat zOffsetSrcPlane, 
           const int bIDtarget, occa::memory &o_out);

static void buildKernel(occa::properties kernelInfo);
void execute();

private:
  void _setup(mesh_t * mesh_, const occa::memory &o_Uin_, const int nFields_, const dlong fieldOffset_, const int bID_, occa::memory &o_out_);

  mesh_t *mesh;
  int nFields;

  dlong fieldOffset;
  dfloat area;  
  int bID;                   
  occa::memory o_bID; 

  pointInterpolation_t *interp;
  occa::memory o_Uint;
  occa::memory o_maskIds;

  occa::memory o_U;
  occa::memory o_out;

  oogs_t *ogs;
};

#endif
