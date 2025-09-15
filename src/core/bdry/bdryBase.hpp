#if !defined(nekrs_bcmap_hpp_)
#define nekrs_bcmap_hpp_

#include "nekrsSys.hpp"
#include "mesh3D.h"
#include "bcType.h"
#include "deviceMemory.hpp" 
#include <set>

class bdryBase
{

public:
  static constexpr int bcType_interpolation = p_bcType_interpolation;
  static constexpr int bcType_zeroDirichlet = p_bcType_zeroDirichlet;
  static constexpr int bcType_udfDirichlet = p_bcType_udfDirichlet;
  static constexpr int bcType_zeroDirichletX_zeroNeumann = p_bcType_zeroDirichletX_zeroNeumann;
  static constexpr int bcType_zeroDirichletY_zeroNeumann = p_bcType_zeroDirichletY_zeroNeumann;
  static constexpr int bcType_zeroDirichletZ_zeroNeumann = p_bcType_zeroDirichletZ_zeroNeumann;
  static constexpr int bcType_zeroDirichletN_zeroNeumann = p_bcType_zeroDirichletN_zeroNeumann;
  static constexpr int bcType_zeroDirichletX_udfNeumann = p_bcType_zeroDirichletX_udfNeumann;
  static constexpr int bcType_zeroDirichletY_udfNeumann = p_bcType_zeroDirichletY_udfNeumann;
  static constexpr int bcType_zeroDirichletZ_udfNeumann = p_bcType_zeroDirichletZ_udfNeumann;
  static constexpr int bcType_zeroDirichletN_udfNeumann = p_bcType_zeroDirichletN_udfNeumann;
  static constexpr int bcType_zeroDirichletYZ_zeroNeumann = p_bcType_zeroDirichletYZ_zeroNeumann;
  static constexpr int bcType_zeroDirichletXZ_zeroNeumann = p_bcType_zeroDirichletXZ_zeroNeumann;
  static constexpr int bcType_zeroDirichletXY_zeroNeumann = p_bcType_zeroDirichletXY_zeroNeumann;
  static constexpr int bcType_zeroDirichletT_zeroNeumann = p_bcType_zeroDirichletT_zeroNeumann;
  static constexpr int bcType_zeroNeumann = p_bcType_zeroNeumann;
  static constexpr int bcType_udfNeumann = p_bcType_udfNeumann;
  static constexpr int bcType_udfRobin = p_bcType_udfRobin;
  static constexpr int bcType_none = p_bcType_none;

  virtual ~bdryBase() = default;

  virtual void setup() = 0;

  std::string typeText(int bid, const std::string& field) const;

  bool useNek() const
  {
    return importFromNek;
  };

  int typeId(int bid, std::string field) const
  {
    if (bid < 1) {
      return bcType_none;
    }

    try {
      return bToBc.at({lowerCase(field), bid - 1});
    } catch (const std::out_of_range &oor) {
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "lookup of bid %d field %s failed!\n", bid, field.c_str());
    }

    return -1;
  };

  int size(const std::string &field) const
  {
    int cnt = 0;
    for (auto &entry : bToBc) {
      if (entry.first.first == lowerCase(field)) {
        cnt++;
      }
    }
    return cnt;
  };

  std::map<std::pair<std::string, int>, int> bIdToTypeId() const
  {
    return bToBc;
  };

  void setBcMap(std::string field, bool isVector, const std::vector<int>& map)
  {
    fields.insert({field, isVector});
    for (int i = 0; i < map.size(); i++) {
      bToBc[make_pair(field, i)] = map[i];
    }
  };

    
  void checkAlignment(mesh_t *mesh) const;

  virtual void addKernelConstants(occa::properties &kernelInfo)
  {
    const std::string installDir = getenv("NEKRS_HOME");
    kernelInfo["includes"].asArray();
    kernelInfo["includes"] += installDir + "/include/core/bdry/bcType.h";
  };

  bool hasRobin(std::string field) const
  {
    const auto nid = size(field);

    for (int bid = 1; bid <= nid; bid++) {
      const auto bcType = typeId(bid, field);
      if (bcType == bdryBase::bcType_udfRobin) {
        return true;
      }
    }

    return false;
  };

  bool hasUnalignedMixed(std::string field) const
  {
    const auto nid = size(field);

    for (int bid = 1; bid <= nid; bid++) {
      const auto bcType = typeId(bid, field);
      if (bcType == bdryBase::bcType_zeroDirichletN_zeroNeumann) {
        return true;
      }
      if (bcType == bdryBase::bcType_zeroDirichletN_udfNeumann) {
        return true;
      }
      if (bcType == bdryBase::bcType_zeroDirichletT_zeroNeumann) {
        return true;
      }
    }

    return false;
  };

  int typeElliptic(int bid, const std::string& field, std::string fieldComponent = "") const;

  void setupField(const std::vector<std::string>& slist, const std::string& field, bool isVector = false)
  {              
    if (slist.size() == 0) {
      return;
    }         
    
    if (slist.size()) {
      importFromNek = false;
    
      if (slist.size() == 1 && slist[0] == "none") {
        return;
      }
    }         
              
    fields.insert({lowerCase(field), isVector});
    
    if (isVector) {
      vectorFieldSetup(lowerCase(field), slist);
    } else {
      scalarFieldSetup(lowerCase(field), slist);
    } 
  }

  const std::map<std::string, int> vBcTextToID = {
      //    {"periodic", 0},
      {"zerodirichlet", bdryBase::bcType_zeroDirichlet},
      {"interpolation", bdryBase::bcType_interpolation},
      {"udfdirichlet", bdryBase::bcType_udfDirichlet},
      {"zeroxvalue/zeroneumann", bdryBase::bcType_zeroDirichletX_zeroNeumann},
      {"zerodirichlety/zeroneumann", bdryBase::bcType_zeroDirichletY_zeroNeumann},
      {"zerodirichletz/zeroneumann", bdryBase::bcType_zeroDirichletZ_zeroNeumann},
      {"zerodirichletn/zeroneumann", bdryBase::bcType_zeroDirichletN_zeroNeumann},
      {"zerodirichletx/udfneumann", bdryBase::bcType_zeroDirichletX_udfNeumann},
      {"zerodirichlety/udfneumann", bdryBase::bcType_zeroDirichletY_udfNeumann},
      {"zerodirichletz/udfneumann", bdryBase::bcType_zeroDirichletZ_udfNeumann},
      {"zerodirichletn/udfneumann", bdryBase::bcType_zeroDirichletN_udfNeumann},
      {"zerodirichletyz/zeroneumann", bdryBase::bcType_zeroDirichletYZ_zeroNeumann},
      {"zerodirichletxz/zeroneumann", bdryBase::bcType_zeroDirichletXZ_zeroNeumann},
      {"zerodirichletxy/zeroneumann", bdryBase::bcType_zeroDirichletXY_zeroNeumann},
      // {"zerodirichlett/zeroneumann", bdryBase::bcType_zeroDirichletT_zeroNeumann},
      {"zeroneumann", bdryBase::bcType_zeroNeumann},
      {"none", bdryBase::bcType_none}};

  const std::map<int, std::string> vBcIDToText = {
      //    {0, "periodic"},
      {bdryBase::bcType_zeroDirichlet, "zeroDirichlet"},
      {bdryBase::bcType_interpolation, "interpolation"},
      {bdryBase::bcType_udfDirichlet, "udfDirichlet"},
      {bdryBase::bcType_zeroDirichletX_zeroNeumann, "zeroDirichletX/zeroNeumann"},
      {bdryBase::bcType_zeroDirichletY_zeroNeumann, "zeroDirichletY/zeroNeumann"},
      {bdryBase::bcType_zeroDirichletZ_zeroNeumann, "zeroDirichletZ/zeroNeumann"},
      {bdryBase::bcType_zeroDirichletN_zeroNeumann, "zeroDirichletN/zeroNeumann"},
      {bdryBase::bcType_zeroDirichletX_udfNeumann, "zeroDirichletX/udfNeumann"},
      {bdryBase::bcType_zeroDirichletY_udfNeumann, "zeroDirichletY/udfNeumann"},
      {bdryBase::bcType_zeroDirichletZ_udfNeumann, "zeroDirichletZ/udfNeumann"},
      {bdryBase::bcType_zeroDirichletN_udfNeumann, "zeroDirichletN/udfNeumann"},
      {bdryBase::bcType_zeroDirichletYZ_zeroNeumann, "zeroDirichletYZ/zeroNeumann"},
      {bdryBase::bcType_zeroDirichletXZ_zeroNeumann, "zeroDirichletXZ/zeroNeumann"},
      // {bdryBase::bcType_zeroDirichletT_zeroNeumann, "zeroDirichletT/zeroNeumann"},
      {bdryBase::bcType_zeroNeumann, "zeroNeumann"},
      {bdryBase::bcType_none, "none"}};

  const std::map<std::string, int> sBcTextToID = { //    {"periodic", 0},
      {"interpolation", bdryBase::bcType_interpolation},
      {"udfdirichlet", bdryBase::bcType_udfDirichlet},
      {"udfrobin", bdryBase::bcType_udfRobin},
      {"zeroneumann", bdryBase::bcType_zeroNeumann},
      {"udfneumann", bdryBase::bcType_udfNeumann},
      {"none", bdryBase::bcType_none}};

  const std::map<int, std::string> sBcIDToText = { //    {0, "periodic"},
      {bdryBase::bcType_interpolation, "interpolation"},
      {bdryBase::bcType_udfDirichlet, "udfDirichlet"},
      {bdryBase::bcType_udfRobin, "udfRobin"},
      {bdryBase::bcType_zeroNeumann, "zeroNeumann"},
      {bdryBase::bcType_udfNeumann, "udfNeumann"},
      {bdryBase::bcType_none, "none"}};

  bool hasOutflow(const std::string &field) const;
  bool isOutflow(int bcType) const;
  void printBcTypeMapping(const std::string &field) const;

  deviceMemory<dfloat> o_usrwrk;

protected:
  static std::map<std::string, bool>  fields;
  static std::map<std::pair<std::string, int>, int> bToBc;
  static bool importFromNek;

private:
  void vectorFieldSetup(std::string field, std::vector<std::string> slist);
  void scalarFieldSetup(std::string field, std::vector<std::string> slist);
};

#endif
