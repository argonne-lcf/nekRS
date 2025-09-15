#include <fstream>
#include <set>
#include <utility>

#include "platform.hpp"
#include "udf.hpp"

#include "elliptic.hpp"
#include "alignment.hpp"
#include "nrs.hpp"

void nrs_t::bdry::setup()
{
  std::map<std::string, bool> sectionsPar;

  sectionsPar.insert({"GEOM", true});
  sectionsPar.insert({"FLUID VELOCITY", true});

  int nscal = 0;
  platform->options.getArgs("NUMBER OF SCALARS", nscal);
  for (int i = 0; i < nscal; i++) {
    sectionsPar.insert({"SCALAR" + scalarDigitStr(i), false});
  }

  int count = 0;
  int expectedCount = 0;

  auto process = [&](const std::string &sectionPar, bool isVector) {
    std::vector<std::string> staleOptions;
    for (auto const &option : platform->options) {
      if (option.first.find(sectionPar) != std::string::npos) {
        if (option.first.compare(sectionPar + " SOLVER") == 0 &&
            option.second.find("NONE") == std::string::npos) {
          expectedCount++;
        }

        if (option.first.find("BOUNDARY TYPE MAP") != std::string::npos) {
          count++;

          if (sectionPar == "GEOM" && option.first.find("DERIVED") != std::string::npos) {
            deriveGeomBoundaryConditions(serializeString(option.second, ','));
          } else {
            setupField(serializeString(option.second, ','), sectionPar, isVector);
          }

          staleOptions.push_back(option.first);
        }
      }
    }
    for (auto const &key : staleOptions) {
      platform->options.removeArgs(key);
    }
  };

  for (auto &&[sectionPar, isVector] : sectionsPar) {
    process(sectionPar, isVector);
  }

  nekrsCheck(count > 0 && count != expectedCount,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "boundaryTypeMap specfied for %d fields but not all %d fields!",
             count,
             expectedCount);

  addKernelConstants(platform->kernelInfo);
}

void nrs_t::bdry::deriveGeomBoundaryConditions(std::vector<std::string> velocityBCs)
{
  if (velocityBCs.size() == 0) {
    return;
  }

  geomConditionsDerived = true;

  const std::string field = "geom";

  fields.insert({field, true});

  for (int bid = 0; bid < velocityBCs.size(); bid++) {
    const std::string keyIn = velocityBCs[bid];

    std::string key = "zerodirichletn/zeroneumann"; // default

    if (keyIn.compare("none") == 0) {
      key = "none";
    }

    if (keyIn.compare("zerodirichlet") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("udfdirichlet") == 0) {
      key = "udfdirichlet";
    }

    if (keyIn.compare("p") == 0) {
      key = "periodic";
    }

    if (keyIn.compare("w") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("wall") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("inlet") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("v") == 0) {
      key = "zerodirichlet";
    }

    if (key.compare("int") == 0) {
      key = "zerodirichlet";
    }
    if (key.compare("interpolation") == 0) {
      key = "zerodirichlet";
    }

    if (keyIn.compare("mv") == 0) {
      key = "udfdirichlet";
    }
    if (keyIn.compare("udfdirichlet+moving") == 0) {
      key = "udfdirichlet";
    }

    nekrsCheck(vBcTextToID.find(key) == vBcTextToID.end(),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "Invalid bcType (%s)\n",
               key.c_str());

    bToBc[make_pair(field, bid)] = vBcTextToID.at(key);
  }
}

bool nrs_t::bdry::useDerivedGeomBoundaryConditions()
{
  if (importFromNek) {
    return true;
  } else {
    return geomConditionsDerived;
  }
}
