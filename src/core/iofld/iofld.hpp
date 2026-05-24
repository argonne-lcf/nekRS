#if !defined(nekrs_iofld_hpp_)
#define nekrs_iofld_hpp_

#include <variant>
#include <functional>
#include <optional>

#include "platform.hpp"
#include "mesh3D.h"

class iofld
{
protected:
  // shared global map to keep track of the step count
  static std::map<std::string, int> stepCounter;

public:
  using variantType = std::variant<std::reference_wrapper<int>,
                                   std::reference_wrapper<long long int>,
                                   std::reference_wrapper<float>,
                                   std::reference_wrapper<double>>;

  enum class mode { read, write };

  virtual ~iofld()
  {
    if (mesh_vis != mesh) {
      meshFree(mesh_vis);
    }
  };

  void open(mesh_t *mesh_, mode mode_, const std::string &fileNameBase_, int step_ = -1)
  {
    mesh = mesh_;
    mesh_vis = mesh;
    mesh_hrefine = mesh;

    engineMode = mode_;

    folderName = "";
    platform->options.getArgs("CHECKPOINT DIRECTORY", folderName);
    if (!folderName.empty()) {
      folderName += "/";
      if (!fs::is_directory(folderName)) {
        if (platform->comm.mpiRank() == 0) {
          std::cout << " directory: " << folderName << " does not exist, reset to ./" << std::endl << std::flush;
        }
        folderName = "";
      }
    }

    fileNameBase = fileNameBase_;
    step = step_;

    nekrsCheck(engineMode != iofld::mode::read && engineMode != iofld::mode::write,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s\n",
               "invalid iofld::mode!");

    openEngine();

    initialized = true;
  };

  virtual void openEngine() = 0;

  int getStepCounter() const
  {
    return stepCounter[fileNameBase];
  };

  int N = 0;
  bool uniform = false;
  int precision = 32;
  bool outputMesh = false;
  bool redistribute = true;
  bool pointInterpolation = false;

  std::vector<int> hRefineScheduleSim = {};
  std::vector<int> hRefineScheduleFld = {};
  std::vector<int> hRefineScheduleApply = {};
  dlong Nelements;

  int hRefineScale(std::vector<int> &schedule) {
    int scale = 1;
    for (int ncut : schedule) {
      scale *= std::pow(ncut, mesh->dim);
    }
    return scale;
  };

  void hRefineDiffSchedule(int NelementsGlobalSim, int NelementsGlobalFld)
  {
    hRefineScheduleApply.clear();
    if (hRefineScheduleSim.size()==0) return;

    auto printSchedule = [&](std::vector<int> schedule, std::string name) {
      if (platform->comm.mpiRank() == 0) {
        std::cout << name << " [";
        for (int i = 0; i < schedule.size(); i++) {
          if (i) std::cout << ", ";
          std::cout << schedule[i];
        }
        std::cout << "]\n";
      }
    };

    printSchedule(hRefineScheduleSim, " h-refine schedule, sim:");
    printSchedule(hRefineScheduleFld, " h-refine schedule, fld:");

    auto checkSchedule = [&]() {
      int NelementsBaseSim = NelementsGlobalSim / hRefineScale(hRefineScheduleSim);
      int NelementsBaseFld = NelementsGlobalFld / hRefineScale(hRefineScheduleFld);
      nekrsCheck(NelementsBaseSim != NelementsBaseFld,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "h-refine base meshes of Sim and Fld mismatched E= %d vs %d!\n",
                 NelementsBaseSim,
                 NelementsBaseFld);

      int ierr = (hRefineScheduleFld.size() > hRefineScheduleSim.size()) ? 1 : 0;
      for (int i = 0; i < hRefineScheduleFld.size(); i++) {
        if (hRefineScheduleFld[i] != hRefineScheduleSim[i]) {
          ierr = 1;
        }
      }
      nekrsCheck(ierr,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s\n",
                 "h-refine schedules mismatched: Fld must be a subset of Sim!");
    };

    checkSchedule();

    // generate diff schedule
    const int ndiff = hRefineScheduleSim.size() - hRefineScheduleFld.size();
    hRefineScheduleApply.resize(ndiff, 0);
    for (int i = 0; i < ndiff; i++) {
      hRefineScheduleApply[i] = hRefineScheduleSim[i + hRefineScheduleFld.size()];
    }

    printSchedule(hRefineScheduleApply, " h-refine schedule, dif:");
  };

  void writeAttribute(const std::string &key_, const std::string &val)
  {
    nekrsCheck(!initialized, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "illegal to call prior to iofld::open()!");

    std::string key = lowerCase(key_);

    if (key == "polynomialorder") {
      N = std::stoi(val);
      nekrsCheck(N < 1 || N > 15, MPI_COMM_SELF, EXIT_FAILURE, "invalid polynomial order %d\n", N);
    } else if (key == "precision") {
      precision = stoi(val);
      nekrsCheck(precision != 64 && precision != 32,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "invalid precision value %d\n",
                 precision);
    } else if (key.find("uniform") == 0 || key.find("equidistant") == 0) {
      uniform = (val == "true") ? true : false;
    } else if (key == "outputmesh") {
      outputMesh = (val == "true") ? true : false;
    } else if (key == "redistribute") {
      redistribute = (val == "true") ? true : false;
    } else if (key == "hschedule") {
      hRefineScheduleSim.clear();
      for (auto &&s : serializeString(val, ',')) {
        hRefineScheduleSim.push_back(std::stoi(s));
      }
    } else {
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "invalid attribute %s\n", key_.c_str());
    }
  };

  void readAttribute(const std::string &key_, const std::string &val)
  {
    nekrsCheck(!initialized, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "illegal to call prior to iofld::open()!");

    std::string key = lowerCase(key_);

    if (key == "interpolate") {
      pointInterpolation = (val == "true") ? true : false;
      if (pointInterpolation) {
        redistribute = false;
      }
    } else if (key == "hschedule") {
      hRefineScheduleSim.clear();
      for (auto &&s : serializeString(val, ',')) {
        hRefineScheduleSim.push_back(std::stoi(s));
      }
    } else {
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "invalid attribute %s\n", key_.c_str());
    }
  }

  std::vector<int> elementMask;

  void writeElementFilter(const std::vector<int> &elementMask_)
  {
    elementMask = elementMask_;
  };

  void process()
  {
    const auto tStart = MPI_Wtime();

    if (platform->comm.mpiRank() == 0) {
      if (engineMode == iofld::mode::write) {
        std::cout << "writing to field file ..." << std::endl << std::flush;
      }
    }

    nekrsCheck(!initialized, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "illegal to call prior to iofld::open()!");

    if (platform->comm.mpiRank() == 0) {
      std::cout << " user variables: ";
      for (const auto &entry : userSingleValues) {
        std::cout << entry.first << " ";
      }
      for (const auto &entry : userFields) {
        std::cout << entry.first << " ";
      }
      std::cout << std::endl;
    }

    size_t bytes;
    if (engineMode == iofld::mode::read) {
      nekrsCheck(elementMask.size(),
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s\n",
                 "element filter for iofld::mode::read is not supported yet!");
      bytes = read();
    }

    if (engineMode == iofld::mode::write) {
      if (platform->comm.mpiRank() == 0) {
        std::cout << " io step: " << getStepCounter() << std::endl;
        std::cout << " settings: N=" << N << "  precision=" << precision << "  uniform=" << uniform
                  << std::endl
                  << std::flush;
      }
      platform->timer.tic("checkpointing");
      bytes = write();
      addStep();
      platform->timer.toc("checkpointing");
    }

    MPI_Barrier(platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      const auto elapsed = MPI_Wtime() - tStart;
      std::cout << " elapsed time: " << elapsed << "s";
      if (bytes) {
        std::cout << "  (" << bytes / elapsed / 1e9 << " GB/s)";
      }
      std::cout << std::endl;
    }
  };

  void addVariable(const std::string &name, occa::memory u)
  {
    addVariable(name, std::vector<occa::memory>{u});
  };

  virtual void validateUserFields(const std::string &name) = 0;

  template <typename T> void addVariable(const std::string &name, std::vector<deviceMemory<T>> u)
  {
    std::vector<occa::memory> u_;
    for (const auto &entry : u) {
      u_.push_back(entry);
    }
    addVariable(name, std::vector<occa::memory>{u_});
  };

  void addVariable(const std::string &name, std::vector<occa::memory> u)
  {
    validateUserFields(name);

    for (const auto &v : u) {
      nekrsCheck(v.dtype() != occa::dtype::get<float>() && v.dtype() != occa::dtype::get<double>(),
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s!\n",
                 "Invalid typed occa::memory");
      nekrsCheck(v.size() > mesh->Nlocal,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "Size of variable %s is larger than mesh!\n",
                 name.c_str());
    }

    userFields.insert_or_assign(name, u);
  };

  virtual void validateUserSingleValues(const std::string &name) = 0;

  template <typename T> void addVariable(const std::string &name, T &u)
  {
    static_assert(!std::is_same<T, occa::memory>::value, "Type occa::memory& not supported!");

    validateUserSingleValues(name);

    userSingleValues.insert_or_assign(name, std::ref(u));
    // add error hanling for unspported data types!
  };

  std::vector<std::string> _availableVariables;

  std::vector<std::string> availableVariables() const
  {
    return _availableVariables;
  };

  bool initialized = false;

  bool isInitialized() const
  {
    return initialized;
  };

  mode engineMode;

  std::string folderName;
  std::string fileNameBase;

  int step = 0;

  std::map<std::string, std::vector<occa::memory>> userFields;
  std::map<std::string, variantType> userSingleValues;

  mesh_t *mesh;
  mesh_t *mesh_vis;
  mesh_t *mesh_hrefine;

  mesh_t *genVisMesh(dlong Nelements_ = 0, int N_ = 0, bool isVis = true)
  {
    if (isVis) {
      if (mesh_vis != mesh) {
        meshFree(mesh_vis);
      }
    } else if (mesh_hrefine != mesh) {
      meshFree(mesh_hrefine);
    }

    auto p = (N_ > 0) ? N_ : mesh->N;
    auto E = (Nelements_ > 0) ? Nelements_ : mesh->Nelements;
    if (platform->comm.mpiRank() == 0 && platform->verbose()) {
      std::cout << " gernerating vis mesh with N=" << p << " E=" << E << std::endl;
    }

    hlong NelementsGlobal = E;
    MPI_Allreduce(MPI_IN_PLACE, &NelementsGlobal, 1, MPI_INT, MPI_SUM, platform->comm.mpiComm());

    auto meshNew = new mesh_t();
    meshNew->Nelements = E;
    meshNew->NelementsGlobal = NelementsGlobal;
    meshNew->dim = mesh->dim;
    meshNew->Nverts = mesh->Nverts;
    meshNew->Nfaces = mesh->Nfaces;
    meshNew->NfaceVertices = mesh->NfaceVertices;
    meshLoadReferenceNodesHex3D(meshNew, p, 0);

    auto intpKernel = [&]() {
      const std::string ext = platform->serial() ? ".c" : ".okl";
      std::string kernelName;
      std::string orderSuffix;
      if (p > mesh->N) {
        kernelName = "coarsenHex3D";
        if (engineMode == iofld::mode::write) {
          kernelName = "prolongateHex3D";
        }
        orderSuffix = std::string("_Nf_") + std::to_string(p) + std::string("_Nc_") + std::to_string(mesh->N);
      } else {
        kernelName = "prolongateHex3D";
        if (engineMode == iofld::mode::write) {
          kernelName = "coarsenHex3D";
        }
        orderSuffix = std::string("_Nf_") + std::to_string(mesh->N) + std::string("_Nc_") + std::to_string(p);
      }
      return platform->kernelRequests.load("mesh-" + kernelName + orderSuffix);
    };
    meshNew->intpKernel[mesh->N] = intpKernel();

    meshNew->hRefineIntpKernel = mesh->hRefineIntpKernel;

    return meshNew;
  };

  template <typename T = std::vector<occa::memory>>
  std::optional<std::reference_wrapper<T>> inquireVariable(const std::string &name)
  {
    nekrsCheck(!initialized, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "illegal to call prior to iofld::open()!");

    if (userFields.count(name) == 0 && userSingleValues.count(name) == 0) {
      return std::nullopt;
    }

    if constexpr (std::is_same_v<T, std::vector<occa::memory>>) {
      return userFields.at(name);
    } else {
      auto visitor = [](const auto &value) -> variantType {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::reference_wrapper<int>> ||
                      std::is_same_v<std::decay_t<decltype(value)>, std::reference_wrapper<long long int>> ||
                      std::is_same_v<std::decay_t<decltype(value)>, std::reference_wrapper<float>> ||
                      std::is_same_v<std::decay_t<decltype(value)>, std::reference_wrapper<double>>) {
          return value;
        } else {
          throw std::bad_variant_access(); // Mismatched type
        }
      };

      // retrieve the reference from the variant, unwrap it
      return std::get<std::reference_wrapper<T>>(std::visit(visitor, userSingleValues.at(name))).get();
    }
  };

  virtual void close() = 0;

  virtual size_t read() = 0;
  virtual size_t write() = 0;

private:
  void addStep()
  {
    stepCounter[fileNameBase]++;
  };
};

#endif
