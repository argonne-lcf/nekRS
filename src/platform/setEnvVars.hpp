#if !defined(nekrs_setenv_hpp_)
#define nekrs_setenv_hpp_ 

#include <iostream>
#include <string>
#include <cstdlib>
#include "inipp.hpp"

static void setEnvVars(const std::string& installDir, inipp::Ini ini)
{
  std::string buf;

  if (!getenv("NEKRS_CACHE_DIR")) {
    std::string dir = std::string(fs::current_path()) + "/.cache";
    setenv("NEKRS_CACHE_DIR", dir.c_str(), 1);
  }

  buf = installDir + "/kernels";
  if (!getenv("NEKRS_KERNEL_DIR")) {
    setenv("NEKRS_KERNEL_DIR", buf.c_str(), 1);
  }

  ini.extract("general", "nekrs_gpu_mpi", buf);
  if (!getenv("NEKRS_GPU_MPI")) {
    setenv("NEKRS_GPU_MPI", buf.c_str(), 1);
  }

  ini.extract("general", "cxx", buf);
  setenv("NEKRS_CXX", buf.c_str(), 1);

  ini.extract("general", "cxxflags", buf);
  setenv("NEKRS_CXXFLAGS", buf.c_str(), 1);

  ini.extract("general", "cc", buf);
  setenv("NEKRS_CC", buf.c_str(), 1);

  ini.extract("general", "cflags", buf);
  setenv("NEKRS_CFLAGS", buf.c_str(), 1);

  ini.extract("general", "fc", buf);
  setenv("NEKRS_FC", buf.c_str(), 1);

  ini.extract("general", "fflags", buf);
  if (!getenv("NEKRS_FFLAGS")) {
    if (!buf.empty()) {
      setenv("NEKRS_FFLAGS", buf.c_str(), 1);
    }
  }

  ini.extract("general", "nek5000_pplist", buf);
  setenv("NEKRS_NEK5000_PPLIST", buf.c_str(), 1);

  ini.extract("general", "nekrs_mpi_underlying_compiler", buf);
  if (!getenv("NEKRS_MPI_UNDERLYING_COMPILER")) {
    setenv("NEKRS_MPI_UNDERLYING_COMPILER", buf.c_str(), 1);
  }

  ini.extract("general", "nekrs_udf_includes", buf);
  if (getenv("NEKRS_UDF_INCLUDES")) {
    if (!buf.empty()) {
      buf = std::string(getenv("NEKRS_UDF_INCLUDES")) + ";" + buf;
    }
  }
  if (!buf.empty()) {
    setenv("NEKRS_UDF_INCLUDES", buf.c_str(), 1);
  }

  ini.extract("general", "nekrs_udf_rpath", buf);
  if (getenv("NEKRS_UDF_RPATH")) {
    if (!buf.empty()) {
      buf = std::string(getenv("NEKRS_UDF_RPATH")) + ";" + buf;
    }
  }
  if (!buf.empty()) {
    setenv("NEKRS_UDF_RPATH", buf.c_str(), 1);
  }

  ini.extract("general", "nekrs_udf_libs", buf);
  if (getenv("NEKRS_UDF_LIBS")) {
    if (!buf.empty()) {
      buf = std::string(getenv("NEKRS_UDF_LIBS")) + ";" + buf;
    }
  }
  if (!buf.empty()) {
    setenv("NEKRS_UDF_LIBS", buf.c_str(), 1);
  }

  ini.extract("general", "nekrs_adios2_install_dir", buf);
  if (!getenv("NEKRS_ADIOS2_INSTALL_DIR") && !buf.empty()) {
    setenv("NEKRS_ADIOS2_INSTALL_DIR", buf.c_str(), 1);
  }

  ini.extract("general", "occa_cxx", buf);
  if (!getenv("OCCA_CXX")) {
    setenv("OCCA_CXX", buf.c_str(), 1);
  }

  ini.extract("general", "occa_cxxflags", buf);
  if (!getenv("OCCA_CXXFLAGS")) {
    setenv("OCCA_CXXFLAGS", buf.c_str(), 1);
  }

  ini.extract("general", "occa_cuda_compiler_flags", buf);
  if (!getenv("OCCA_CUDA_COMPILER_FLAGS")) {
    setenv("OCCA_CUDA_COMPILER_FLAGS", buf.c_str(), 1);
  }

  ini.extract("general", "occa_hip_compiler_flags", buf);
  if (!getenv("OCCA_HIP_COMPILER_FLAGS")) {
    setenv("OCCA_HIP_COMPILER_FLAGS", buf.c_str(), 1);
  }

  ini.extract("general", "occa_dpcpp_compiler", buf);
  if (!getenv("OCCA_DPCPP_COMPILER")) {
    setenv("OCCA_DPCPP_COMPILER", buf.c_str(), 1);
  }

  ini.extract("general", "occa_dpcpp_compiler_flags", buf);
  if (!getenv("OCCA_DPCPP_COMPILER_FLAGS")) {
    setenv("OCCA_DPCPP_COMPILER_FLAGS", buf.c_str(), 1);
  }

  ini.extract("general", "occa_opencl_compiler_flags", buf);
  if (!getenv("OCCA_OPENCL_COMPILER_FLAGS")) {
    setenv("OCCA_OPENCL_COMPILER_FLAGS", buf.c_str(), 1);
  }

  ini.extract("general", "occa_mode_default", buf);
  if (!getenv("NEKRS_OCCA_MODE_DEFAULT")) {
    setenv("NEKRS_OCCA_MODE_DEFAULT", buf.c_str(), 1);
  }
}

#endif
