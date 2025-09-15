cmake_minimum_required(VERSION 3.21)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

set (CMAKE_CXX_STANDARD 17)

set(NEKRS_USE_DFLOAT_FLOAT OFF CACHE BOOL "use dfloat = float")

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED)

set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

#set(CMAKE_VERBOSE_MAKEFILE on)

set(INCLUDE_DIRS
  ${CASE_DIR}
  ${GSDIR}/gslib/include
  ${NEKRS_INSTALL_DIR}/occa/include
  ${NEKRS_INSTALL_DIR}/include
  ${NEKRS_INSTALL_DIR}/include/platform
  ${NEKRS_INSTALL_DIR}/include/platform/ogs
  ${NEKRS_INSTALL_DIR}/include/platform/linAlg
  ${NEKRS_INSTALL_DIR}/include/platform/utils
  ${NEKRS_INSTALL_DIR}/include/core/mesh
  ${NEKRS_INSTALL_DIR}/include/core
  ${NEKRS_INSTALL_DIR}/include/core/iofld
  ${NEKRS_INSTALL_DIR}/include/core/bdry
  ${NEKRS_INSTALL_DIR}/include/core/udf
  ${NEKRS_INSTALL_DIR}/include/core/plugins
  ${NEKRS_INSTALL_DIR}/include/core/linearSolver
  ${NEKRS_INSTALL_DIR}/include/core/nekInterface
  ${NEKRS_INSTALL_DIR}/include/core/neknek
  ${NEKRS_INSTALL_DIR}/include/core/elliptic
  ${NEKRS_INSTALL_DIR}/include/core/pointInterpolation
  ${NEKRS_INSTALL_DIR}/include/core/pointInterpolation/findpts
  ${NEKRS_INSTALL_DIR}/include/solver
  ${NEKRS_INSTALL_DIR}/include/solver/fluid
  ${NEKRS_INSTALL_DIR}/include/solver/geom
  ${NEKRS_INSTALL_DIR}/include/solver/scalar
  ${NEKRS_INSTALL_DIR}/include/solver/scalar/cvode
  ${NEKRS_INSTALL_DIR}/include/solver/scalar/regularization
  ${NEKRS_INSTALL_DIR}/include/app
  ${NEKRS_INSTALL_DIR}/include/app/nrs
  ${NEKRS_INSTALL_DIR}/include/app/nrs/bdry
  ${NEKRS_INSTALL_DIR}/include/app/nrs/plugins
  ${NEKRS_INSTALL_DIR}/include/app/nrs/postProcessing
)

# dummy
add_library(OKL EXCLUDE_FROM_ALL okl.cpp)
target_include_directories(OKL PRIVATE $ENV{NEKRS_UDF_INCLUDES} ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(OKL PRIVATE __okl__)

add_library(udf SHARED udf.cpp)
target_link_libraries(udf PRIVATE MPI::MPI_CXX)
target_include_directories(udf PRIVATE $ENV{NEKRS_UDF_INCLUDES} ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

if(cpptrace_FOUND)
  find_package(cpptrace QUIET PATHS ${NEKRS_INSTALL_DIR})
  target_link_libraries(udf PRIVATE cpptrace::cpptrace)
endif()

if(NEKRS_USE_DFLOAT_FLOAT)
  target_compile_definitions(udf PUBLIC NEKRS_USE_DFLOAT_FLOAT)
endif()

if(APPLE)
  target_link_options(udf PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()

foreach(LIB $ENV{NEKRS_UDF_LIBS})
 target_link_libraries(udf PRIVATE ${LIB}) 
endforeach()

#foreach(PATH $ENV{NEKRS_UDF_RPATH})
#  target_link_options(udf PRIVATE "LINKER:-rpath,${PATH}")
#endforeach()

if(DEFINED ENV{NEKRS_ASCENT_INSTALL_DIR})
  find_package(Ascent
               PATHS $ENV{NEKRS_ASCENT_INSTALL_DIR} NO_DEFAULT_PATH)
  if (ASCENT_FOUND)
    target_link_libraries(udf PRIVATE ascent::ascent_mpi)
    target_compile_definitions(udf PUBLIC NEKRS_ENABLE_ASCENT)
  endif()
endif()

if(EXISTS "${CASE_DIR}/udf.cmake")
  message("-- Found ${CASE_DIR}/udf.cmake")
  include(${CASE_DIR}/udf.cmake)
endif()
