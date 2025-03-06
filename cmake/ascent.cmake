add_definitions(-DASCENT_ENABLED)

set(ASCENT_SOURCES
    src/insitu/nekrsAscent.cpp
    src/insitu/simulationData.cpp
)

add_library(ascent-lib STATIC ${ASCENT_SOURCES})

###############################################################################
# ASCENT
###############################################################################

if(ENABLE_ASCENT)
  find_package(Ascent REQUIRED)
  if(${ASCENT_FOUND})
    set(ASCENT_ENABLED TRUE)

    #list(APPEND INSITU_SOURCES ascent.cpp)
    message(STATUS ${ASCENT_INSTALL_PREFIX})
    message(STATUS ${ASCENT_INCLUDE_DIRS})
    include_directories(${ASCENT_INCLUDE_DIRS} ${CONDUIT_DIR}/include/conduit)
    target_include_directories(ascent-lib 
      PRIVATE
      "$<TARGET_PROPERTY:nekrs-lib,INTERFACE_INCLUDE_DIRECTORIES>"
      ${ASCENT_INCLUDE_DIRS}
      ${CONDUIT_DIR}/include/conduit)
    
    #target_sources(ascent-lib PRIVATE src/insitu/inSituAscent.cpp src/insitu/inSituManager.cpp)
    target_link_libraries(ascent-lib PRIVATE ascent::ascent_mpi)
    #if(CMAKE_CUDA_COMPILER)
    #  set_property(TARGET ascent-lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    #endif()
  endif()
endif()

#install(TARGETS insitu DESTINATION lib)
