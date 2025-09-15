set(BENCH_SOURCES
        src/platform/kernelBenchmarker.cpp
        src/core/elliptic/MG/fdm/benchmark.cpp
        src/core/elliptic/axHelm/benchmark.cpp
        src/core/advsub/benchmark.cpp
)

set(BENCH_INCLUDE
    src/platform 
    src/core/elliptic/MG/fdm 
    src/core/elliptic/axHelm 
    src/core/advsub 
)

add_executable(axhelm-bin src/core/elliptic/axHelm/main.cpp)
set_target_properties(axhelm-bin PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs-bench-axhelm)
target_link_libraries(axhelm-bin PRIVATE nekrs-lib)
target_include_directories(axhelm-bin PRIVATE ${PLATFORM_INCLUDE})

add_executable(advsub-bin src/core/advsub/main.cpp)
set_target_properties(advsub-bin PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs-bench-advsub)
target_link_libraries(advsub-bin PRIVATE nekrs-lib)
target_include_directories(advsub-bin PRIVATE ${PLATFORM_INCLUDE})

add_executable(fdm-bin src/core/elliptic/MG/fdm/main.cpp)
set_target_properties(fdm-bin PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs-bench-fdm)
target_link_libraries(fdm-bin PRIVATE nekrs-lib)
target_include_directories(fdm-bin PRIVATE ${PLATFORM_INCLUDE})
