set(SOLVER_SOURCES_DIR "src/solver")

set(SOLVER_SOURCES
    ${SOLVER_SOURCES_DIR}/fluid/fluidSolver.cpp
    ${SOLVER_SOURCES_DIR}/geom/geomSolver.cpp
    ${SOLVER_SOURCES_DIR}/scalar/scalarSolver.cpp
    ${SOLVER_SOURCES_DIR}/scalar/cvode/registerCvodeKernels.cpp
    ${SOLVER_SOURCES_DIR}/scalar/cvode/cvode.cpp
    ${SOLVER_SOURCES_DIR}/scalar/cvode/cbGMRES.cpp
)

set(SOLVER_INCLUDE
    ${SOLVER_SOURCES_DIR}
    ${SOLVER_SOURCES_DIR}/fluid
    ${SOLVER_SOURCES_DIR}/geom
    ${SOLVER_SOURCES_DIR}/scalar
    ${SOLVER_SOURCES_DIR}/scalar/cvode
)
