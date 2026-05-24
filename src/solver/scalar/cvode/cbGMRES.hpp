#ifndef cbGMRES_HPP
#define cbGMRES_HPP

#ifdef ENABLE_CVODE

#include "CvodeNVectorHelpers.hpp"

int cbGMRESSolve(SUNLinearSolver S, N_Vector x, N_Vector b, sunrealtype delta);
void cbGMRESSetup(SUNLinearSolver S);

#endif
#endif
