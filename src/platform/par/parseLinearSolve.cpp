void parseSolverTolerance(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{

  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  const std::vector<std::string> validValues = {
      {"relative"},
  };

  std::string residualTol;
  if (ini->extract(parScope, "residualtol", residualTol) ||
      ini->extract(parScope, "residualtolerance", residualTol)) {
    if (residualTol.find("relative") != std::string::npos) {
      options.setArgs(parSectionName + "LINEAR SOLVER STOPPING CRITERION", "RELATIVE");

      const auto subStr = residualTol.substr(residualTol.find("relative"));
      if (subStr != "relative") {
        auto relTolerance = std::strtod(parseValueForKey(subStr, "relative").c_str(), nullptr);
        if (relTolerance > 0) {
          options.setArgs(parSectionName + "SOLVER RELATIVE TOLERANCE", to_string_f(relTolerance));
        }
      }
    }

    std::vector<std::string> entries = serializeString(residualTol, '+');
    for (std::string entry : entries) {
      double tolerance = std::strtod(entry.c_str(), nullptr);
      if (tolerance > 0) {
        options.setArgs(parSectionName + "SOLVER TOLERANCE", entry);
      } else {
        checkValidity(rank, validValues, entry);
      }
    }
  }

  std::string absoluteTol;
  if (ini->extract(parScope, "absolutetol", absoluteTol)) {
    bool issueError = false;
    std::string solver;
    if (ini->extract(parScope, "solver", solver)) {
      issueError |= (solver != "cvode" && solver != "none");
    } else {
      solver = options.getArgs(parSectionName + "SOLVER");
      issueError |= !options.compareArgs(parSectionName + "SOLVER", "CVODE");
    }
    if (issueError) {
      append_error("absoluteTol is only supported for solver=cvode");
    }
    options.setArgs(parSectionName + "CVODE ABSOLUTE TOLERANCE", absoluteTol);
  }
}

void parseLinearSolver(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  std::string noop;
  bool applyDefault = (options.getArgs(parSectionName + "SOLVER", noop) == 0);

  if (applyDefault) {
    options.setArgs(parSectionName + "SOLVER", "CG");
    if (options.compareArgs(parSectionName + "PRECONDITIONER", "JACOBI")) {
#if 0
      options.setArgs(parSectionName + "SOLVER", "CG+COMBINED");
#else
      options.setArgs(parSectionName + "SOLVER", "CG");
#endif
    }

    if (parScope == "fluid pressure") {
      options.setArgs(parSectionName + "SOLVER", "GMRES+FLEXIBLE+NVECTOR=15");
      if (std::is_same<dfloat, float>::value) options.setArgs(parSectionName + "SOLVER", "GMRES+IR+FLEXIBLE+NVECTOR=5");
    }
    if (parScope == "fluid mesh") {
      options.setArgs(parSectionName + "SOLVER", "NONE");
    }

    if (parScope == "fluid velocity" || parScope == "geom") {
      options.setArgs(parSectionName + "SOLVER", "CG+BLOCK");
    }
  }

  std::string p_solver;
  if (!ini->extract(parScope, "solver", p_solver)) {
    return;
  }

  const std::vector<std::string> validValues = {
      {"user"},
      {"cvode"},
      {"none"},
      {"nvector"},
      {"flexible"},
      {"gmres"},
      {"pgmres"},
      {"cg"},
      {"pcg"},
      {"combined"},
      {"block"},
      {"ir"},
      {"maxiter"},
  };
  std::vector<std::string> list = serializeString(p_solver, '+');
  for (const std::string s : list) {
    checkValidity(rank, validValues, s);
  }
  if (p_solver.find("gmres") != std::string::npos) {
    //
  } else if (p_solver.find("cg") != std::string::npos) {
    if (p_solver.find("ir") != std::string::npos) {
        std::ostringstream ss;
        ss << "CG solver not support iterative refinement!\n";
        append_value_error(ss.str());
    } else if (p_solver.find("flexible") != std::string::npos) {
      if (p_solver.find("combined") != std::string::npos) {
        std::ostringstream ss;
        ss << "combined CG solver not support flexible!\n";
        append_value_error(ss.str());
      }
    } else  if (p_solver.find("combined") != std::string::npos) {
      if (!options.compareArgs(parSectionName + "PRECONDITIONER", "JACOBI")) {
        std::ostringstream ss;
        ss << "combined CG solver only supports Jacobi preconditioner!\n";
        append_value_error(ss.str());
      }
    }
  } else if (p_solver.find("user") != std::string::npos) {
    p_solver = "USER";
  } else if (p_solver.find("cvode") != std::string::npos) {
    p_solver = "CVODE";
  } else if (p_solver.find("none") != std::string::npos) {
    p_solver = "NONE";
  } else {
    append_error("Invalid solver for " + parScope);
  }
  options.setArgs(parSectionName + "SOLVER", upperCase(p_solver));
}

void parseInitialGuess(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  std::string initialGuess;

  const std::vector<std::string> validValues = {
      {"projectionaconj"},
      {"projection"},
      {"extrapolation"},
      {"previous"},
      // settings
      {"nvector"},
      {"start"},
  };

  options.setArgs(parSectionName + "INITIAL GUESS", "EXTRAPOLATION");
  if (parScope == "fluid pressure") {
    options.setArgs(parSectionName + "INITIAL GUESS", "PROJECTION-ACONJ");
  }

  if (ini->extract(parScope, "initialguess", initialGuess)) {
    if (initialGuess.find("extrapolation") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "EXTRAPOLATION");

      if (parScope == "fluid pressure") {
        append_error("initialGuess = extrapolation not supported for pressure!\n");
      }
      return;
    }

    const int defaultNumVectors = (parScope == "fluid pressure") ? 10 : 5;
    int proj = false;

    if (initialGuess.find("projectionaconj") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "PROJECTION-ACONJ");
      proj = true;
    } else if (initialGuess.find("projection") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "PROJECTION");
      proj = true;
    } else if (initialGuess.find("previous") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "PREVIOUS");
    } else {
      std::ostringstream error;
      error << "Could not parse initialGuess = " << initialGuess << "!\n";
      append_error(error.str());
    }

    if (proj) {
      options.setArgs(parSectionName + "RESIDUAL PROJECTION VECTORS", std::to_string(defaultNumVectors));
      options.setArgs(parSectionName + "RESIDUAL PROJECTION START", "5");
    }

    const std::vector<std::string> list = serializeString(initialGuess, '+');

    for (std::string s : list) {
      checkValidity(rank, validValues, s);

      const auto nVectorStr = parseValueForKey(s, "nvector");
      if (!nVectorStr.empty() && proj) {
        options.setArgs(parSectionName + "RESIDUAL PROJECTION VECTORS", nVectorStr);
      }

      const auto startStr = parseValueForKey(s, "start");
      if (!startStr.empty() && proj) {
        options.setArgs(parSectionName + "RESIDUAL PROJECTION START", startStr);
      }
    }
    return;
  }
}

