void parseCoarseGridDiscretization(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));
  std::string p_coarseGridDiscretization;
  const bool continueParsing = ini->extract(parScope, "coarsegriddiscretization", p_coarseGridDiscretization);
  if (!continueParsing) {
    return;
  }

  const std::vector<std::string> validValues = {
      {"semfem"},
      {"fem"},
      {"galerkin"},
  };

  const auto entries = serializeString(p_coarseGridDiscretization, '+');
  for (auto &&s : entries) {
    checkValidity(rank, validValues, s);
  }

  // exit early if not using multigrid as preconditioner
  if (!options.compareArgs(parSectionName + "PRECONDITIONER", "MULTIGRID")) {
    return;
  }

  if (p_coarseGridDiscretization.find("semfem") != std::string::npos) {
    options.setArgs(parSectionName + "MULTIGRID COARSE GRID DISCRETIZATION", "SEMFEM");
  } else if (p_coarseGridDiscretization.find("fem") != std::string::npos) {
    options.setArgs(parSectionName + "GALERKIN COARSE OPERATOR", "FALSE");
    if (p_coarseGridDiscretization.find("galerkin") != std::string::npos) {
      options.setArgs(parSectionName + "GALERKIN COARSE OPERATOR", "TRUE");
    }
  }
}

void parseCoarseSolver(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  std::string p_coarseSolver;

  const bool keyExist = ini->extract(parScope, "coarsesolver", p_coarseSolver) ||
                        ini->extract(parScope, "semfemsolver", p_coarseSolver);
  if (!keyExist) {
    if (parScope == "fluid pressure") {
      p_coarseSolver = "boomeramg";
    } else {
      p_coarseSolver = "smoother";
    }
  }

  const std::vector<std::string> validValues = {
      {"smoother"},
      {"jpcg"},
      {"boomeramg"},
//      {"amgx"},
      {"combined"},
      {"maxiter"},
      {"cpu"},
      {"device"},
      {"overlap"},
      {"residualtol"},
  };

  std::vector<std::string> entries = serializeString(p_coarseSolver, '+');
  for (std::string entry : entries) {
    checkValidity(rank, validValues, entry);
  }
  entries.erase(std::remove(entries.begin(), entries.end(), "boomeramg"), entries.end());

  const int smoother = p_coarseSolver.find("smoother") != std::string::npos;
  const int cg = p_coarseSolver.find("jpcg") != std::string::npos;

  const int amgx = p_coarseSolver.find("amgx") != std::string::npos;
  const int boomer = p_coarseSolver.find("boomeramg") != std::string::npos;
  if (amgx + boomer > 1) {
    append_error("Conflicting solver types in coarseSolver!\n");
  }

  if (boomer) {
    std::string smoother;
    options.getArgs(parSectionName + "MULTIGRID SMOOTHER", smoother);

    if ((smoother.find("DAMPEDJACOBI") != std::string::npos) && 
        options.compareArgs(parSectionName + "PRECONDITIONER", "MULTIGRID+SEMFEM")) {
      options.setArgs("BOOMERAMG ITERATIONS", "2");
    }
  }

  if (boomer || amgx) {
    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", "BOOMERAMG");
    if (amgx) {
      options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", "AMGX");
      if (!AMGXenabled()) {
        append_error("AMGX was requested but is not enabled!\n");
      }
    }

    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER PRECISION", "FP32");

    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "CPU");
    if (options.getArgs(parSectionName + "PRECONDITIONER") == "SEMFEM") {
      options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "DEVICE");
    }

    for (std::string entry : entries) {
      if (entry.find("smoother") != std::string::npos) {
        auto val = options.getArgs(parSectionName + "MULTIGRID COARSE SOLVER"); 
        options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", val + "+SMOOTHER");
      } else if (entry.find("cpu") != std::string::npos) {
        options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "CPU");
      } else if (entry.find("device") != std::string::npos) {
        options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "DEVICE");
      } else if (entry.find("overlap") != std::string::npos) {
        std::string currentSettings = options.getArgs(parSectionName + "MGSOLVER CYCLE");
        options.setArgs(parSectionName + "MGSOLVER CYCLE", currentSettings + "+OVERLAPCRS");
      } else {
        if (entry.find("boomeramg") != std::string::npos) {
          append_error("Invalid coarseGrid qualifier " + entry + "!\n");
        }
      }
    }
  } else if (cg) {
    const std::vector<std::string> validValues = {
#if 0 // not supported for now
        "smoother",
#endif
        {"jpcg"},
        {"maxiter"},
        {"residualtol"},
    };

    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", "JPCG");
    for (std::string entry : entries) {
      checkValidity(rank, validValues, entry);
      if (entry == "jpcg") continue;
      auto val = options.getArgs(parSectionName + "MULTIGRID COARSE SOLVER"); 
      options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", val + upperCase("+" + entry));
    }

    {
      auto val = options.getArgs(parSectionName + "MULTIGRID COARSE SOLVER"); 
      options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", val + upperCase("+COMBINED"));
    }

    options.removeArgs(parSectionName + "MULTIGRID COARSE SOLVER PRECISION");
    options.removeArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION");
  } else {
    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVER", "SMOOTHER");
    options.removeArgs(parSectionName + "MULTIGRID COARSE SOLVER PRECISION");
    options.removeArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION");
    if (options.compareArgs(parSectionName + "MGSOLVER CYCLE", "OVERLAPCRS")) {
      append_error("Overlap qualifier invalid if coarse solver is smoother!\n");
    }
  }

  if (amgx && options.compareArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "CPU")) {
    append_error("AMGX on CPU is not supported!\n");
  }

  if (boomer && options.compareArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "GPU")) {
    if (hypreWrapperDevice::enabled()) {
      append_error("HYPRE is not configured to run on the GPU!\n");
    }
  }

  const bool runSolverOnDevice = options.compareArgs(parSectionName + "MULTIGRID COARSE SOLVER LOCATION", "DEVICE");
  const bool overlapCrsSolve = options.compareArgs(parSectionName + "MGSOLVER CYCLE", "OVERLAPCRS");

  {
    auto val = p_coarseSolver;

    std::string pattern = "+overlap";
    if (auto pos = val.find(pattern); pos != std::string::npos) {
      val.erase(pos, pattern.length());
    }

    pattern = "+cpu";
    if (auto pos = val.find(pattern); pos != std::string::npos) {
      val.erase(pos, pattern.length());
    }

    if (overlapCrsSolve && val != "boomeramg") {
       append_error("Overlaping coarse grid solve is only supported with boomerAMG!\n");
    }
  }

  if (overlapCrsSolve && runSolverOnDevice) {
    append_error("Cannot overlap coarse grid solve when running coarse solver on the GPU!\n");
  }
}

std::vector<int> checkForIntInInputs(const std::vector<std::string> &inputs)
{
  std::vector<int> values;
  for (std::string s : inputs) {
    if (is_number(s)) {
      values.emplace_back(std::stoi(s));
    }
  }
  return values;
}

void parseSmoother(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string p_smoother;

  std::string parSection = upperCase(parPrefixFromParSection(parScope));

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID")) {
    options.setArgs(parSection + "MULTIGRID SMOOTHER", "FOURTHOPTCHEBYSHEV+DAMPEDJACOBI");
    options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", "1");
    options.setArgs(parSection + "MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", "1.1");
    if (parScope == "fluid pressure") {
      if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID+SEMFEM")) {
        options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", "2");
      } else {
        options.setArgs(parSection + "MULTIGRID SMOOTHER", "FOURTHOPTCHEBYSHEV+ASM");
        options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", "3");
      }
    }
  }

  if (!ini->extract(parScope, "smoothertype", p_smoother)) {
    return;
  }

  std::string p_preconditioner;
  ini->extract(parScope, "preconditioner", p_preconditioner);

  const std::vector<std::string> validValues = {
      {"asm"},
      {"ras"},
      {"cheby"},
      {"fourthcheby"},
      {"fourthoptcheby"},
      {"jac"},
      {"degree"},
      {"mineigenvalueboundfactor"},
      {"maxeigenvalueboundfactor"},
  };

  {
    const std::vector<std::string> list = serializeString(p_smoother, '+');
    for (const std::string s : list) {
      checkValidity(rank, validValues, s);
    }
  }

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID")) {
    std::vector<std::string> list;
    list = serializeString(p_smoother, '+');

    if (p_smoother.find("cheb") != std::string::npos) {
      bool surrogateSmootherSet = false;
      std::string chebyshevType = "";
      if (p_smoother.find("fourthopt") != std::string::npos) {
        chebyshevType = "FOURTHOPTCHEBYSHEV";
      } else if (p_smoother.find("fourth") != std::string::npos) {
        chebyshevType = "FOURTHCHEBYSHEV";
      } else {
        // using 1st-kind Chebyshev, so set a reasonable lmin multiplier
        chebyshevType = "CHEBYSHEV";
        options.setArgs(parSection + "MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", "0.1");
      }

      for (std::string s : list) {
        const auto degreeStr = parseValueForKey(s, "degree");
        if (!degreeStr.empty()) {
          options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", degreeStr);
        }

        const auto minEigBoundStr = parseValueForKey(s, "mineigenvalueboundfactor");
        if (!minEigBoundStr.empty()) {
          if (chebyshevType.find("FOURTH") != std::string::npos) {
            append_error(
                "minEigenvalueBoundFactor not supported for 4th kind or Opt. 4th kind Chebyshev smoother!\n");
          }
          options.setArgs(parSection + "MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minEigBoundStr);
        }

        const auto maxEigBoundStr = parseValueForKey(s, "maxeigenvalueboundfactor");
        if (!maxEigBoundStr.empty()) {
          options.setArgs(parSection + "MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxEigBoundStr);
        }

        if (s.find("jac") != std::string::npos) {
          surrogateSmootherSet = true;
          options.setArgs(parSection + "MULTIGRID SMOOTHER", chebyshevType + "+DAMPEDJACOBI");
        } else if (s.find("asm") != std::string::npos) {
          surrogateSmootherSet = true;
          options.setArgs(parSection + "MULTIGRID SMOOTHER", chebyshevType + "+ASM");
        } else if (s.find("ras") != std::string::npos) {
          surrogateSmootherSet = true;
          options.setArgs(parSection + "MULTIGRID SMOOTHER", chebyshevType + "+RAS");
        }
      }

      if (!surrogateSmootherSet) {
        append_error("Inner Chebyshev smoother not set");
      }
      return;
    }

    // Non-Chebyshev smoothers
    options.removeArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE");
    options.removeArgs(parSection + "MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR");
    if (p_smoother.find("asm") == 0) {
      options.setArgs(parSection + "MULTIGRID SMOOTHER", "ASM");
    } else if (p_smoother.find("ras") == 0) {
      options.setArgs(parSection + "MULTIGRID SMOOTHER", "RAS");
    } else if (p_smoother.find("jac") == 0) {
      append_error("Jacobi smoother requires Chebyshev");
      options.setArgs(parSection + "MULTIGRID SMOOTHER", "DAMPEDJACOBI");
    } else {
      append_error("Unknown ::smootherType");
    }
  }
}

void parsePreconditioner(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  const std::vector<std::string> validValues = {
      {"none"},
      {"jac"},
      {"semfem"},
      {"femsem"},
      {"pmg"},
      {"multigrid"},
      {"additive"},
      {"multiplicative"},
  };

  std::string parSection = upperCase(parPrefixFromParSection(parScope));

  std::string p_preconditioner;
  if (!ini->extract(parScope, "preconditioner", p_preconditioner)) {
    p_preconditioner = "jac";
    if (parScope.find("fluid pressure") != std::string::npos) {
      p_preconditioner = "multigrid";
    }
  }

  const std::vector<std::string> list = serializeString(p_preconditioner, '+');
  for (std::string s : list) {
    checkValidity(rank, validValues, s);
  }

  const auto mg = p_preconditioner.find("pmg") != std::string::npos ||
                  p_preconditioner.find("multigrid") != std::string::npos;

  const auto semfem = p_preconditioner.find("semfem") != std::string::npos ||
                      p_preconditioner.find("femsem") != std::string::npos;
 
  if (p_preconditioner.find("none") != std::string::npos) {
    options.setArgs(parSection + "PRECONDITIONER", "NONE");
    return;
  } else if (p_preconditioner.find("jac") != std::string::npos) {
    options.setArgs(parSection + "PRECONDITIONER", "JACOBI");
    options.setArgs(parSection + "ELLIPTIC PRECO COEFF FIELD", "TRUE");
  } else if (mg) {
    options.setArgs(parSection + "PRECONDITIONER", "MULTIGRID");
    options.setArgs(parSection + "ELLIPTIC PRECO COEFF FIELD", "FALSE");
    std::string key = "VCYCLE+MULTIPLICATIVE";
    if (p_preconditioner.find("additive") != std::string::npos) {
      key = "VCYCLE+ADDITIVE";
    } else if (p_preconditioner.find("multiplicative") != std::string::npos) {
      key = "VCYCLE+MULTIPLICATIVE";
    }
    options.setArgs(parSection + "MGSOLVER CYCLE", key);

    if (semfem) {
      options.setArgs(parSection + "PRECONDITIONER", "MULTIGRID+SEMFEM");
      options.setArgs(parSection + "MGSOLVER CYCLE", "VCYCLE+MULTIPLICATIVE");
      options.setArgs(parSection + "ELLIPTIC PRECO COEFF FIELD", "FALSE");
    }
  } else if (p_preconditioner.find("semfem") != std::string::npos ||
             p_preconditioner.find("femsem") != std::string::npos) {
    options.setArgs(parSection + "PRECONDITIONER", "SEMFEM");
#if 0
    std::string p_coarseGridDiscretization;
    if (ini->extract(parScope, "coarsegriddiscretization", p_coarseGridDiscretization)) {
      if (p_coarseGridDiscretization.find("semfem") != std::string::npos) {
        smoothed = false;
      }
    }
#endif
  }

  parseSmoother(rank, options, ini, parScope);

  parseCoarseGridDiscretization(rank, options, ini, parScope);

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID") ||
      options.compareArgs(parSection + "PRECONDITIONER", "SEMFEM")) {
    parseCoarseSolver(rank, options, ini, parScope);
  }

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID")) {
    std::string p_mgschedule;
    if (ini->extract(parScope, "pmgschedule", p_mgschedule)) {
      const auto semfem = options.getArgs(parSection + "PRECONDITIONER") == "SEMFEM";
      if (semfem) {
        append_error("pMGSchedule not supported for preconditioner = semfem.\n");
      }

      options.setArgs(parSection + "MULTIGRID SCHEDULE", p_mgschedule);

      options.removeArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE");

      // validate multigrid schedule
      // note: default order here is not actually required
      auto [scheduleMap, errorString] = ellipticParseMultigridSchedule(p_mgschedule, options, 3);
      if (!errorString.empty()) {
        append_error(errorString);
      }

      int minDegree = std::numeric_limits<int>::max();
      for (auto &&[cyclePosition, smootherOrder] : scheduleMap) {
        auto [polyOrder, isDownLeg] = cyclePosition;
        minDegree = std::min(minDegree, polyOrder);
      }

      const auto INVALID = std::numeric_limits<int>::lowest();

      // bail if degree is set _and_ it conflicts
      std::string p_smoother;
      if (ini->extract(parScope, "smoothertype", p_smoother)) {
        for (auto &&s : serializeString(p_smoother, '+')) {
          if (s.find("degree") != std::string::npos) {
            const auto degreeStr = parseValueForKey(s, "degree");
            if (!degreeStr.empty()) {
              const auto specifiedDegree = std::stoi(degreeStr);
              for (auto &&[cyclePosition, smootherOrder] : scheduleMap) {
                auto [polyOrder, isDownLeg] = cyclePosition;
                const bool degreeConflicts = smootherOrder != specifiedDegree;
                const bool isMinOrder = polyOrder == minDegree;
                const bool minOrderInvalid = smootherOrder == INVALID;

                if (isMinOrder && minOrderInvalid) {
                  continue;
                }

                if (degreeConflicts) {
                  append_error(
                      "order specified in pMGSchedule conflicts with that specified in smootherType!\n");
                }
              }
            }
          }
        }
      }

      // bail if coarse degree is set, but we're not smoothing on the coarsest level
      if (scheduleMap[{minDegree, true}] > 0) {

        if (!options.compareArgs(parSection + "MULTIGRID COARSE SOLVER", "SMOOTHER")) {
          append_error("specified coarse Chebyshev degree, but coarseSolver=smoother is not set.\n");
        }
      }
    }
  }
}


