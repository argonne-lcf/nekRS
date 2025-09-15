void parseCvodeSolver(const int rank, setupAide &options, inipp::Ini *ini)
{
#ifndef ENABLE_CVODE
  append_error("ERROR: CVODE not enabled! Recompile with -DENABLE_CVODE=ON\n");
  return;
#endif

  // default values
  double relativeTol;
  double absoluteTol;
  double hmax;
  int maxSteps = 500;
  double epsLin;
  double sigScale;
  bool recycleProps;
  bool mixedPrecisionJtv;

  std::string integrator = "bdf";

  const std::string parScope = "cvode";

  if (ini->extract(parScope, "relativetol", relativeTol)) {
    options.setArgs("CVODE RELATIVE TOLERANCE", to_string_f(relativeTol));
  }

  options.setArgs("CVODE GMRES BASIS VECTORS", "10");
  options.setArgs("CVODE SOLVER", "CBGMRES");

  // parse cvode linear solver
  [&]() {
    std::string p_solver;

    if (!ini->extract("cvode", "solver", p_solver)) {
      return;
    }

    const std::vector<std::string> validValues = {
        {"gmres"},
        {"cbgmres"},
        {"nvector"},
    };

    std::vector<std::string> list = serializeString(p_solver, '+');
    for (const std::string s : list) {
      checkValidity(rank, validValues, s);
    }

    if (p_solver.find("gmres") != std::string::npos) {
      std::vector<std::string> list;
      list = serializeString(p_solver, '+');

      std::string n = "10";

      for (std::string s : list) {
        const auto nvectorStr = parseValueForKey(s, "nvector");
        if (!nvectorStr.empty()) {
          n = nvectorStr;
        }
      }
      options.setArgs("CVODE GMRES BASIS VECTORS", n);

      if (p_solver.find("cb") != std::string::npos) {
        options.setArgs("CVODE SOLVER", "CBGMRES");
      } else {
        options.setArgs("CVODE SOLVER", "GMRES");
      }
    }
  }();

  if (options.compareArgs("VERBOSE", "TRUE")) {
    options.setArgs("CVODE VERBOSE", "TRUE");
  }

  options.setArgs("CVODE STOP TIME", "TRUE");

  if (ini->extract(parScope, "hmaxratio", hmax)) {
    options.setArgs("CVODE HMAX RATIO", std::to_string(hmax));
    options.setArgs("CVODE STOP TIME", "FALSE");
  }

  if (ini->extract(parScope, "epslin", epsLin)) {
    options.setArgs("CVODE EPS LIN", std::to_string(epsLin));
  }

  if (ini->extract(parScope, "maxSteps", maxSteps)) {
    options.setArgs("CVODE MAX STEPS", std::to_string(maxSteps));
  }

  int maxOrder;
  if (ini->extract(parScope, "maxOrder", maxOrder)) {
    options.setArgs("CVODE MAX TIMESTEPPER ORDER", std::to_string(maxOrder));
  }

  options.setArgs("CVODE GS TYPE", "CLASSICAL");
  std::string gstype;
  if (ini->extract(parScope, "gstype", gstype)) {
    if (gstype == "classical") {
      options.setArgs("CVODE GS TYPE", "CLASSICAL");
    } else if (gstype == "modified") {
      options.setArgs("CVODE GS TYPE", "MODIFIED");
    } else {
      append_error("Invalid gsType for " + parScope);
    }
  }

  options.setArgs("CVODE INTEGRATOR", upperCase(integrator));

  if (ini->extract(parScope, "dqsigma", sigScale)) {
    options.setArgs("CVODE DQ SIGMA", to_string_f(sigScale));
  }

  bool dealiasing;
  ini->extract(parScope, "dealiasing", dealiasing);
  if (dealiasing && !options.compareArgs("OVERINTEGRATION", "TRUE")) {
    append_error("dealiasing for CVODE only is not supported!" + parScope);
  }

  std::string recyclePropsStr;
  if (ini->extract(parScope, "jtvrecycleproperties", recyclePropsStr)) {
    recycleProps = checkForTrue(recyclePropsStr);
    if (recycleProps) {
      options.setArgs("CVODE JTV RECYCLE PROPERTIES", "TRUE");
    } else {
      options.setArgs("CVODE JTV RECYCLE PROPERTIES", "FALSE");
    }
  }

  options.setArgs("CVODE SHARED RHO", "FALSE");
  std::string sharedRhoStr;
  if (ini->extract(parScope, "sharedrho", sharedRhoStr)) {
    bool sharedRho = checkForTrue(sharedRhoStr);
    if (sharedRho) {
      options.setArgs("CVODE SHARED RHO", "TRUE");
    } else {
      options.setArgs("CVODE SHARED RHO", "FALSE");
    }
  }

  std::string mixedPrecisionStr;
  if (ini->extract(parScope, "jtvmixedprecision", mixedPrecisionStr)) {
    mixedPrecisionJtv = checkForTrue(mixedPrecisionStr);
    if (mixedPrecisionJtv) {
      options.setArgs("CVODE MIXED PRECISION JTV", "TRUE");
    } else {
      options.setArgs("CVODE MIXED PRECISION JTV", "FALSE");
    }
  }

  if (mixedPrecisionJtv) {
    append_error("cvode::jtvmixedprecision is not supported yet!\n");
  }
}

