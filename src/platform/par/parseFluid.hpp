void parsePressureSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  const std::string parScope = "fluid pressure";

  options.setArgs("FLUID PRESSURE ELLIPTIC COEFF FIELD", "FALSE");

  options.setArgs("FLUID PRESSURE HELMHOLTZ TYPE", "POISSON");

  parseSolverTolerance(rank, options, ini, parScope);

  parseInitialGuess(rank, options, ini, parScope);

  parsePreconditioner(rank, options, ini, parScope);

  parseLinearSolver(rank, options, ini, parScope);

  parseBoomerAmgSection(rank, options, ini);

  if (ini->sections.count("amgx")) {
    if (!AMGXenabled()) {
      append_error("AMGX was requested but is not compiled!\n");
    }
    std::string configFile;
    if (ini->extract("amgx", "configfile", configFile)) {
      options.setArgs("AMGX CONFIG FILE", configFile);
    }
  }
}

void parseVelocitySection(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::string vsolver;
  std::string sbuf;

  options.setArgs("FLUID VELOCITY ELLIPTIC COEFF FIELD", "TRUE");
  if (options.getArgs("FLUID STRESSFORMULATION").empty()) {
    options.setArgs("FLUID STRESSFORMULATION", "FALSE");
  }

  const std::string parScope = "fluid velocity";

  parseCheckpointing(rank, options, ini, "fluid");

  parseInitialGuess(rank, options, ini, parScope);

  parsePreconditioner(rank, options, ini, parScope);

  parseLinearSolver(rank, options, ini, parScope);

  parseSolverTolerance(rank, options, ini, parScope);

  std::string v_bcMap;
  if (ini->extract(parScope, "boundarytypemap", v_bcMap)) {
    options.setArgs("FLUID VELOCITY BOUNDARY TYPE MAP", v_bcMap);
  }

  double rho;
  if (ini->extract(parScope, "density", rho) || ini->extract(parScope, "rho", rho)) {
    options.setArgs("FLUID DENSITY", to_string_f(rho));
  }

  if (ini->extract(parScope, "viscosity", sbuf) || ini->extract(parScope, "mu", sbuf)) {
    int err = 0;
    double viscosity = parseFormula(sbuf.c_str(), &err);
    if (err) {
      append_error("Invalid expression for viscosity");
    }
    if (viscosity < 0) {
      viscosity = fabs(1 / viscosity);
    }
    options.setArgs("FLUID VISCOSITY", to_string_f(viscosity));
  }

  parseRegularization(rank, options, ini, parScope);
}

void parseConstFlowRate(const int rank, setupAide &options, inipp::Ini *ini)
{
  const std::vector<std::string> validValues = {
      {"constflowrate"},
      {"meanvelocity"},
      {"meanvolumetricflow"},
      {"bid"},
      {"direction"},
  };

  options.setArgs("CONSTANT FLOW RATE", "FALSE");

  std::string flowRateDescription;
  if (ini->extract("general", "constflowrate", flowRateDescription)) {
    options.setArgs("CONSTANT FLOW RATE", "TRUE");
    bool flowRateSet = false;
    bool flowDirectionSet = false;
    bool issueError = false;
    const std::vector<std::string> list = serializeString(flowRateDescription, '+');
    for (std::string s : list) {
      checkValidity(rank, validValues, s);

      const auto meanVelocityStr = parseValueForKey(s, "meanvelocity");
      if (!meanVelocityStr.empty()) {
        flowRateSet = true;
        options.setArgs("FLOW RATE", meanVelocityStr);
        options.setArgs("CONSTANT FLOW RATE TYPE", "BULK");
      }

      const auto meanVolumetricFlowStr = parseValueForKey(s, "meanvolumetricflow");
      if (!meanVolumetricFlowStr.empty()) {
        flowRateSet = true;
        options.setArgs("FLOW RATE", meanVolumetricFlowStr);
        options.setArgs("CONSTANT FLOW RATE TYPE", "VOLUMETRIC");
      }

      if (s.find("bid") == 0) {
        if (flowDirectionSet) {
          issueError = true;
        }
        flowDirectionSet = true;
        std::vector<std::string> items = serializeString(s, '=');

        std::string bidStr;
        if (items.size() == 2) {
          bidStr = items[1];
        } else {
          std::ostringstream error;
          error << "could not parse " << s << "!\n";
          append_error(error.str());
        }
        std::vector<std::string> bids = serializeString(items[1], ',');
        if (bids.size() == 2) {
          const int fromBID = std::stoi(bids[0]);
          const int toBID = std::stoi(bids[1]);
          options.setArgs("CONSTANT FLOW FROM BID", std::to_string(fromBID));
          options.setArgs("CONSTANT FLOW TO BID", std::to_string(toBID));
        } else {
          std::ostringstream error;
          error << "could not parse " << s << "!\n";
          append_error(error.str());
        }

        append_error(
            "Specifying a constant flow direction with a pair of BIDs is currently not supported.\n");
      }

      if (s.find("direction") == 0) {
        if (flowDirectionSet) {
          issueError = true;
        }
        flowDirectionSet = true;
        std::vector<std::string> items = serializeString(s, '=');
        if (items.size() == 2) {
          std::string direction = items[1];
          issueError = (direction.find("x") == std::string::npos &&
                        direction.find("y") == std::string::npos && direction.find("z") == std::string::npos);
          options.setArgs("CONSTANT FLOW DIRECTION", upperCase(direction));
        } else {
          std::ostringstream error;
          error << "could not parse " << s << "!\n";
          append_error(error.str());
        }
      }
    }
    if (!flowDirectionSet) {
      append_error("Flow direction has not been set in GENERAL:constFlowRate!\n");
    }
    if (!flowRateSet) {
      append_error("Flow rate has not been set in GENERAL:constFlowRate!\n");
    }
    if (issueError) {
      append_error("Error parsing GENERAL:constFlowRate!\n");
    }
  }
}

