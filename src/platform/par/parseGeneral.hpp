void parseGeneralSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  // GENERAL
  bool verbose = false;
  if (ini->extract("general", "verbose", verbose)) {
    if (verbose) {
      options.setArgs("VERBOSE", "TRUE");
    }
  }

  std::string startFrom;
  if (ini->extract("general", "startfrom", startFrom)) {
    options.setArgs("RESTART FILE NAME", startFrom);
  }

  int N;
  if (ini->extract("general", "polynomialorder", N)) {
    options.setArgs("POLYNOMIAL DEGREE", std::to_string(N));
    if (N > 10) {
      append_error("polynomialOrder > 10 is currently not supported");
    }
  } else {
    append_error("cannot find mandatory parameter GENERAL::polynomialOrder");
  }

  // udf file
  {
    std::string udfFile;
    if (ini->extract("general", "udf", udfFile)) {
      options.setArgs("UDF FILE", udfFile);
    }
  }

  // usr file
  {
    std::string usrFile;
    if (ini->extract("general", "usr", usrFile)) {
      options.setArgs("NEK USR FILE", usrFile);
    }
  }

  // oudf file
  {
    std::string oudfFile;
    if (ini->extract("general", "oudf", oudfFile)) {
      options.setArgs("UDF OKL FILE", oudfFile);
    }
  }

  {
    options.setArgs("SUBCYCLING STEPS", "0");
    int NSubCycles = 0;
    if (ini->extract("general", "advectionsubcyclingsteps", NSubCycles)) {
      options.setArgs("SUBCYCLING STEPS", std::to_string(NSubCycles));
    }
  }

  std::string dtString;
  if (ini->extract("general", "dt", dtString)) {
    const std::vector<std::string> validValues = {
        {"targetcfl"},
        {"max"},
        {"initial"},
    };

    bool useVariableDt = false;
    for (auto &&variableDtEntry : validValues) {
      if (dtString.find(variableDtEntry) != std::string::npos) {
        useVariableDt = true;
      }
    }

    if (useVariableDt) {
      bool userSuppliesInitialDt = false;
      bool userSuppliesTargetCFL = false;
      options.setArgs("VARIABLE DT", "TRUE");
      options.setArgs("TARGET CFL", "0.5");
      std::vector<std::string> entries = serializeString(dtString, '+');
      for (std::string entry : entries) {
        checkValidity(rank, validValues, entry);

        const auto maxStr = parseValueForKey(entry, "max");
        if (!maxStr.empty()) {
          options.setArgs("MAX DT", maxStr);
        }

        const auto initialStr = parseValueForKey(entry, "initial");
        if (!initialStr.empty()) {
          options.setArgs("DT", initialStr);
        }

        const auto cflStr = parseValueForKey(entry, "targetcfl");
        if (!cflStr.empty()) {
          options.setArgs("TARGET CFL", cflStr);
          const double targetCFL = std::stod(cflStr);
          int NSubCycles = std::ceil(targetCFL / 2.0);
          if (targetCFL <= 0.51) {
            NSubCycles = 0;
          }

          int NSubCyclesSpecified = 0;
          if (ini->extract("general", "advectionsubcyclingsteps", NSubCyclesSpecified)) {
            options.setArgs("SUBCYCLING STEPS", std::to_string(NSubCyclesSpecified));
          } else {
            options.setArgs("SUBCYCLING STEPS", std::to_string(NSubCycles));
          }

          userSuppliesTargetCFL = true;
        }
      }

      // if targetCFL is not set, try to infer from subcyclingSteps
      if (!userSuppliesTargetCFL) {
        std::string subCyclingString;
        if (ini->extract("general", "advectionsubcyclingsteps", subCyclingString)) {
          if (subCyclingString.find("auto") != std::string::npos) {
            append_error("subCyclingSteps = auto requires the targetCFL to be set");
            options.setArgs("SUBCYCLING STEPS", "0"); // dummy
          }
        }

        int NSubCycles = 0;
        double targetCFL = 0.5;
        options.getArgs("SUBCYCLING STEPS", NSubCycles);
        if (NSubCycles == 0) {
          targetCFL = 0.5;
        } else {
          targetCFL = 2 * NSubCycles;
        }
        options.setArgs("TARGET CFL", to_string_f(targetCFL));
      }

      // guard against using a higher initial dt than the max
      if (userSuppliesInitialDt) {
        double initialDt = 0.0;
        double maxDt = 0.0;
        options.getArgs("DT", initialDt);
        options.getArgs("MAX DT", maxDt);
        if (maxDt > 0 && initialDt > maxDt) {
          std::ostringstream error;
          error << "initial dt " << initialDt << " is larger than max dt " << maxDt << "\n";
          append_error(error.str());
        }
      }
    } else {
      const double dt = std::stod(dtString);
      options.setArgs("DT", to_string_f(fabs(dt)));
    }
  }

  // check if dt is provided if numSteps or endTime > 0
  {
    int numSteps = 0;
    options.getArgs("NUMBER TIMESTEPS", numSteps);

    double endTime = 0;
    options.getArgs("END TIME", endTime);

    if (numSteps > 0 || endTime > 0) {
      if (options.compareArgs("VARIABLE DT", "FALSE")) {
        const std::string dtString = options.getArgs("DT");
        if (dtString.empty()) {
          append_error("dt not defined!\n");
        }
      }
    }
  }

  options.setArgs("BDF ORDER", "2");
  std::string timeStepper;
  if (ini->extract("general", "timestepper", timeStepper)) {
    if (timeStepper == "bdf3" || timeStepper == "tombo3") {
      options.setArgs("BDF ORDER", "3");
    } else if (timeStepper == "bdf2" || timeStepper == "tombo2") {
      options.setArgs("BDF ORDER", "2");
    } else if (timeStepper == "bdf1" || timeStepper == "tombo1") {
      options.setArgs("BDF ORDER", "1");
    } else {
      std::ostringstream error;
      error << "Could not parse general::timeStepper = " << timeStepper;
      append_error(error.str());
    }
  }

  options.setArgs("EXT ORDER", "3");
  {
    int NSubCycles = 0;
    options.getArgs("SUBCYCLING STEPS", NSubCycles);
    if (NSubCycles) {
      int bdfOrder;
      options.getArgs("BDF ORDER", bdfOrder);
      options.setArgs("EXT ORDER", std::to_string(bdfOrder));
    }
  }

  parseConstFlowRate(rank, options, ini);

  double endTime;
  std::string stopAt = "numsteps";
  ini->extract("general", "stopat", stopAt);
  if (stopAt == "numsteps") {
    int numSteps = 0;
    if (ini->extract("general", "numsteps", numSteps)) {
      options.setArgs("NUMBER TIMESTEPS", std::to_string(numSteps));
      endTime = -1;
    } else {
      append_error("cannot find mandatory parameter GENERAL::numSteps");
    }
    options.setArgs("NUMBER TIMESTEPS", std::to_string(numSteps));
  } else if (stopAt == "endtime") {
    if (!ini->extract("general", "endtime", endTime)) {
      append_error("cannot find mandatory parameter GENERAL::endTime");
    }
    options.setArgs("END TIME", to_string_f(endTime));
  } else if (stopAt == "elapsedtime") {
    double elapsedTime;
    if (!ini->extract("general", "elapsedtime", elapsedTime)) {
      append_error("cannot find mandatory parameter GENERAL::elapsedTime");
    }
    options.setArgs("STOP AT ELAPSED TIME", to_string_f(elapsedTime));
  } else {
    std::ostringstream error;
    error << "Could not parse general::stopAt = " << stopAt;
    append_error(error.str());
  }

  options.setArgs("CHECKPOINT ENGINE", "NEK");
  std::string checkpointEngine;
  if (ini->extract("general", "checkpointengine", checkpointEngine)) {
    if (checkpointEngine == "nek") {
      options.setArgs("CHECKPOINT ENGINE", "NEK");
    } else if (checkpointEngine == "adios") {
      options.setArgs("CHECKPOINT ENGINE", "ADIOS");
#ifndef NEKRS_ENABLE_ADIOS
      append_error("ADIOS engine was requested but is not enabled!\n");
#endif
    } else {
      append_error("invalid checkpointEngine");
    }
  }

  int checkpointPrecision = 0;
  if (ini->extract("general", "checkpointprecision", checkpointPrecision)) {
    if (checkpointPrecision == 64) {
      options.setArgs("CHECKPOINT PRECISION", "FP64");
    } else if (checkpointPrecision == 32) {
      options.setArgs("CHECKPOINT PRECISION", "FP32");
    } else {
      append_error("invalid checkpointPrecision");
    }
  }

  double writeInterval = 0;
  if (!ini->extract("general", "writeinterval", writeInterval)) {
    ini->extract("general", "checkpointinterval", writeInterval);
  }
  options.setArgs("CHECKPOINT INTERVAL", std::to_string(writeInterval));

  std::string writeControl;
  if (!ini->extract("general", "writecontrol", writeControl)) {
    ini->extract("general", "checkpointcontrol", writeControl);
  }

  if (writeControl.size()) {
    checkValidity(rank, {"steps", "simulationtime"}, writeControl);

    if (writeControl == "steps") {
      options.setArgs("CHECKPOINT CONTROL", "STEPS");
    } else if (writeControl == "simulationtime") {
      options.setArgs("CHECKPOINT CONTROL", "SIMULATIONTIME");
    } else {
      std::ostringstream error;
      error << "could not parse general::checkpointControl = " << writeControl;
      append_error(error.str());
    }
  }

  bool dealiasing = true;
  options.setArgs("OVERINTEGRATION", "TRUE");
  if (ini->extract("general", "dealiasing", dealiasing)) {
    if (dealiasing) {
      options.setArgs("OVERINTEGRATION", "TRUE");
    } else {
      options.setArgs("OVERINTEGRATION", "FALSE");
    }
  }

  int cubN = round((3. / 2) * (N + 1) - 1) - 1;
  if (!dealiasing) {
    cubN = 0;
  }
  ini->extract("general", "cubaturepolynomialorder", cubN);
  options.setArgs("CUBATURE POLYNOMIAL DEGREE", std::to_string(cubN));

  {
    parseRegularization(rank, options, ini, "general");
  }
}

