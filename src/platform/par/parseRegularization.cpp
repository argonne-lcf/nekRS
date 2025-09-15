void parseRegularization(const int rank, setupAide &options, inipp::Ini *ini, std::string parSection)
{
  int N;
  options.getArgs("POLYNOMIAL DEGREE", N);
  const bool isScalar = (parSection.find("scalar") != std::string::npos);
  const bool isVelocity = parSection.find("fluid velocity") != std::string::npos;
  std::string sbuf;

  std::string parPrefix = upperCase(parPrefixFromParSection(parSection));

  [&]() {
    std::string regularization;
    if (ini->extract(parSection, "regularization", regularization)) {
      const std::vector<std::string> validValues = {
          {"none"},
          {"hpfrt"},
          {"gjp"},
          {"avm"},
          {"c0"},
          {"nmodes"},
          {"cutoffratio"},
          {"scalingcoeff"},
          {"activationwidth"},
          {"decaythreshold"},
          {"noisethreshold"},

      };
      const std::vector<std::string> list = serializeString(regularization, '+');
      for (const std::string s : list) {
        checkValidity(rank, validValues, s);
      }
      if (regularization.find("none") != std::string::npos) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "NONE");
        return;
      }
      const bool usesAVM = std::find(list.begin(), list.end(), "avm") != list.end();
      const bool usesGJP = std::find(list.begin(), list.end(), "gjp") != list.end();
      const bool usesHPFRT = std::find(list.begin(), list.end(), "hpfrt") != list.end();

      if (!usesAVM && !usesHPFRT && !usesGJP) {
        append_error("unknown regularization!\n");
      }

      if (usesAVM && isVelocity) {
        append_error("avm regularization is only enabled for scalars!\n");
      }

      if (usesGJP) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "GJP");
        options.setArgs(parPrefix + "REGULARIZATION GJP SCALING COEFF", "0.8");
        for (std::string s : list) {
          const auto penaltyStr = parseValueForKey(s, "scalingcoeff");
          if (!penaltyStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION GJP SCALING COEFF", penaltyStr);
          }
        }
      }

      if (usesHPFRT) {
        options.setArgs(parPrefix + "HPFRT MODES", "1");
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "HPFRT");
        if (usesGJP) {
          options.setArgs(parPrefix + "REGULARIZATION METHOD", "GJP+HPFRT");
        }
      }

      if (usesAVM) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY");
        if (usesGJP) {
          options.setArgs(parPrefix + "REGULARIZATION METHOD", "GJP+AVM_AVERAGED_MODAL_DECAY");
        }
        options.setArgs(parPrefix + "REGULARIZATION AVM ACTIVATION WIDTH", to_string_f(1.0));
        options.setArgs(parPrefix + "REGULARIZATION AVM DECAY THRESHOLD", to_string_f(2.0));
        options.setArgs(parPrefix + "REGULARIZATION AVM C0", "FALSE");

        for (std::string s : list) {

          const auto nmodeStr = parseValueForKey(s, "nmodes");
          if (!nmodeStr.empty()) {
            append_error("nModes qualifier is invalid for avm!\n");
          }
          const auto cutoffratioStr = parseValueForKey(s, "cutoffratio");
          if (!cutoffratioStr.empty()) {
            append_error("cutoffRatio qualifier is invalid for avm!\n");
          }

          const auto absTolStr = parseValueForKey(s, "noisethreshold");
          if (!absTolStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM ABSOLUTE TOL", absTolStr);
          }

          const auto scalingcoeffStr = parseValueForKey(s, "scalingcoeff");
          if (!scalingcoeffStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM SCALING COEFF", scalingcoeffStr);
          }

          if (s.find("c0") != std::string::npos) {
            options.setArgs(parPrefix + "REGULARIZATION AVM C0", "TRUE");
          }

          const auto rampConstantStr = parseValueForKey(s, "activationwidth");
          if (!rampConstantStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM ACTIVATION WIDTH", rampConstantStr);
          }
          const auto thresholdStr = parseValueForKey(s, "decaythreshold");
          if (!thresholdStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM DECAY THRESHOLD", thresholdStr);
          }
        }

        if (options.getArgs(parPrefix + "REGULARIZATION AVM ABSOLUTE TOL").empty()) {
          append_error("absoluteTol qualifier required for avm!\n");
        }
      }

      if (usesHPFRT) {
        bool setsStrength = false;
        for (std::string s : list) {
          const auto nmodeStr = parseValueForKey(s, "nmodes");
          if (!nmodeStr.empty()) {
            double value = std::stod(nmodeStr);
            value = round(value);
            options.setArgs(parPrefix + "HPFRT MODES", to_string_f(value));
          }
          const auto cutoffRatioStr = parseValueForKey(s, "cutoffratio");
          if (!cutoffRatioStr.empty()) {
            double filterCutoffRatio = std::stod(cutoffRatioStr);
            double NFilterModes = round((N + 1) * (1 - filterCutoffRatio));
            options.setArgs(parPrefix + "HPFRT MODES", to_string_f(NFilterModes));
          }

          const auto scalingCoeffStr = parseValueForKey(s, "scalingcoeff");
          if (!scalingCoeffStr.empty()) {
            setsStrength = true;
            int err = 0;
            double weight = parseFormula(scalingCoeffStr.c_str(), &err);
            if (err) {
              append_error("Invalid expression for scalingCoeff");
            }
            options.setArgs(parPrefix + "HPFRT STRENGTH", to_string_f(weight));
          }
        }
        if (!setsStrength) {
          append_error("required parameter scalingCoeff for hpfrt regularization is not "
                       "set!\n");
        }
      }
      return;
    } else {
      // if options already exist, don't overwrite them with defaults from general
      std::string regularizationMethod;
      if (options.getArgs(parPrefix + "REGULARIZATION METHOD", regularizationMethod)) {
        return;
      }

      // use default settings, if applicable
      std::string defaultSettings;
      if (ini->extract("general", "regularization", defaultSettings)) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", options.getArgs("REGULARIZATION METHOD"));
        options.setArgs(parPrefix + "HPFRT MODES", options.getArgs("HPFRT MODES"));

        if (defaultSettings.find("hpfrt") != std::string::npos) {
          options.setArgs(parPrefix + "HPFRT STRENGTH", options.getArgs("HPFRT STRENGTH"));
        }

        if (defaultSettings.find("gjp") != std::string::npos) {
          options.setArgs(parPrefix + "REGULARIZATION GJP SCALING COEFF",
                          options.getArgs("REGULARIZATION GJP SCALING COEFF"));
        }

        if (defaultSettings.find("avm") != std::string::npos) {
          if (isVelocity) {
            // Catch if the general block is using AVM + no [VELOCITY] specification
            append_error("avm regularization is only enabled for scalars!\n");
          }
          options.setArgs(parPrefix + "REGULARIZATION VISMAX COEFF",
                          options.getArgs("REGULARIZATION VISMAX COEFF"));
          options.setArgs(parPrefix + "REGULARIZATION SCALING COEFF",
                          options.getArgs("REGULARIZATION SCALING COEFF"));
          options.setArgs(parPrefix + "REGULARIZATION MDH ACTIVATION WIDTH",
                          options.getArgs("REGULARIZATION MDH ACTIVATION WIDTH"));
          options.setArgs(parPrefix + "REGULARIZATION MDH THRESHOLD",
                          options.getArgs("REGULARIZATION MDH THRESHOLD"));
          options.setArgs(parPrefix + "REGULARIZATION AVM C0", options.getArgs("REGULARIZATION AVM C0"));
          options.setArgs(parPrefix + "REGULARIZATION HPF MODES",
                          options.getArgs("REGULARIZATION HPF MODES"));
        }
      }
    }
  }();

  // if regularization method has not been set, fall back to none
  std::string regularizationMethod;
  if (options.getArgs(parPrefix + "REGULARIZATION METHOD", regularizationMethod) == 0) {
    options.setArgs(parPrefix + "REGULARIZATION METHOD", "NONE");
  }
}

