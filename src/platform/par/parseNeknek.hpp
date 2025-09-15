void parseNekNekSection(const int rank, setupAide &options, inipp::Ini *par)
{
  dlong boundaryEXTOrder = 1;
  if (par->extract("neknek", "boundaryextorder", boundaryEXTOrder)) {
    options.setArgs("NEKNEK BOUNDARY EXT ORDER", std::to_string(boundaryEXTOrder));
  }

  std::string multirateStr;
  if (par->extract("neknek", "multiratetimestepping", multirateStr)) {
    const std::vector<std::string> validValues = {
        {"yes"},
        {"true"},
        {"1"},
        {"no"},
        {"false"},
        {"0"},
        {"correctorsteps"},
    };
    const std::vector<std::string> list = serializeString(multirateStr, '+');
    for (std::string entry : list) {
      checkValidity(rank, validValues, entry);
      const auto correctorStepsStr = parseValueForKey(entry, "correctorsteps");
      if (!correctorStepsStr.empty()) {
        const int correctorSteps = std::stoi(correctorStepsStr);
        options.setArgs("NEKNEK MULTIRATE CORRECTOR STEPS", std::to_string(correctorSteps));
      }
    }
    const bool multirate = checkForTrue(list[0]);
    options.setArgs("NEKNEK MULTIRATE TIMESTEPPER", multirate ? "TRUE" : "FALSE");
  }

  const bool multirate = options.compareArgs("NEKNEK MULTIRATE TIMESTEPPER", "TRUE");

  if (multirate) {
    int correctorSteps = 0;
    options.getArgs("NEKNEK MULTIRATE CORRECTOR STEPS", correctorSteps);
    if (boundaryEXTOrder > 1 && correctorSteps == 0) {
      append_error("Multirate timestepper with boundaryEXTOrder > 1 and correctorSteps = 0 is unstable!\n");
    }
    if (options.compareArgs("VARIABLE DT", "TRUE")) {
      append_error("Multirate timestepper with variable timestep is not supported!\n");
    }
  }
}

