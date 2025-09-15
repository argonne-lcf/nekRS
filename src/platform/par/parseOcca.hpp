void parseOccaSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::string backendSpecification;
  if (ini->extract("occa", "backend", backendSpecification)) {
    const std::vector<std::string> validBackends = {
        {"serial"},
        {"cpu"},
        {"cuda"},
        {"hip"},
        {"dpcpp"},
        {"opencl"},
    };
    const std::vector<std::string> validArchitectures = {
        {"arch"}, // include the arch= specifier here
        {"x86"},
    };

    std::vector<std::string> validValues = validBackends;
    validValues.insert(validValues.end(), validArchitectures.begin(), validArchitectures.end());

    const std::vector<std::string> list = serializeString(backendSpecification, '+');
    for (const std::string entry : list) {
      const std::vector<std::string> arguments = serializeString(entry, '=');
      for (const std::string argument : arguments) {
        checkValidity(rank, validValues, argument);
      }
    }

    std::string threadModel = "";
    std::string architecture = "";
    for (const std::string entry : list) {
      const std::vector<std::string> arguments = serializeString(entry, '=');
      if (arguments.size() == 1) {
        for (const std::string backend : validBackends) {
          if (backend == arguments.at(0)) {
            threadModel = backend;
          }
        }
      } else if (arguments.size() == 2) {
        for (const std::string arch : validArchitectures) {
          if (arch == arguments.at(1)) {
            architecture = arch;
          }
        }
      } else {
        std::ostringstream error;
        error << "Could not parse string \"" << entry << "\" while parsing OCCA:backend.\n";
        append_error(error.str());
      }
    }

    if (threadModel.empty()) {
      std::ostringstream error;
      error << "Could not parse valid backend from \"" << backendSpecification
            << "\" while parsing OCCA:backend.\n";
      append_error(error.str());
    }

    options.setArgs("THREAD MODEL", upperCase(threadModel));

    if (!architecture.empty()) {
      options.setArgs("ARCHITECTURE", upperCase(architecture));
    }
  }

  std::string deviceNumber;
  if (ini->extract("occa", "devicenumber", deviceNumber)) {
    options.setArgs("DEVICE NUMBER", upperCase(deviceNumber));
  }

  std::string platformNumber;
  if (ini->extract("occa", "platformnumber", platformNumber)) {
    options.setArgs("PLATFORM NUMBER", upperCase(platformNumber));
  }
}

