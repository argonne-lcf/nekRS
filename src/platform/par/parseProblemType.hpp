void parseProblemTypeSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  {
    bool stressFormulation;
    if (ini->extract("problemtype", "stressformulation", stressFormulation)) {
      if (stressFormulation) {
        options.setArgs("FLUID STRESSFORMULATION", "TRUE");
      }
    }

    options.setArgs("EQUATION TYPE", upperCase("navierstokes"));

    std::string eqn;
    if (ini->extract("problemtype", "equation", eqn)) {
      const std::vector<std::string> validValues = {
          {"stokes"},
          {"navierstokes"},
          {"stress"},
          {"variableviscosity"},
      };
      const std::vector<std::string> list = serializeString(eqn, '+');

      auto eqnType = list[0];
      options.setArgs("EQUATION TYPE", upperCase(eqnType));

      for (std::string entry : list) {
        checkValidity(rank, validValues, entry);
      }

      if (std::strstr(eqn.c_str(), "stress") || std::strstr(eqn.c_str(), "variableviscosity")) {
        options.setArgs("FLUID STRESSFORMULATION", "TRUE");
      }
    }
  }
}

