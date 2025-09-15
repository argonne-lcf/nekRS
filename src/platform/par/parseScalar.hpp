void parseScalarSections(const int rank, setupAide &options, inipp::Ini *ini)
{
  nscal = scalarMap.size();
  options.setArgs("NUMBER OF SCALARS", std::to_string(nscal));

  const auto sections = ini->sections;

  auto parseScalarSection = [&](const auto &sec) {
    std::istringstream stream(sec.first);
    std::string firstWord, secondWord;
    stream >> firstWord >> secondWord;
    if (firstWord != "scalar") {
      return;
    }

    std::string sid;
    if (secondWord.empty()) {
      sid = " DEFAULT";
    } else {
      auto it = scalarMap.find(secondWord);
      if (it != scalarMap.end()) {
        sid = scalarDigitStr(scalarMap.at(secondWord));
        options.setArgs("SCALAR" + sid + " NAME", upperCase(secondWord));
      } else {
        append_error("SCALAR " + secondWord + " not found");
      }
    }

    const auto parScope = sec.first;
    parseCheckpointing(rank, options, ini, parScope);
    parseRegularization(rank, options, ini, parScope);

    std::string solver;
    ini->extract(parScope, "solver", solver);

    if (solver == "cvode") {
      cvodeRequested = true;
      options.setArgs("SCALAR" + sid + " SOLVER", "CVODE");
    }

    options.setArgs("SCALAR" + sid + " ELLIPTIC COEFF FIELD", "TRUE");

    parseInitialGuess(rank, options, ini, parScope);

    parsePreconditioner(rank, options, ini, parScope);

    parseLinearSolver(rank, options, ini, parScope);

    parseSolverTolerance(rank, options, ini, parScope);

    std::string sbuf;

    options.setArgs("SCALAR" + sid + " MESH", "FLUID");
    if (ini->extract(parScope, "mesh", sbuf)) {
      auto keys = serializeString(sbuf, '+');
      if (keys.at(0) != "fluid") {
        append_error("Valid mesh values are fluid or fluid+solid");
      }

      if (keys.size() > 1) {
        if (keys.at(1) == "solid") {
          options.setArgs("SCALAR" + sid + " MESH", "FLUID+SOLID");
        } else {
          append_error("Valid mesh values are fluid or fluid+solid");
        }
      }
    }

    if (ini->extract(parScope, "diffusionCoeff", sbuf) || ini->extract(parScope, "conductivity", sbuf)) {
      int err = 0;
      double diffusivity = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for diffusionCoeff");
      }
      if (diffusivity < 0) {
        diffusivity = fabs(1 / diffusivity);
      }
      options.setArgs("SCALAR" + sid + " DIFFUSIONCOEFF", to_string_f(diffusivity));
    }

    if (ini->extract(parScope, "diffusionCoeffSolid", sbuf) ||
        ini->extract(parScope, "conductivitySolid", sbuf)) {
      int err = 0;
      double diffusivity = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for diffusionCoeffSolid");
      }
      if (diffusivity < 0) {
        diffusivity = fabs(1 / diffusivity);
      }
      options.setArgs("SCALAR" + sid + " DIFFUSIONCOEFF SOLID", to_string_f(diffusivity));
    }

    if (ini->extract(parScope, "transportCoeff", sbuf) || ini->extract(parScope, "rhoCp", sbuf)) {
      int err = 0;
      double rho = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for transportCoeff");
      }
      options.setArgs("SCALAR" + sid + " TRANSPORTCOEFF", to_string_f(rho));
    }

    if (ini->extract(parScope, "transportCoeffSolid", sbuf) || ini->extract(parScope, "rhoCpSolid", sbuf)) {
      int err = 0;
      double rho = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for transportCoeffSolid");
      }
      options.setArgs("SCALAR" + sid + " TRANSPORTCOEFF SOLID", to_string_f(rho));
    }

    std::string s_bcMap;
    if (ini->extract(parScope, "boundarytypemap", s_bcMap)) {
      options.setArgs("SCALAR" + sid + " BOUNDARY TYPE MAP", s_bcMap);
    }
  };

  // read default section first.
  if (sections.count("scalar") != 0) {
    parseScalarSection(std::make_pair(std::string("scalar"), sections.at("scalar")));
  }

  // initialize with default settings if available
  const std::string defaultSettingStr = "SCALAR DEFAULT ";
  for (int is = 0; is < nscal; ++is) {
    std::string sid = scalarDigitStr(is);
    const auto options_ = options;
    for (auto [keyWord, value] : options_) {
      auto delPos = keyWord.find(defaultSettingStr);
      if (delPos != std::string::npos) {
        auto newKey = keyWord;
        newKey.erase(delPos, defaultSettingStr.size());
        options.setArgs("SCALAR" + sid + " " + newKey, value);
      }
    }
  }

  // override default settings if specified explicitly
  for (auto &&sec : sections) {
    parseScalarSection(sec);
  }

  // set boundarytypemap from default if available and not specified explicitly
  if (sections.count("scalar") != 0) {
    std::string s_bcMapDefault;
    ini->extract("scalar", "boundarytypemap", s_bcMapDefault);
    for (int is = 0; is < nscal; ++is) {
      std::string sid = scalarDigitStr(is);
      std::string dummy;
      if (!ini->extract("scalar" + sid, "boundarytypemap", dummy)) {
        if (s_bcMapDefault.size() > 0) {
          options.setArgs("SCALAR" + sid + " BOUNDARY TYPE MAP", s_bcMapDefault);
        }
      }
    }
  }

  {
    int nscal;
    options.getArgs("NUMBER OF SCALARS", nscal);
    if (nscal) {
      std::string dummy;
      if (!options.getArgs("SCALAR00 SOLVER", dummy)) {
        append_error("scalar index needs to start from 0");
      }
    }
  }
}

