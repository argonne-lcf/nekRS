void parseEllipticSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  auto list = serializeString(options.getArgs("USER ELLIPTIC FIELDS"), ',');
  for (auto const &entry : list) {
    const auto parScope = "elliptic " + entry;

    std::string poisson;
    if (ini->extract(parScope, "poisson", poisson)) {
      if (checkForTrue(poisson)) {
        options.setArgs(upperCase(parScope) + " HELMHOLTZ TYPE", "POISSON");
      } 
    }
    options.setArgs(upperCase(parScope) + " ELLIPTIC COEFF FIELD", "FALSE");

    parseInitialGuess(rank, options, ini, parScope);

    parsePreconditioner(rank, options, ini, parScope);

    parseLinearSolver(rank, options, ini, parScope);

    parseSolverTolerance(rank, options, ini, parScope);

    std::string variableCoeff;
    if (ini->extract(parScope, "variableCoeff", variableCoeff)) {
      if (checkForTrue(variableCoeff)) {
        options.setArgs(upperCase(parScope) + " ELLIPTIC COEFF FIELD", "TRUE");
      }
    }

#if 0
    std::string fieldType;
    if (ini->extract(parScope, "vectorfield", fieldType)) {
      if (checkForTrue(fieldType)) {
        options.setArgs(upperCase(parScope) + " VECTOR FIELD", "TRUE");
      }
    }
#endif

    std::string bcMap;
    if (ini->extract(parScope, "boundarytypemap", bcMap)) {
      options.setArgs(upperCase(parScope) + " BOUNDARY TYPE MAP", bcMap);
    }
  }
}

