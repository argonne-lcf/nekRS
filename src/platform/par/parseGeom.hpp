void parseGeomSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("geom")) {
    parseLinearSolver(rank, options, ini, "geom");
    if (options.compareArgs("GEOM SOLVER", "USER")) {
      options.setArgs("MOVING MESH", "TRUE");
      options.setArgs("GEOM SOLVER", "NONE");
      options.setArgs("GEOM INTEGRATION ORDER", "3");
    }

    if (!options.compareArgs("GEOM SOLVER", "NONE")) {
      options.setArgs("MOVING MESH", "TRUE");
      options.setArgs("GEOM INTEGRATION ORDER", "3");
      options.setArgs("GEOM HELMHOLTZ TYPE", "POISSON");
      options.setArgs("GEOM ELLIPTIC COEFF FIELD", "TRUE");

      parseInitialGuess(rank, options, ini, "geom");
      parsePreconditioner(rank, options, ini, "geom");
      parseSolverTolerance(rank, options, ini, "geom");

      std::string m_bcMap;
      if (ini->extract("geom", "boundarytypemap", m_bcMap)) {
        options.setArgs("GEOM BOUNDARY TYPE MAP", m_bcMap);
      } else {
        std::string v_bcMap;
        if (ini->extract("fluid velocity", "boundarytypemap", v_bcMap)) {
          options.setArgs("GEOM DERIVED BOUNDARY TYPE MAP", v_bcMap);
        }
      }
    }

    if (options.compareArgs("MOVING MESH", "TRUE")) {
      options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");
    }
  }
}

