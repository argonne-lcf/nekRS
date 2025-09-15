void parseBoomerAmgSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("boomeramg")) {
    append_error("BOOMERAMG section requires field name!");
  }

  std::vector<std::string> list;
  for (auto const &option : options) {
    if (option.second.find("BOOMERAMG") == 0) {
      auto val = option.first;

      const std::string keyword = "MULTIGRID COARSE SOLVER";
      const auto pos = val.find(keyword);
      if (pos == std::string::npos) continue;

      val = val.substr(0, pos);
      val.erase(std::find_if(val.rbegin(), val.rend(),
               [](unsigned char ch) { return !std::isspace(ch); }).base(),
              val.end());
      list.push_back(val);
    }
  }

  for(auto& entry : list) { 
    auto prefix = entry + " ";
    if (ini->sections.count(lowerCase(prefix) + "boomeramg")) {
      int coarsenType;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "coarsentype", coarsenType)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG COARSEN TYPE", std::to_string(coarsenType));
      }
      int interpolationType;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "interpolationtype", interpolationType)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG INTERPOLATION TYPE", std::to_string(interpolationType));
      }
      int smootherType;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "smoothertype", smootherType)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG SMOOTHER TYPE", std::to_string(smootherType));
      }
      int coarseSmootherType;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "coarsesmoothertype", coarseSmootherType)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG COARSE SMOOTHER TYPE", std::to_string(coarseSmootherType));
      }
      int numCycles;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "iterations", numCycles)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG ITERATIONS", std::to_string(numCycles));
      }
      double strongThres;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "strongthreshold", strongThres)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG STRONG THRESHOLD", to_string_f(strongThres));
      }
      double nonGalerkinTol;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "nongalerkintol", nonGalerkinTol)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG NONGALERKIN TOLERANCE", to_string_f(nonGalerkinTol));
      }
      int aggLevels;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "aggressivecoarseninglevels", aggLevels)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG AGGRESSIVE COARSENING LEVELS", std::to_string(aggLevels));
      }
      int chebyRelaxOrder;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "chebyshevrelaxorder", chebyRelaxOrder)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG CHEBYSHEV RELAX ORDER", std::to_string(chebyRelaxOrder));
      }
      double chebyFraction;
      if (ini->extract(lowerCase(prefix) + "boomeramg", "chebyshevfraction", chebyFraction)) {
        options.setArgs(upperCase(prefix) + "BOOMERAMG CHEBYSHEV FRACTION", std::to_string(chebyFraction));
      }
    }
  }
}

