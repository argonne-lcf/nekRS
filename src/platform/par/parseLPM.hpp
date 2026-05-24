void parseLPMSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::string checkpointDirectory;
  if (ini->extract("lpm", "checkpointdirectory", checkpointDirectory)) {
    options.setArgs("LPM CHECKPOINT DIRECTORY", checkpointDirectory);
  }
}

