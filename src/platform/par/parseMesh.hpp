void parseMeshSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("mesh")) {
    std::string meshFile;
    if (ini->extract("mesh", "file", meshFile)) {
      options.setArgs("MESH FILE", meshFile);
    }

    std::string meshPartitioner;
    if (ini->extract("mesh", "partitioner", meshPartitioner)) {
      if (meshPartitioner != "rcb" && meshPartitioner != "rcb+rsb") {
        std::ostringstream error;
        error << "Could not parse mesh::partitioner = " << meshPartitioner;
        append_error(error.str());
      }
      options.setArgs("MESH PARTITIONER", meshPartitioner);
    }

    std::string meshConTol;
    if (ini->extract("mesh", "connectivitytol", meshConTol)) {
      options.setArgs("MESH CONNECTIVITY TOL", meshConTol);
    }

    std::string boundaryIDs;
    if (ini->extract("mesh", "boundaryidmap", boundaryIDs)) {
      options.setArgs("MESH BOUNDARY ID MAP", boundaryIDs);
    }

    if (ini->extract("mesh", "boundaryidmapfluid", boundaryIDs)) {
      options.setArgs("MESHV BOUNDARY ID MAP", boundaryIDs);
    }
  }
}

