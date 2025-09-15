#include <sys/stat.h>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <libgen.h>
#include <string>
#include <cstring>
#include <iostream>

#include "fileUtils.hpp"

static std::string getParentPath(const std::string& filePath) {
    std::size_t pos = filePath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filePath.substr(0, pos);
    }
    return "";
}

void fileSync(const char *file)
{
  const std::string dir = getParentPath(std::string(file));

  int fd;
  fd = open(file, O_RDONLY);
  fsync(fd);
  close(fd);

  fd = open(dir.c_str(), O_RDONLY);
  fsync(fd);
  close(fd);
}

bool isFileNewer(const char *file1, const char *file2)
{
  struct stat s1, s2;
  lstat(file1, &s1);
  if (lstat(file2, &s2) != 0) {
    return true;
  }
  if (s1.st_mtime > s2.st_mtime) {
    return true;
  } else {
    return false;
  }
}

void copyFile(const char *srcFile, const char *dstFile)
{
  std::ifstream src(srcFile, std::ios::binary);
  std::ofstream dst(dstFile, std::ios::trunc | std::ios::binary);
  dst << src.rdbuf();
  src.close();
  dst.close();
  fileSync(dstFile);
}

bool fileExists(const char *file)
{
  return realpath(file, NULL);
}

bool isFileEmpty(const char *file)
{
  std::ifstream f(file);
  const bool isEmpty = f.peek() == std::ifstream::traits_type::eof();
  f.close();
  return isEmpty;
}
