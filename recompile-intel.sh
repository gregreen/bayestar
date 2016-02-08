#!/bin/sh

# Determine if suitable binary exists
procModel=`grep -Po '(?<=model name).*' /proc/cpuinfo | head -n 1 | grep -Po '(?<=:\s).*'`
echo "Intel compilation script detected processor model '${procModel}'"

binDir="/n/fink1/ggreen/bayestar/bin/precomp/${procModel}"
binFileName="${binDir}/bayestar"

if [ -f "${binFileName}" ]; then
  echo "Suitable binary found: '${binFileName}'"
  echo "Copying bayestar binary into local directory"
  sleep 5
  cp "${binFileName}" .
else
  echo "Binary not found: '${binFileName}'."
  echo "Compiling ..."

  # Recompile bayestar
  rm cmake_install.cmake
  rm CMakeCache.txt
  rm -R CMakeFiles/
  #cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
  cmake . -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc
  make -j

  # Copy binary to standard directory so that other jobs can use it
  echo "Moving bayestar binary to '${binDir}'"
  mkdir -p "${binDir}"

  # Copy file to temporary directory on /n/fink1
  binDirTmp="/n/fink1/ggreen/tmp/${RANDOM}"
  mkdir -p "${binDirTmp}"
  cp bayestar "${binDirTmp}/bayestar"

  # Do atomic move operation to standard location
  mv "${binDirTmp}/bayestar" "${binDir}/bayestar"

  # Remove temporary directory on /n/fink1
  rmdir "${binDirTmp}"
fi
