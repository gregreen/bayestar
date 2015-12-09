#!/bin/sh

# Recompile bayestar
rm cmake_install.cmake
rm CMakeCache.txt
rm -R CMakeFiles/
#cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
cmake . -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc
make -j
