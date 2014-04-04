#!/bin/sh

# Recompile bayestar
rm cmake_install.cmake
rm CMakeCache.txt
rm -R CMakeFiles/
cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
make -j
