#!/bin/sh

# Recompile bayestar to generate profiling information
rm cmake_install.cmake
rm CMakeCache.txt
rm -R CMakeFiles/
rm -R profiling/*
cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DPROFILING_GEN=true
make -j
