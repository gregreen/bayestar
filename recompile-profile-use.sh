#!/bin/sh

# Recompile bayestar using profiling information
rm CMakeCache.txt
cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DPROFILING_USE=true
make -j
