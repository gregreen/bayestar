#!/bin/sh

# Profiling options
prof_input=${1:-0}
prof_gen="FALSE"
prof_use="FALSE"
if [[ ${prof_input} -eq 1 ]]; then
    prof_gen="TRUE"
    prof_use="FALSE"
elif [[ ${prof_input} -eq 2 ]]; then
    prof_gen="FALSE"
    prof_use="TRUE"
fi

echo "PROFILING_GEN=${prof_gen}"
echo "PROFILING_USE=${prof_use}"

# Recompile bayestar
rm cmake_install.cmake
rm CMakeCache.txt
rm -R CMakeFiles/
#cmake . -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
#cmake . -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc
#export LIBRARY_PATH="/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/lib:${LIBRARY_PATH}"
#cmake \
#    -DBoost_NO_SYSTEM_PATHS=BOOL:ON \
#    -D_boost_TEST_VERSIONS=1.63 \
#    -DBOOST_ROOT:PATHNAME=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/ \
#    -DBOOST_LIBRARYDIR=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/lib/ \
#    -DBOOST_INCLUDEDIR=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/include/ \
#    -DBoost_NO_BOOST_CMAKE=BOOL:ON \
#    .

#export LIBRARY_PATH="/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/lib:${LIBRARY_PATH}"
cmake \
    -DBoost_NO_SYSTEM_PATHS=BOOL:ON \
    -D_boost_TEST_VERSIONS=1.63 \
    -DBOOST_ROOT:PATHNAME=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/ \
    -DBOOST_LIBRARYDIR=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/lib/ \
    -DBOOST_INCLUDEDIR=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/include/ \
    -DBoost_NO_BOOST_CMAKE=BOOL:ON \
    -DADDITIONAL_LINK_DIRS=/n/sw/fasrcsw/apps/Comp/gcc/7.1.0-fasrc01/boost/1.63.0-fasrc01/lib \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DPROFILING_GEN=${prof_gen} \
    -DPROFILING_USE=${prof_use} \
    .

make VERBOSE=1 -j
