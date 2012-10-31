cmake_minimum_required(VERSION 2.6)
project(bayestar)

#
# Check requirements
#
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

### GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_INLINE=1 -DGSL_RANGE_CHECK=0")

### OpenMP support
#find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})

### Fixed-size types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS")

#
# Executable
#
add_executable(bayestar src/main.cpp src/model.cpp src/sampler.cpp src/interpolation.cpp src/stats.cpp src/chain.cpp src/data.cpp)

#
# Link libraries
#
target_link_libraries(bayestar rt)
target_link_libraries(bayestar hdf5 hdf5_cpp)
target_link_libraries(bayestar ${GSL_LIBRARIES})
#target_link_libraries(bayestar ${OpenMP_LIBRARIES})