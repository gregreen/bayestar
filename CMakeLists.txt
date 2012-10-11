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

#
# Executable
#
add_executable(bayestar src/main.cpp src/model.cpp src/sampler.cpp src/interpolation.cpp)

#
# Link libraries
#
target_link_libraries(bayestar ${GSL_LIBRARIES})