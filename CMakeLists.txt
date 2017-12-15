cmake_minimum_required(VERSION 2.6)
project(bayestar)

#
# Use pkg-config to find some libraries
#
INCLUDE(FindPkgConfig)
find_package(PkgConfig REQUIRED)

#
# Check requirements
#
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Git version
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
	FIND_PACKAGE(Git)
	IF(GIT_FOUND)
		EXECUTE_PROCESS(
			COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			OUTPUT_VARIABLE "GIT_BUILD_VERSION"
			ERROR_QUIET
			OUTPUT_STRIP_TRAILING_WHITESPACE)
		MESSAGE( STATUS "Git version: ${GIT_BUILD_VERSION}" )
	ELSE(GIT_FOUND)
		SET(GIT_BUILD_VERSION 0)
	ENDIF(GIT_FOUND)
ELSE(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
	SET(GIT_BUILD_VERSION "$ENV{BAYESTARCOMMIT}")
ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

#
# Configure a header file to pass some of the CMake
# settings to the source code
#
#set(DATADIR "${CMAKE_INSTALL_PREFIX}/data/")
set(DATADIR "${CMAKE_SOURCE_DIR}/data/")
configure_file(
	"${PROJECT_SOURCE_DIR}/src/bayestar_config.h.in"
	"${PROJECT_BINARY_DIR}/src/bayestar_config.h"
	ESCAPE_QUOTES
)
include_directories("${PROJECT_BINARY_DIR}/src")

#
# Optimizations
#
IF(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ipo -axHOST -no-prec-div ")
ELSE(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=native ")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O1 -g ")
ENDIF(NOT CMAKE_COMPILER_IS_GNUCXX)

#
# Non-compiler-specific flags
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 ")

#
# Standard revision, ABI, etc.
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

#
# Profiling
#
IF(CMAKE_COMPILER_IS_GNUCXX)
	IF(PROFILING_GEN)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate -pg ")
		#set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -fprofile-generate=profiling/ -pg ")
		MESSAGE( STATUS "Setting flags to generate profiling information." )
	ENDIF(PROFILING_GEN)
	IF(PROFILING_USE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use ")
		#set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -fprofile-use=profiling/ ")
		MESSAGE( STATUS "Setting flags to use profiling information." )
	ENDIF(PROFILING_USE)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

### HDF5
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
MESSAGE( STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}" )
MESSAGE( STATUS "HDF5_LIBRARY_DIRS: ${HDF5_LIBRARY_DIRS}" )
MESSAGE( STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}" )
MESSAGE( STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}" )

### OpenCV
pkg_check_modules(OPENCV opencv)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
MESSAGE( STATUS "OPENCV_LIBRARY_DIRS: ${OPENCV_LIBRARY_DIRS}" )
MESSAGE( STATUS "OPENCV_INCLUDE_DIRS: ${OPENCV_INCLUDE_DIRS}" )
MESSAGE( STATUS "OPENCV_LIBRARIES: ${OPENCV_LIBRARIES}" )

### GSL
pkg_check_modules(GSL gsl)
#find_package(GSL REQUIRED)
MESSAGE( STATUS "GSL_INCLUDE_DIRS: ${GSL_INCLUDE_DIR}" )
MESSAGE( STATUS "GSL_LIBRARY_DIRS: ${GSL_LIBRARY_DIRS}" )
include_directories(${GSL_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_INLINE=1 -DGSL_RANGE_CHECK=0")

### Boost
#pkg_check_modules(BOOST boost)
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
MESSAGE( STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}" )
MESSAGE( STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}" )


### OpenMP support
#find_package(OpenMP REQUIRED)
IF(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
ELSE(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF(NOT CMAKE_COMPILER_IS_GNUCXX)
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})

### EIGEN3
pkg_check_modules(EIGEN3 eigen3)
#MESSAGE( STATUS "Eigen3 found: ${EIGEN3_FOUND}" )
#find_package(Eigen3 REQUIRED NO_MODULES)
include_directories(${EIGEN3_INCLUDE_DIRS})
MESSAGE( STATUS "EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}" )
MESSAGE( STATUS "EIGEN3_LIBRARIES: ${EIGEN3_LIBRARIES}" )

### Parallel Tempering Sampler
#include_directories(/home/greg/projects/sampler-cpp/src/)

### Fixed-size types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS")

### Add additional link directories to search in
#link_directories(${ADDITIONAL_LINK_DIRS})
#MESSAGE( STATUS "Additional link directories: ${ADDITIONAL_LINK_DIRS}" )

#
# Executable
#
add_executable(bayestar src/main.cpp src/model.cpp src/sampler.cpp
                        src/interpolation.cpp src/stats.cpp src/chain.cpp
                        src/data.cpp src/binner.cpp src/los_sampler.cpp
												src/h5utils.cpp src/star_exact.cpp
												src/program_opts.cpp)

#
# Link libraries
#
#cmake_policy(SET CMP0003 OLD)
#cmake_policy(SET CMP0060 NEW)
target_link_libraries(bayestar rt)
target_link_libraries(bayestar ${HDF5_LIBRARIES})# ${HDF5_CXX_LIBRARIES})
#target_link_libraries(bayestar hdf5 hdf5_cpp)
target_link_libraries(bayestar ${GSL_LIBRARIES})
target_link_libraries(bayestar ${Boost_LIBRARIES})
target_link_libraries(bayestar opencv_core opencv_imgproc)
#target_link_libraries(bayestar ${OPENCV_LIBRARIES})
#target_link_libraries(bayestar ${OpenMP_LIBRARIES})
#target_link_libraries(bayestar /home/greg/projects/sampler-cpp/libsampler.a)
target_link_libraries(bayestar ${EIGEN3_LIBRARIES})
target_link_libraries(bayestar -Wl,--verbose)
IF(ADDITIONAL_LINK_DIRS)
	target_link_libraries(bayestar -L${ADDITIONAL_LINK_DIRS})
    MESSAGE( STATUS "Additional link directories: ${ADDITIONAL_LINK_DIRS}" )
ENDIF(ADDITIONAL_LINK_DIRS)
