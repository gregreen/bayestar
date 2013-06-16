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
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -g -rdynamic ") #-fno-math-errno -funsafe-math-optimizations ")
ENDIF(NOT CMAKE_COMPILER_IS_GNUCXX)

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

### OpenCV
pkg_check_modules(OPENCV opencv)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
MESSAGE( STATUS "OPENCV_LIBRARY_DIRS: ${OPENCV_LIBRARY_DIRS}" )
MESSAGE( STATUS "OPENCV_INCLUDE_DIRS: ${OPENCV_INCLUDE_DIRS}" )
MESSAGE( STATUS "OPENCV_LIBRARIES: ${OPENCV_LIBRARIES}" )

### GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_INLINE=1 -DGSL_RANGE_CHECK=0")

### Boost
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

### OpenMP support
#find_package(OpenMP REQUIRED)
IF(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
ELSE(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF(NOT CMAKE_COMPILER_IS_GNUCXX)
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})

### Fixed-size types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS")

#
# Executable
#
add_executable(bayestar src/main.cpp src/model.cpp src/sampler.cpp
                        src/interpolation.cpp src/stats.cpp src/chain.cpp
                        src/data.cpp src/binner.cpp src/los_sampler.cpp src/h5utils.cpp)

#
# Link libraries
#
target_link_libraries(bayestar rt)
target_link_libraries(bayestar hdf5 hdf5_cpp)
target_link_libraries(bayestar ${GSL_LIBRARIES})
target_link_libraries(bayestar ${Boost_LIBRARIES})
target_link_libraries(bayestar opencv_core opencv_imgproc)
#target_link_libraries(bayestar ${OPENCV_LIBRARIES})
#target_link_libraries(bayestar ${OpenMP_LIBRARIES})
