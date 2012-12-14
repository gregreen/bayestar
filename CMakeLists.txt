cmake_minimum_required(VERSION 2.6)
project(bayestar)

#
# Check requirements
#
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

#
# configure a header file to pass some of the CMake settings to the source
# code
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
# Intel Compiler Flags
#
IF(NOT CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ipo -axHOST -no-prec-div ") 
#ELSE(NOT CMAKE_COMPILER_IS_GNUCXX)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math ")
ENDIF(NOT CMAKE_COMPILER_IS_GNUCXX)

### OpenCV
find_package(OpenCV REQUIRED)

### GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_INLINE=1 -DGSL_RANGE_CHECK=0")

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
add_executable(bayestar src/main.cpp src/model.cpp src/sampler.cpp src/interpolation.cpp src/stats.cpp src/chain.cpp src/data.cpp src/binner.cpp src/los_sampler.cpp src/h5utils.cpp)

#
# Link libraries
#
target_link_libraries(bayestar rt)
target_link_libraries(bayestar hdf5 hdf5_cpp)
target_link_libraries(bayestar ${GSL_LIBRARIES})

target_link_libraries(bayestar opencv_core opencv_imgproc)# ${OpenCV_LIBS})
#target_link_libraries(bayestar ${OpenMP_LIBRARIES})
