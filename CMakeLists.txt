# By Keisuke Fukuda
# Copied from
# https://github.com/cginternals/cmake-init
# To be replaced before going public.
# 
# CMake options
# 

project("allreduce-microbench" C CXX)

# CMake version
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

find_package(MPI)
find_package(CUDA)

file(GLOB cmakeFiles ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
foreach(cmakeFile ${cmakeFiles})
  message(STATUS "INCLUDE ${cmakeFile}")
  INCLUDE(${cmakeFile})
endforeach(cmakeFile)

list(APPEND CXXFLAGS "-std=c++11 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH "Pascal")
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

if(USE_INPLACE)
  set(CXXFLAGS "${CXXFLAGS} -DUSE_INPLACE")
endif()

# Run mpicxx -show to get compile flags for MPI
set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/mpinvcc.sh")
set(CMAKE_CXX_FLAGS ${CXXFLAGS})

# debug
# get_cmake_property(_variableNames VARIABLES)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
#   endforeach()

