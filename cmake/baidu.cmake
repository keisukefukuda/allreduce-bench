find_path(BAIDU_INCLUDE_PATH
  NAMES "collectives.h"
  HINTS "${BAIDU_ROOT}")

#--------------------------------------------------------------
# Baidu-allreduce
#--------------------------------------------------------------
if(USE_INPLACE)
  message(STATUS "bench_baidu_cuda is not built because USE_INPLACE is specified")
else()
  if(BAIDU_INCLUDE_PATH AND CUDA_FOUND)
    message(STATUS "BAIDU_INCLUDE_DIR=${BAIDU_INCLUDE_PATH}")
    add_executable(bench_baidu_cuda main.cpp mpi_util.cpp)
    target_include_directories(bench_baidu_cuda
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
      PUBLIC "${BAIDU_INCLUDE_PATH}")
      
    target_compile_definitions(bench_baidu_cuda
      PUBLIC "-DTARGET_BAIDU"
      PUBLIC "-DUSE_CUDA"
      PUBLIC "-DHEADER=\"targets/baidu.h\""
      )

    # Our custom compiler driver script ("mpinvcc.sh") checks if the command lines
    # includes "-DUSE_CUDA" and switches to nvcc.
    # target_compile_definitions() is effective only compile time,
    # so we add -DUSE_CUDA to link time options.
    set_target_properties(bench_baidu_cuda PROPERTIES LINK_FLAGS "-DUSE_CUDA")
  else()
    message("Baidu-allreduce was not found or CUDA is not enabled: bench_baidu_cuda is not built.")
  endif()
endif()

