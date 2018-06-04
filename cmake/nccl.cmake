find_library(NCCL_LIBRARY
  NAMES nccl libnccl libnccl.so
  HINTS ENV LD_LIBRARY_PATH)
find_path(NCCL_INCLUDE_PATH
  NAMES "nccl.h"
  HINTS "${NCCL_LIBRARY}/../../include")

#--------------------------------------------------------------
# NCCL
#--------------------------------------------------------------
if(USE_INPLACE)
  message(STATUS "bench_nccl_cuda is not built because USE_INPLACE is specified")
else()
  if(NCCL_LIBRARY AND CUDA_FOUND)
    message(STATUS "NCCL_LIBRARY=${NCCL_LIBRARY}")
    add_executable(bench_nccl_cuda main.cpp mpi_util.cpp)
    target_compile_definitions(bench_nccl_cuda
      PUBLIC "-DTARGET_NCCL"
      PUBLIC "-DUSE_CUDA"
      )
    target_link_libraries(bench_nccl_cuda ${NCCL_LIBRARY})

    # Our custom compiler driver script ("mpinvcc.sh") checks if the command lines
    # includes "-DUSE_CUDA" and switches to nvcc.
    # target_compile_definitions() is effective only compile time,
    # so we add -DUSE_CUDA to link time options.
    set_target_properties(bench_nccl_cuda PROPERTIES LINK_FLAGS "-DUSE_CUDA")
  else()
    message("NCCL was not found or CUDA is not enabled: bench_nccl_cuda is not built.")
  endif()
endif()

