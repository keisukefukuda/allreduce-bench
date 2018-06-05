find_path(PSCL_INCLUDE_PATH
  NAMES "ibverbs_communicator.h"
  HINTS "${PSCL_ROOT}" "${PSCL_ROOT}/ibcomm")

find_library(PSCL_LIBRARIES
  NAMES "ibcomm_cuda"
  HINTS "${PSCL_ROOT}" "${PSCL_ROOT}/build")

#--------------------------------------------------------------
# Pscl-allreduce
#--------------------------------------------------------------
if(USE_INPLACE)
  message(STATUS "bench_pscl_cuda is not built because USE_INPLACE is specified")
else()
  if(PSCL_INCLUDE_PATH AND CUDA_FOUND)
    message(STATUS "PSCL_LIBRARIES=${PSCL_LIBRARIES}")
    message(STATUS "PSCL_INCLUDE_PATH=${PSCL_INCLUDE_PATH}")
    add_executable(
      bench_pscl_cuda
      main.cpp mpi_util.cpp
      )
    target_include_directories(bench_pscl_cuda
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
      PUBLIC "${PSCL_INCLUDE_PATH}")
      
    target_compile_definitions(bench_pscl_cuda
      PUBLIC "-DTARGET_PSCL"
      PUBLIC "-DUSE_CUDA"
      PUBLIC "-DHEADER=\"targets/pscl.h\""
      )
    target_link_libraries(bench_pscl_cuda ${PSCL_LIBRARIES})

    # Our custom compiler driver script ("mpinvcc.sh") checks if the command lines
    # includes "-DUSE_CUDA" and switches to nvcc.
    # target_compile_definitions() is effective only compile time,
    # so we add -DUSE_CUDA to link time options.
    set_target_properties(bench_pscl_cuda PROPERTIES LINK_FLAGS "-DUSE_CUDA")
  else()
    message("Pscl-allreduce was not found or CUDA is not enabled: bench_pscl_cuda is not built.")
  endif()
endif()

