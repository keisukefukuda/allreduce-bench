find_path(PFNPROTO_INCLUDE_PATH
  NAMES "ibcomm/ibverbs_communicator.h"
  HINTS "${PFNPROTO_ROOT}"
  )

find_library(PFNPROTO_LIBRARIES
  NAMES "ibcomm_cuda"
  HINTS "${PFNPROTO_ROOT}" "${PFNPROTO_ROOT}/build")

#--------------------------------------------------------------
# Pfnproto-allreduce
#--------------------------------------------------------------
if(USE_INPLACE)
  message(STATUS "bench_pfnproto_ring_cuda is not built because USE_INPLACE is specified")
else()
  if(PFNPROTO_INCLUDE_PATH AND CUDA_FOUND)
    message(STATUS "PFNPROTO_LIBRARIES=${PFNPROTO_LIBRARIES}")
    message(STATUS "PFNPROTO_INCLUDE_PATH=${PFNPROTO_INCLUDE_PATH}")
    add_executable(
      bench_pfnproto_ring_cuda
      main.cpp mpi_util.cpp
      )
    target_include_directories(bench_pfnproto_ring_cuda
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
      PUBLIC "${PFNPROTO_INCLUDE_PATH}")
      
    target_compile_definitions(bench_pfnproto_ring_cuda
      PUBLIC "-DTARGET_PFNPROTO"
      PUBLIC "-DUSE_CUDA"
      PUBLIC "-DHEADER=\"targets/pfnproto-ring.h\""
      )
    target_link_libraries(bench_pfnproto_ring_cuda ${PFNPROTO_LIBRARIES})

    # Our custom compiler driver script ("mpinvcc.sh") checks if the command lines
    # includes "-DUSE_CUDA" and switches to nvcc.
    # target_compile_definitions() is effective only compile time,
    # so we add -DUSE_CUDA to link time options.
    set_target_properties(bench_pfnproto_ring_cuda PROPERTIES LINK_FLAGS "-DUSE_CUDA")
  else()
    message("Pfnproto-allreduce was not found or CUDA is not enabled: bench_pfnproto_ring_cuda is not built.")
  endif()
endif()

