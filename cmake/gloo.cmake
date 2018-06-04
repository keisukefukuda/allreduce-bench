find_path(GLOO_INCLUDE_PATH
  NAMES "gloo/mpi/context.h"
  HINTS ${GLOO_ROOT} ${GLOO_ROOT}/build ENV CPATH
  )
find_path(GLOO_CONFIG_PATH
  NAMES "gloo/config.h"
  HINTS ${GLOO_ROOT} ${GLOO_ROOT}/build ENV CPATH
  )
message(STATUS "GLOO_INCLUDE_PATH=${GLOO_INCLUDE_PATH}")

find_library(GLOO_LIBRARY
  NAMES gloo libgloo libgloo.so
  HINTS ENV LD_LIBRARY_PATH
  HINTS ${GLOO_ROOT}
  HINTS ${GLOO_ROOT}/build/gloo)

file(GLOB cmakeFiles ${MYPROJECT_CMAKE_MACRO_DIR}/*.cmake)
foreach(cmakeFile ${cmakeFiles})
  message("INCLUDE ${cmakeFile}")
  INCLUDE(${cmakeFile})
endforeach(cmakeFile)
  
#--------------------------------------------------------------
# gloo
#--------------------------------------------------------------
if(GLOO_CONFIG_PATH)
  if(USE_INPLACE)
    if(CUDA_FOUND)
      add_executable(bench_gloo_cuda main.cpp mpi_util.cpp)
      target_compile_definitions(bench_gloo_cuda
        PUBLIC "-DTARGET_GLOO"
        PUBLIC "-DUSE_CUDA"
        )
      target_link_libraries(bench_gloo_cuda ${GLOO_LIBRARY})
      target_link_libraries(bench_gloo_cuda ${GLOO_CUDA_LIBRARY})
      target_link_libraries(bench_gloo_cuda "ibverbs")
      set_target_properties(bench_gloo_cuda PROPERTIES LINK_FLAGS "-DUSE_CUDA")
      target_include_directories(bench_gloo_cuda PUBLIC ${GLOO_INCLUDE_PATH})
      target_include_directories(bench_gloo_cuda PUBLIC ${GLOO_CONFIG_PATH})
    endif()
    
    add_executable(bench_gloo main.cpp mpi_util.cpp)
    target_compile_definitions(bench_gloo
      PUBLIC "-DTARGET_GLOO"
      )
    target_link_libraries(bench_gloo ${GLOO_LIBRARY})
    target_link_libraries(bench_gloo "ibverbs")
    target_include_directories(bench_gloo PUBLIC ${GLOO_INCLUDE_PATH})
    target_include_directories(bench_gloo PUBLIC ${GLOO_CONFIG_PATH})
  else()
    message(STATUS "bench_gloo and bench_gloo_cuda are not built because USE_INPLACE is not specified")
  endif()
else()
  message(STATUS "gloo was not found: bench_gloo_cuda and bench_gloo are not built.")
endif()

