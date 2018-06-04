#--------------------------------------------------------------
# MPI
#--------------------------------------------------------------
add_executable(bench_mpi main.cpp mpi_util.cpp)
target_compile_definitions(bench_mpi
  PUBLIC "-DTARGET_MPI"
  )

if(CUDA_FOUND)
  add_executable(bench_mpi_cuda main.cpp mpi_util.cpp)
  target_compile_definitions(bench_mpi_cuda
    PUBLIC "-DTARGET_MPI"
    PUBLIC "-DUSE_CUDA"
    )
  set_target_properties(bench_mpi_cuda PROPERTIES LINK_FLAGS "-DUSE_CUDA")
else()
  message("CUDA was not found: bench_mpi_cuda is not built.")
endif()

