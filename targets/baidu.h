#ifndef ALLREDUCE_BENCH_LIB_BAIDU_H
#define ALLREDUCE_BENCH_LIB_BAIDU_H

#include <iostream>
#include <sstream>

#include <mpi.h>

// https://github.com/keisukefukuda/baidu-allreduce
#include <collectives.cpp>  // Yes, I know. Don't say anything

#ifndef USE_CUDA
#error "Baidu is available only with CUDA"
#endif

#define CUDACHECK(cmd)                                                                            \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)


template <class T>
class Benchmark {
    MPI_Comm mpi_comm_;
    int rank_;
    int size_;

public:
    using ElementType = T;
    Benchmark(MPI_Comm comm = MPI_COMM_WORLD) : mpi_comm_(comm) {
        int device = mpiutil::get_intra_rank();
        InitCollectives(device);
    }

    ~Benchmark() { }

    //! return a boolean value if the Benchmark is CUDA-aware
    bool gpu() const { return true; }
    // return the name of the Benchmark
    std::string name() const { return "Baidu"; }

    /**
     * \brief Return a log file name that indicates parameters of the run
     *
     * ex.) size1024-rank0.log
     * NOTE: This name is a file name, not a path.
     *       The full path should be built by the caller (driver)
     */
    std::string log_file_name(int len, int nproc) const {
        std::stringstream ss;
        ss << this->name() << "-"
           << "size" << (sizeof(T) * len) << "-"
           << "nproc" << nproc << ".dat";
        return ss.str();
    }

#ifdef USE_INPLACE
#error "Baidu does not support in-place allreduce"
#else
    void operator()(const T* sendbuf, T* recvbuf, int len) {
        RingAllreduce<T>(sendbuf, (size_t)len, recvbuf);
    }
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_BAIDU_H
