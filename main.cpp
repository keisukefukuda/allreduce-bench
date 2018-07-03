/**
 * \file main.cpp
 *
 * Main benchmark code for allreduce-microbenchmark
 *
 * Compile-time configuration macros
 *
 * ex)
 * $ mkdir build-dir
 * $ cd build-dir
 * $ cmake ${SOURCE_PATH} -DTARGET_MPI -DUSE_CUDA
 *
 *   * Benchmark target:
 *     * `TARGET_MPI`  : Build MPI version
 *     * `TARGET_NCCL` : Build NCCL version
 *     * `TARGET_GLOO` : Build Gloo version
 *   * CUDA
 *     * `USE_CUDA` : Build CUDA-aware version of the benchmark
 *   * gloo
 *     * GLOO_ROOT : Root directory where gloo is installed
 *                   (typically, your-gloo-copy/build)
 *
 *   * API
 *     * `USE_INPLACE` : Call send/recv APIs in in-place mode
 *       (Several targets are (de-)activated depending on the INPLACE value
 *
 */
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <mpi.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include "mpi_util.h"
#include "util.h"

#include HEADER

#define CUDA_SAFE_CALL(cmd)                                     \
    do {                                                        \
        cudaError_t e = cmd;                                    \
        if (e != cudaSuccess) {                                 \
            printf("Failed: '" #cmd "' failed at %s:%d '%s'\n", \
                   __FILE__, __LINE__, cudaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)


template <class BenchmarkType>
class BenchmarkDriver {
    int mpi_size_;
    int mpi_rank_;
    int mpi_intra_rank_;
    std::string out_dir_;  //!< Output directory

public:
    using ElementType = typename BenchmarkType::ElementType;

    /**
     * \brief Constructor of BenchmarkDriver
     *
     * \param out_dir Name of a directory to which log files are stored
     */
    BenchmarkDriver(const std::string& out_dir) : out_dir_(out_dir) {
        SetupOutputDir();

        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        mpi_intra_rank_ = mpiutil::get_intra_rank();

#ifdef USE_CUDA
        CUDA_SAFE_CALL(cudaSetDevice(mpi_intra_rank_));
#endif
    }

    /**
     * \brief Setup an output directory
     */
    void SetupOutputDir() { util::do_mkdir(out_dir_); }
    
#ifdef USE_CUDA
    void run_allreduce(int repeat, int array_size) {
        thrust::host_vector<ElementType> sendbuf(array_size);
        thrust::host_vector<ElementType> recvbuf(array_size);

        thrust::device_vector<ElementType> gpu_sendbuf(array_size);
        thrust::device_vector<ElementType> gpu_recvbuf(array_size);

        util::create_rand_vector(sendbuf);
        gpu_sendbuf = sendbuf;
        CUDA_SAFE_CALL(cudaStreamSynchronize(0));

        run_bench(repeat,
                  thrust::raw_pointer_cast(gpu_sendbuf.data()),
                  thrust::raw_pointer_cast(gpu_recvbuf.data()), array_size);
    }
#else
    void run_allreduce(int repeat, int array_len) {
        std::vector<ElementType> sendbuf(array_len);
        std::vector<ElementType> recvbuf(array_len);

        util::create_rand_vector(sendbuf);

        run_bench(repeat, sendbuf.data(), recvbuf.data(), array_len);
    }

#endif

private:
    /**
     * \brief Actually run a benchmark with given data and parameters
     *
     * \param repeat Number of repeat
     * \param sendbuf send buffer
     * \param recvbuf receive buffer
     * \param array_len Length of the array (so the size is sizeof(T) * array_len
     */
    void run_bench(int repeat, ElementType* sendbuf, ElementType* recvbuf, int array_len) {
        BenchmarkType bench(MPI_COMM_WORLD);
        std::chrono::system_clock::time_point start, end;

        std::string fname = bench.log_file_name(array_len, mpi_size_);
        std::string path = out_dir_ + "/" + fname;

        FILE* fp = nullptr;

        if (mpi_rank_ == 0) {
            fp = fopen(path.c_str(), "a");
            if (!fp) {
                std::cerr << "Error: Can't open log file '" << path << "'" << std::endl;
                exit(-1);
            }
        }

        double elapsed;
        for (int i = 0; i < repeat; i++) {
            if (mpi_rank_ == 0) {
                std::cerr << "Running " << bench.name() << " [" << i << "/" << repeat << "]" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            start = std::chrono::system_clock::now();
#ifdef USE_INPLACE
            bench(sendbuf, array_len);
#else
            bench(sendbuf, recvbuf, array_len);
#endif
            end = std::chrono::system_clock::now();

            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
            if (mpi_rank_ == 0) {
                fprintf(fp, "%s, %e\n", bench.name().c_str(), elapsed);
            }
        }

        if (mpi_rank_ == 0) {
            fclose(fp);
        }
    }
};

void usage(int argc, char* argv[]) {
    std::cerr << "Usage: " << argv[0] << " "
              << "{output dir} "
              << "{data size} "
              << "{repeat}" << std::endl;
}

size_t parseArraySize(const char *arg) {
    std::string digits;
    std::string unit;

    int i;

    // read the number
    for (i = 0; arg[i] != '\0'; i++) {
        if (isdigit(arg[i])) {
            digits += arg[i];
        } else {
            break;
        }
    }

    for (;  arg[i] != '\0'; i++) {
        unit += tolower(arg[i]);
    }

    size_t size = atoi(digits.c_str());

    if (unit == "k" || unit == "kb") {
        size *= 1024;
    } else if (unit == "m" || unit == "mb") {
        size *= 1024 * 1024;
    } else if (unit == "g" || unit == "g") {
        size *= 1024 * 1024 * 1024;
    } else {
        std::cerr << "Error: can't parse data size: '" << arg << "'" << std::endl;
        exit(-1);
    }

    return size;
}

int main(int argc, char* argv[]) {
    using TargetType = float;

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 4) {
        if (rank == 0) {
            fprintf(stderr, "Expected 3 arguments, but got %d\n", argc - 1);
            usage(argc, argv);
        }
        MPI_Finalize();
        exit(-1);
    }

    std::string out_dir(argv[1]);
    assert(out_dir.size() > 0);

    size_t array_len = parseArraySize(argv[2]) / sizeof(TargetType);

    // effect
    int repeat = atoi(argv[3]);
    assert(0 < repeat);

    BenchmarkDriver<Benchmark<TargetType>> drv(out_dir);
    drv.run_allreduce(repeat, array_len);

    MPI_Finalize();

    return 0;
}
