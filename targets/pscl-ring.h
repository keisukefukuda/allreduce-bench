#ifndef ALLREDUCE_BENCH_LIB_PSCL_RING_H
#define ALLREDUCE_BENCH_LIB_PSCL_RING_H

#include <iostream>
#include <sstream>
#include <vector>

#include <ibverbs_communicator.h>
#include <mpi.h>

#ifndef USE_CUDA
#error "PSCL is available only with CUDA"
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
    std::unique_ptr<IBVerbsCommunicator> ibcomm_;

public:
    using ElementType = T;
    Benchmark(MPI_Comm comm = MPI_COMM_WORLD) : mpi_comm_(comm) {
#if 0
        long chunk_size = 8L * (1 << 10) * (1 << 10);
        char envvar[100];
        snprintf(envvar, 100, "%ld", chunk_size);
        printf("IBCOMM_BUFFSIZE=%s\n", envvar);
        setenv("IBCOMM_BUFFSIZE", envvar, 1);
#endif

        MPI_Comm_rank(mpi_comm_, &rank_);
        MPI_Comm_size(mpi_comm_, &size_);

        ibcomm_.reset(new IBVerbsCommunicator(size_));

        std::vector<uint32_t> qps(size_ * 3);

        for (int i = 0; i < size_; i++) {
            if (i == rank_) {
                continue;
            }
            ProcessInfo pinfo = ibcomm_->createQueuePair(i);
            qps[i * 3 + 0] = pinfo.lid;
            qps[i * 3 + 1] = pinfo.qp_n;
            qps[i * 3 + 2] = pinfo.psn;
        }

        MPI_Alltoall(MPI_IN_PLACE, 3, MPI_UINT32_T, qps.data(), 3, MPI_UINT32_T, comm);

        for (int i = 0; i < size_; i++) {
            if (i == rank_) {
                ibcomm_->registerMyself(i);
            } else {
                ProcessInfo pinfo;
                pinfo.lid = qps[i * 3 + 0];
                pinfo.qp_n = qps[i * 3 + 1];
                pinfo.psn = qps[i * 3 + 2];
                ibcomm_->registerQueuePair(i, pinfo);
            }
        }
    }

    ~Benchmark() {}

    //! return a boolean value if the Benchmark is CUDA-aware
    bool gpu() const { return true; }
    // return the name of the Benchmark
    std::string name() const { return "PSCL-RING"; }

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
#error "PSCL does not support in-place allreduce"
#else
    void operator()(const T* sendbuf, T* recvbuf, int len) {
#ifdef USE_CUDA
        ibcomm_->setTimerBase();
        ibcomm_->ring_allreduce_cuda_pool(sendbuf, recvbuf, len);
#endif
    }
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_PSCL_RING_H
