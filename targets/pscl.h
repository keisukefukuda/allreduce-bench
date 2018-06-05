#ifndef ALLREDUCE_BENCH_LIB_PSCL_H
#define ALLREDUCE_BENCH_LIB_PSCL_H

#include <iostream>
#include <sstream>

#include <mpi.h>
#include <ibverbs_communicator.h>

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
        MPI_Comm_rank(mpi_comm_, &rank_);
        MPI_Comm_size(mpi_comm_, &size_);

        ibcomm_.reset(new IBVerbsCommunicator(size_));

        ibcomm_->registerMyself(rank_);

        for (int i = 0; i < rank_; i++) {
            ProcessInfo pinfo;
            MPI_Recv(&pinfo, sizeof(pinfo), MPI_BYTE, i, 0, comm, MPI_STATUS_IGNORE);
            pinfo = ibcomm_->registerProcess(i, pinfo);
            MPI_Send(&pinfo, sizeof(pinfo), MPI_BYTE, i, 0, comm);
        }

        for (int i = rank_ + 1; i < size_; i++) {
            auto pinfo = ibcomm_->createQueuePair(i);
            MPI_Send(&pinfo, sizeof(pinfo), MPI_BYTE, i, 0, comm);
            MPI_Recv(&pinfo, sizeof(pinfo), MPI_BYTE, i, 0, comm, MPI_STATUS_IGNORE);
            ibcomm_->registerQueuePair(i, pinfo);
        }
    }

    ~Benchmark() {
    }

    //! return a boolean value if the Benchmark is CUDA-aware
    bool gpu() const { return true; }
    // return the name of the Benchmark
    std::string name() const { return "PSCL"; }

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
#error "NCCL does not support in-place allreduce"
#else
    void operator()(const T* sendbuf, T* recvbuf, int len) {
#ifdef USE_CUDA
        ibcomm_->ring_allreduce_cuda_pool(sendbuf, recvbuf, len);
#endif
    }
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_PSCL_H
