#ifndef ALLREDUCE_BENCH_LIB_NCCL_H
#define ALLREDUCE_BENCH_LIB_NCCL_H

#include <iostream>
#include <sstream>

#include <mpi.h>
#include <nccl.h>

#ifndef USE_CUDA
#error "NCCL is available only with CUDA"
#endif

#define CUDACHECK(cmd)                                                                            \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                                            \
    do {                                                                                          \
        ncclResult_t r = cmd;                                                                     \
        if (r != ncclSuccess) {                                                                   \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

template <class T>
struct NCCL_DatatypeTraits {
    static ncclDataType_t type() { return ncclUint8; }
    static constexpr bool IsEmbType() { return false; }

    static constexpr int count(size_t n) { return sizeof(T) * n; }
};

#define DEF_NCCL_DATATYPE(ctype_, mpitype_)                \
    template <>                                            \
    struct NCCL_DatatypeTraits<ctype_> {                   \
        static ncclDataType_t type() { return mpitype_; }  \
        static constexpr bool IsEmbType() { return true; } \
        static constexpr int count(size_t n) { return n; } \
    }

DEF_NCCL_DATATYPE(char, ncclChar);
DEF_NCCL_DATATYPE(unsigned char, ncclUint8);

DEF_NCCL_DATATYPE(uint32_t, ncclUint32);
DEF_NCCL_DATATYPE(int32_t, ncclInt32);

DEF_NCCL_DATATYPE(uint64_t, ncclUint64);
DEF_NCCL_DATATYPE(int64_t, ncclInt64);

DEF_NCCL_DATATYPE(float, ncclFloat32);
DEF_NCCL_DATATYPE(double, ncclFloat64);

template <class T>
class Benchmark {
    MPI_Comm mpi_comm_;
    int rank_;
    int size_;
    ncclComm_t ncclcomm_;

public:
    using ElementType = T;
    Benchmark(MPI_Comm comm = MPI_COMM_WORLD) : mpi_comm_(comm) {
        MPI_Comm_rank(mpi_comm_, &rank_);
        MPI_Comm_size(mpi_comm_, &size_);

        ncclUniqueId nccl_uid;
        if (rank_ == 0) {
            NCCLCHECK(ncclGetUniqueId(&nccl_uid));
        }
        MPI_Bcast(&nccl_uid, sizeof(nccl_uid), MPI_BYTE, 0, mpi_comm_);
        NCCLCHECK(ncclCommInitRank(&ncclcomm_, size_, nccl_uid, rank_));
    }

    ~Benchmark() { ncclCommDestroy(ncclcomm_); }

    //! return a boolean value if the Benchmark is CUDA-aware
    bool gpu() const { return true; }
    // return the name of the Benchmark
    std::string name() const { return "NCCL"; }

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
        auto nccl_type = NCCL_DatatypeTraits<T>::type();

        NCCLCHECK(
            ncclAllReduce(reinterpret_cast<const void*>(sendbuf), reinterpret_cast<void*>(recvbuf), len, nccl_type, ncclSum, ncclcomm_, 0));
        CUDACHECK(cudaStreamSynchronize(0));
    }
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_NCCL_H
