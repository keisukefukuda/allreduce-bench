#ifndef ALLREDUCE_BENCH_LIB_MPI
#define ALLREDUCE_BENCH_LIB_MPI

#include <cstdlib>
#include <sstream>
#include <string>

#include <mpi.h>

#include <gloo/mpi/context.h>
#include <gloo/transport/ibverbs/device.h>
#ifdef USE_CUDA
#include <gloo/cuda_allreduce_ring.h>
#else
#include <gloo/allreduce_ring.h>
#endif

//////////////////////////////////////////////////////////////
// MPI-related utilities

// From https://github.com/keisukefukuda/tapas/blob/master/include/tapas/mpi_util.h

template <class T>
struct MPI_DatatypeTraits {
    static MPI_Datatype type() { return MPI_BYTE; }
    static constexpr bool IsEmbType() { return false; }

    static constexpr int count(size_t n) { return sizeof(T) * n; }
};

#define DEF_MPI_DATATYPE(ctype_, mpitype_)                 \
    template <>                                            \
    struct MPI_DatatypeTraits<ctype_> {                    \
        static MPI_Datatype type() { return mpitype_; }    \
        static constexpr bool IsEmbType() { return true; } \
        static constexpr int count(size_t n) { return n; } \
    }

DEF_MPI_DATATYPE(char, MPI_CHAR);
DEF_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
DEF_MPI_DATATYPE(wchar_t, MPI_WCHAR);

DEF_MPI_DATATYPE(short, MPI_SHORT);
DEF_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);

DEF_MPI_DATATYPE(int, MPI_INT);
DEF_MPI_DATATYPE(unsigned int, MPI_UNSIGNED);

DEF_MPI_DATATYPE(long, MPI_LONG);
DEF_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);

DEF_MPI_DATATYPE(long long, MPI_LONG_LONG);
DEF_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);

DEF_MPI_DATATYPE(float, MPI_FLOAT);
DEF_MPI_DATATYPE(double, MPI_DOUBLE);
DEF_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);

template <class T>
class Benchmark {
    using Ctx = gloo::mpi::Context;

    MPI_Comm comm_;
    std::shared_ptr<Ctx> context_;

public:
    using ElementType = T;
    Benchmark(MPI_Comm comm) : comm_(comm), context_(std::make_shared<Ctx>(comm_)) {
        const char* hca_name = getenv("GLOO_HCA_NAME");
        const char* hca_port = getenv("GLOO_HCA_NAME");

        std::string name;
        int port;

        if (hca_name && strlen(hca_name) > 0) {
            name = hca_name;
        } else {
            name = "mlx4_0";
        }

        if (hca_port && strlen(hca_port) > 0 && isdigit(hca_port[0])) {
            port = atoi(hca_port);
        } else {
            port = 1;
        }

        gloo::transport::ibverbs::attr attr;
        attr.name = name;
        attr.port = port;
        attr.index = 0;
        auto dev = gloo::transport::ibverbs::CreateDevice(attr);
        context_->connectFullMesh(dev);
    }

    //! return a boolean value if the Benchmark is CUDA-aware
    bool gpu() const {
#ifdef USE_CUDA
        return true;
#else
        return false;
#endif
    }
    // return the name of the Benchmark
    std::string name() const {
#ifdef USE_CUDA
        return "gloo-CUDA";
#else
        return "gloo";
#endif
    }

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
    void operator()(T* buf, int len) {
#ifdef USE_CUDA
        gloo::CudaAllreduceRing<T> allred(context_, {buf}, len);
#else
        gloo::AllreduceRing<T> allred(context_, {buf}, len);
#endif
        allred.run();
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
#error "Not implemented: Gloo only supports in-place allreduce."
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_MPI
