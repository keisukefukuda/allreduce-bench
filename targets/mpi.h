#ifndef ALLREDUCE_BENCH_LIB_MPI
#define ALLREDUCE_BENCH_LIB_MPI

#include <sstream>
#include <string>

#include <mpi.h>

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
    int rank_;

public:
    using ElementType = T;
    Benchmark(MPI_Comm comm) { MPI_Comm_rank(comm, &rank_); }

    //! return a boolean value if the Benchmark is CUDA-aware
    bool gpu() const {
#ifdef USE_CUDA
        return true;
#else
        return false;
#endif
    }

    // return the name of the target
    std::string name() const {
        std::string name = 
#if defined(OMPI_MAJOR_VERSION)
            std::string("Open MPI-")
            + std::to_string(OMPI_MAJOR_VERSION)
            + std::to_string(OMPI_MINOR_VERSION)
            + std::to_string(OMPI_RELEASE_VERSION)
            ;
#elif defined(MVAPICH2_VERSION)
        std::string("MVAPICH-") + MVAPICH2_VERSION;
#elif defined(MPICH_VERSION)
        std::string("MPICH-") + MPICH_VERSION;
#else
        "MPI";
#endif
        
#ifdef USE_CUDA
        name += "-CUDA";
#endif
        return name;
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
        auto mpi_type = MPI_DatatypeTraits<T>::type();

        int ret = MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<void*>(buf), len, mpi_type, MPI_SUM, MPI_COMM_WORLD);
        assert(ret == MPI_SUCCESS);
    }
#else
    void operator()(const T* sendbuf, T* recvbuf, int len) {
        auto mpi_type = MPI_DatatypeTraits<T>::type();

        int ret =
            MPI_Allreduce(reinterpret_cast<const void*>(sendbuf), reinterpret_cast<void*>(recvbuf), len, mpi_type, MPI_SUM, MPI_COMM_WORLD);
        assert(ret == MPI_SUCCESS);
    }
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_MPI
