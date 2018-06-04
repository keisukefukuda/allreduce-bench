#ifndef ALLREDUCE_BENCH_LIB_MPI
#define ALLREDUCE_BENCH_LIB_MPI

#include <sstream>
#include <string>

#include <mpi.h>

//////////////////////////////////////////////////////////////
// MPI-related utilities


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
        auto mpi_type = mpiutil::MPI_DatatypeTraits<T>::type();

        int ret = MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<void*>(buf), len, mpi_type, MPI_SUM, MPI_COMM_WORLD);
        assert(ret == MPI_SUCCESS);
    }
#else
    void operator()(const T* sendbuf, T* recvbuf, int len) {
        auto mpi_type = mpiutil::MPI_DatatypeTraits<T>::type();

        int ret =
            MPI_Allreduce(reinterpret_cast<const void*>(sendbuf), reinterpret_cast<void*>(recvbuf), len, mpi_type, MPI_SUM, MPI_COMM_WORLD);
        assert(ret == MPI_SUCCESS);
    }
#endif
};

#endif  // ALLREDUCE_BENCH_LIB_MPI
