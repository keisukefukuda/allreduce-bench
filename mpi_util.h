#ifndef MPI_UTIL__
#define MPI_UTIL__

namespace mpiutil {

/**
 * \brief Obtain process rank in the local (physical) node
 */

int get_intra_rank();

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

}  // namespace mpiutil

#endif
