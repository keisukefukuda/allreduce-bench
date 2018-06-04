#include <cassert>
#include <cstdlib>
#include <cstring>

#include <iostream>  // to be removed
#include <map>
#include <vector>

#include <mpi.h>
#include <unistd.h>

namespace mpiutil {

/**
 * \brief Obtain process rank in the local (physical) node
 */
int get_intra_rank() {
    const char* ompi_local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");

    if (ompi_local_rank) {
        int irank = atoi(ompi_local_rank);
        assert(irank >= 0 && irank <= 8);  // maybe 16 or something in future
        return irank;
    }

    const char* mv2_local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");

    if (mv2_local_rank) {
        int irank = atoi(mv2_local_rank);
        assert(irank >= 0 && irank <= 8);  // maybe 16 or something in future
        return irank;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Gather all host names and assign color for each hostname.
    constexpr int HN_LEN = 200;
    char hostname[HN_LEN] = "";
    gethostname(hostname, HN_LEN);

    int irank = -1;
    int mycolor;

    if (rank == 0) {
        std::vector<char> buf(HN_LEN * size);
        std::vector<int> colors(size);

        MPI_Gather(
            reinterpret_cast<void*>(hostname), HN_LEN, MPI_CHAR, reinterpret_cast<void*>(buf.data()), HN_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        // assign color for each hostname
        std::map<std::string, int> colmap;
        colors[0] = 0;
        int last = 0;
        for (int i = 0; i < size; i++) {
            std::string hn(&(buf[i * HN_LEN]));
            if (colmap.find(hn) == colmap.end()) {
                colmap[hn] = last;
                last++;
            }
            colors[i] = colmap[hn];
        }
        MPI_Scatter(reinterpret_cast<void*>(colors.data()), 1, MPI_INT, &mycolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(reinterpret_cast<void*>(hostname), HN_LEN, MPI_CHAR, nullptr, HN_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, 1, MPI_INT, &mycolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, mycolor, 0, &newcomm);
    MPI_Comm_rank(newcomm, &irank);
    auto err = MPI_Comm_free(&newcomm);

    if (err != MPI_SUCCESS) {
        std::cerr << "ERROR: MPI_Comm_free() failed." << std::endl;
        exit(-1);
    }

    assert(0 <= irank && irank < 8);
    return irank;
}

}  // namespace mpiutil
