#include <algorithm>
#include <iostream>
#include <random>

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace util {

template <class Vector>
void create_rand_vector(Vector& vec) {
    std::mt19937 rnd(0);
    std::uniform_real_distribution<> dist(-1, 1);

    std::generate(vec.begin(), vec.end(), [&]() { return dist(rnd); });
}

void do_mkdir(const std::string& dir_name) {
    struct stat st;

    if (stat(dir_name.c_str(), &st) != 0) {
        // `path` does not exist. Create it
        mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
        int ret = mkdir(dir_name.c_str(), mode);
        if (ret != 0 && errno != EEXIST) {
            std::cerr << "Error: Can't create '" << dir_name << "' : error" << std::endl;
            exit(-1);
        }
    } else if (!S_ISDIR(st.st_mode)) {
        std::cerr << "Error: Can't create " << dir_name << " : Not a directory" << std::endl;
        exit(-1);
    }
}

};  // namespace util
