# AllReduce Microbenchmark
A tiny benchmark program to measure Allreduce() performance from several library

## Build

```
$ mkdir build
$ cd build
$ cmake ..
$ make -j5
```

Some variables to CMake:

 * -DGLOO_ROOT=$GLOO_ROOT
 * -DUSE_INPLACE=1
 * `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CPATH`
 

#### How to build gloo

```bash
$ cd gloo
$ mkdir build
$ cd build
$ cmake .. -DUSE_MPI=yes -DUSE_IBVERBS=yes -DUSE_CUDA=1
$ make -j
```

