# Ising_model

The purpose of this project is to implement a cellular automaton in order to simulate an Ising model on a GPU and observe its states after specified iterations. Multiple versions have been written to compare the different features offered by CUDA.

### Versions
The different versions implemented are explained in greater detail in the report. Here is a resume of them and their mapped number used to call them:
```
0: "SEQ" # sequential CPU version
1: "CUDA_THREADS" # each thread computes a single element of the array (one kernel for all iterations, using `cooperative groups`)
2: "CUDA_BLOCKS"} # each thread computes a block of elements (one kernel for all iterations, using `cooperative groups`)
3: "CUDA_THREADS_SHARED"} # customizable number of blocks and threads per block, employing shared memory within a block (one kernel for all iterations, using `cooperative groups`)
4: "CUDA_THREADS_GEN"} # more general version of 1, suitable for any array size (one kernel launch per iteration)
5: "CUDA_BLOCKS_GEN"} # more general version of 2, suitable for any array size (one kernel launch per iteration)
6: "CUDA_THREADS_SHARED_GEN" # more general version of 3, suitable for any array size (one kernel launch per iteration)
7: "CUDA_BLOCKS_GEN_GRAPH" # `graph` version of 5 (using stream capture)
8: "CUDA_THREADS_GEN_GRAPH" # `graph` version of 4 (using stream capture)
9: "CUDA_THREADS_SHARED_GEN_GRAPH" # `graph` version of 6 (using stream capture)
10: "CUDA_BLOCKS_GEN_STREAMS" # more general version of 2, suitable for any array size, using one `stream` per neighbouring element (one kernel launch per iteration)
11: "CUDA_BLOCKS_GEN_GRAPH_STREAMS" # `graph` version of 10 (manual graph formation)
12: "ALL_GEN" # run all general versions: 4, 5, 6, 10
13: "ALL_GEN_GRAPH" # run all general graph versions: 7, 8, 9, 11
```

### Branches
* `main`: Runs and times a single call of the version specified. Includes the project's report.
* `nanobench`: Performs benchmarks using the [nanobench](https://github.com/andreas-abel/nanoBench) tool 

### How to build and run the repo

To successfully compile and run the program you need to execute the follow commands:

1. `rm -f -r build/`
2. `mkdir build`
3. `cd build/`
4. `cmake ..`
5. `cmake --build .`
6. `cd bin/`
7. `./output ${version} ${n} ${k} ${#blocks} ${#threads per block}` # The last two arguments are optional depeding the version you run. The input array to the program is randomly generated based on `n` which refers to a single dimension of the square lattice. `k` denotes the number of iterations to be run on the lattice, or, in other words, the number of states the lattice will change.

### Notes
* You may need to alter the minimum version of CMake needed in CMakeLists.txt
* In the beginning of the program exist device queries regarding:
   * support of CUDA's `cooperative groups` feature: if not, use the atomic counter for block synchronization
   * maximum number of blocks availabe in the grid: set the `MAX_BLOCKS` macro accordingly
   * maximum number of threads per block: set the `MAX_THREADS_PER_BLOCK` macro accordingly
   * maximum amount of shared memory available per block (in bytes): set the `MAX_SHARED_MEMORY` macro accordingly
* A flag regarding the correctness of the output state is displayed. Its value is determined by whether the ouput state matches the one of the sequential version.
* CUDA does not offer atomic operations on uint8 pointers. Hence, this [custom function](https://stackoverflow.com/a/59329536) was used when necessary.
* For further statistics regarding each benchmark you perform using `nanobench` you may use: `python3 -m pyperf stats file1.json`
* To compare two results you may use: `python3 -m pyperf compare_to --table file1.json file2.json`
