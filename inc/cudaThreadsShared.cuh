#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define MAX_BLOCKS 2147483647

#define MAX_THREADS_PER_BLOCK 1024

#define MAX_SHARED_PER_BLOCK 49152

/* Kernel function: Each thread computes the evolution of a number of elements
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           n(input): single dimension of the array
 *           k(input): number of iterations
 *           blockChunk(input): number of elements in a block
 *           blockCounter(input): number of blocks that have been executed (used for block syncing)
 */
__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter, bool *allBlocksFinished);

/* Wrapper function for the kernel
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 *           threads(input): number of threads per block
 */
void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks, uint32_t threads);

/* Kernel function: Each thread computes the evolution of a number of elements
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           n(input): single dimension of the array
 *           k(input): number of iterations
 *           blockChunk(input): number of elements in a block
 *           blockCounter(input): number of blocks that have been executed (used for block syncing)
 */
__global__ void isingModelGen(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk);

/* Wrapper function for the kernel. This version can handle any number of blocks and threads.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 *           threads(input): number of threads per block
 */
void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks, uint32_t threads);

/* Kernel function: Each thread assigns the final value of a block of elements to the next iteration's input.
 * @params:  out(output): cleared output array
 *           in(input): state of the array after 1 iteration
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 */
__global__ void assignValue(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk);

/* Wrapper function for the kernel. This version contains a graph with node depedencies to improve scheduling and can handle any number of blocks and threads.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 *           threads(input): number of threads per block
 */
void isingCudaGenGraph(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks, uint32_t threads);