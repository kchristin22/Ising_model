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
__global__ void isingModelGen(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk, uint32_t *blockCounter);

/* Wrapper function for the kernel
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 *           threads(input): number of threads per block
 */
void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks, uint32_t threads);