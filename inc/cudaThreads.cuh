#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define MAX_BLOCKS 2147483647

#define MAX_THREADS_PER_BLOCK 1024

/* Kernel function: Each thread computes the evolution of a single element
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           n(input): single dimension of the array
 *           k(input): number of iterations
 *           blockCounter(input): number of blocks that have been executed (used for block syncing)
 */
__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, uint32_t *blockCounter, uint8_t *allBlocksFinished);

/* Wrapper function for the kernel
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 */
void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k);