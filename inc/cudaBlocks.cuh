#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

/* Kernel function: Each thread computes the evolution of a block of elements
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           n(input): single dimension of the array
 *           k(input): number of iterations
 *           blockChunk(input): number of elements in a block
 *           blockCounter(input): number of blocks that have been executed (used for block syncing)
 */
__global__ void isingModelBlocks(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter);

/* Wrapper function for the kernel
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 */
void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks);