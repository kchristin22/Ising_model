#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define MAX_BLOCKS 2147483647

#define MAX_THREADS_PER_BLOCK 1024

using funcP = size_t (*)(size_t, size_t);

/* Kernel function: Each thread computes the evolution of a single element
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           n(input): single dimension of the array
 *           k(input): number of iterations
 *           blockCounter(input): number of blocks that have been executed (used for block syncing)
 */
__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, uint32_t *blockCounter, bool *allBlocksFinished);

/* Wrapper function for the kernel
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array, it must use resources below the constraints of cooperative groups
 *           k(input): number of iterations
 */
void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k);

/* Kernel function: Each thread adds the neighbouring values to an element.
 * @params:  out(output): sum of neighbours of each element of the array after 1 iteration
 *           in(input): input array
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 *           calcOffset(input): function to calculate the offset of the output array elements to compute
 */
__global__ void addValue(uint8_t *out, const uint8_t *in, const size_t n);

/* Kernel function: Each thread assigns the final value of an elements to the next iteration's input.
 * @params:  out(output): cleared output array
 *           in(input): state of the array after 1 iteration
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 */
__global__ void assignClearValue(uint8_t *out, uint8_t *in, const size_t n);

/* Wrapper function for the kernel. This version can handle any size of array.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 */
void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k);

/* Wrapper function for the kernel. This version contains a graph with node depedencies to improve scheduling and can handle any size of array.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 */
void isingCudaGenGraph(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k);