#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define MAX_BLOCKS 2147483647

using funcP = size_t (*)(size_t, size_t);

/* Kernel function: Each thread computes the evolution of a block of elements
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           n(input): single dimension of the array
 *           k(input): number of iterations
 *           blockChunk(input): number of elements in a block
 *           blockCounter(input): number of blocks that have been executed (used for block syncing)
 *           allBlocksFinished(input): flag to indicate if all blocks have been executed (used for block syncing)
 */
__global__ void isingModelBlocks(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter, bool *allBlocksFinished);

/* Wrapper function for the kernel
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array, it must use resources below the constraints of cooperative groups
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 */
void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks);

/* Kernel function: Each thread adds the neighbouring values to an element for a block of elements.
 * @params:  out(output): sum of neighbours of each element of the array after 1 iteration
 *           in(input): input array
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 *           calcOffset(input): function to calculate the offset of the output array elements to compute
 */
__global__ void addValue(uint8_t *out, const uint8_t *in, const size_t n, const uint32_t blockChunk);

/* Kernel function: Each thread assigns the final value of a block of elements to the next iteration's input.
 * @params:  out(output): cleared output array
 *           in(input): state of the array after 1 iteration
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 */
__global__ void assignClearValue(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk);

/* Wrapper function for the kernel. This version can handle any number of blocks.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 */
void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks);

/* Wrapper function for the kernel. This version contains a graph with node depedencies to improve scheduling and can handle any number of blocks.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 */
void isingCudaGenGraph(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks);

/* Kernel function: Each thread adds a neighbouring value to an element for a block of elements.
 * @params:  out(output): sum of neighbours of each element of the array after 1 iteration
 *           in(input): input array
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 *           calcOffset(input): function to calculate the offset of the output array elements to compute
 */
__global__ void addValueStreams(uint8_t *out, const uint8_t *in, const size_t n, const uint32_t blockChunk, const funcP calcOffset);

/* Kernel function: Each thread assigns the final value of a block of elements to the next iteration's input.
 * @params:  out(output): cleared output array
 *           in(input): state of the array after 1 iteration
 *           n(input): single dimension of the array
 *           blockChunk(input): number of elements in a block
 */
__global__ void assignClearValueStreams(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk);

/* Wrapper function for the kernel. This version contains streams to add on the parallelism and can handle any any number of blocks.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 */
void isingCudaGenStreams(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks);

/* Wrapper function for the kernel. This version is a graph representation of the streams version.
 * @params:  out(output): state of the array after k iterations
 *           in(input): input array
 *           k(input): number of iterations
 *           blocks(input): number of blocks to be executed
 */
void isingCudaGenGraphStreams(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks);
