#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define MAX_THREADS_PER_BLOCK 1024

#define MAX_SHARED_PER_BLOCK 49152

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk);

void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const size_t n, const uint32_t k, const uint32_t blocks, const uint32_t threads);