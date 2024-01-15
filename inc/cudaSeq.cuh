#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k);

void isingCudaSeq(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const size_t n, const uint32_t k, const uint32_t kThread);