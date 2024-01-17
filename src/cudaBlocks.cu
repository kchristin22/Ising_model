#include <iostream>
#include "cudaBlocks.cuh"

__global__ void isingModelBlocks(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk)
{
    size_t n2 = n * n;
    size_t start = blockIdx.x * blockChunk;
    size_t end = start + blockChunk;
    if (end > n2)
        end = n2;

    // printf("gridDim.x: %d, blockIdx.x: %d, start: %ld, end: %ld\n", gridDim.x, blockIdx.x, start, end);

    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = start; i < end; i++)
        {
            uint8_t sum = in[i] + in[(i + n) % n2] + in[(i - n + n2) % n2] + in[(i + 1) % n2] + in[(i - 1 + n2) % n2];
            out[i] = sum > 2; // if majority is true (sum in [3,5]), out is true
        }

        memcpy(&in[start], &out[start], sizeof(uint8_t) * (end - start));
        __syncthreads();
    }
}

void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const size_t n, const uint32_t k, const uint32_t blocks)
{
    // check if in vector has the right dimensions
    if (in.size() != n * n)
    {
        std::cout << "Error: input vector has wrong dimensions" << std::endl;
        return;
    }
    out.resize(n * n);

    // Allocate memory on the device
    uint8_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, n * n * sizeof(uint8_t));
    cudaMalloc((void **)&d_out, n * n * sizeof(uint8_t));

    // Copy the input to the device
    cudaMemcpy(d_in, in.data(), n * n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    size_t n2 = n * n;
    uint32_t blockChunk = n2 / blocks;
    uint32_t blockNum = blocks * blockChunk == n2 ? blocks : blocks + 1;

    // Launch the kernel
    isingModelBlocks<<<blockNum, 1>>>(d_out, d_in, n, k, blockChunk);
    cudaDeviceSynchronize();

    // Copy the output back to the host
    cudaMemcpy(out.data(), d_out, n * n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_in);
    cudaFree(d_out);
}