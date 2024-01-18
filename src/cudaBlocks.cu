#include <iostream>
#include "cudaBlocks.cuh"

__global__ void isingModelBlocks(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter)
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
            uint8_t sum = in[i] + in[(i + n) % n2] + in[(i - n + n2) % n2] + in[(i + 1) % n] + in[(i - 1 + n) % n];
            out[i] = sum > 2; // if majority is true (sum in [3,5]), out is true
        }

        atomicAdd(blockCounter, 1);

        while (*blockCounter < gridDim.x && *blockCounter != 0) // if blockCounter is 0, then all blocks have finished and one has initialized the counter to 0
            __threadfence_block();                              // Ensure I have the latest value of blockCounter
        blockCounter = 0;

        memcpy(&in[start], &out[start], sizeof(uint8_t) * (end - start));
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
    cudaError_t error = cudaMalloc((void **)&d_in, n * n * sizeof(uint8_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of d_in failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        // Additional error handling if needed
    }
    error = cudaMalloc((void **)&d_out, n * n * sizeof(uint8_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of d_out failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        // Additional error handling if needed
    }

    uint32_t *blockCounter;
    error = cudaMalloc((void **)&blockCounter, sizeof(uint32_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        // Additional error handling if needed
    }
    error = cudaMemset(blockCounter, 0, sizeof(uint32_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memset of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        // Additional error handling if needed
    }
    

    // Copy the input to the device
    error = cudaMemcpy(d_in, in.data(), n * n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of d_in failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        // Additional error handling if needed
    }

    size_t n2 = n * n;
    uint32_t blockChunk = n2 / blocks;
    uint32_t blockNum = blocks * blockChunk == n2 ? blocks : blocks + 1;

    // Launch the kernel
    isingModelBlocks<<<blockNum, 1>>>(d_out, d_in, n, k, blockChunk, blockCounter);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        printf("Error: %d\n", cudaStatus);
        // Additional error handling if needed
    }
    cudaDeviceSynchronize();

    // Copy the output back to the host
    cudaMemcpy(out.data(), d_out, n * n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(blockCounter);
}