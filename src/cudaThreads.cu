#include <iostream>
#include "cudaThreads.cuh"

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter)
{
    size_t n2 = n * n;
    size_t threadChunk = blockChunk / blockDim.x;
    size_t start = blockIdx.x * blockChunk + threadIdx.x * threadChunk;
    size_t end = threadIdx.x == blockDim.x - 1 ? start + threadChunk + (blockChunk - blockDim.x * threadChunk) : start + threadChunk;
    if (end > n2)
        end = n2;

    // printf("gridDim.x: %d, blockIdx.x: %d, threadChunk: %d, start: %ld, end: %ld\n", gridDim.x, blockIdx.x, threadChunk, start, end);
    printf("threadIdx.x: %d, gridDim.x: %d, start: %ld, end: %ld\n", threadIdx.x, gridDim.x, start, end);

    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = start; i < end; i++)
        {
            uint8_t sum = in[i] + in[(i + n) % n2] + in[(i - n + n2) % n2] + in[(i + 1) % n] + in[(i - 1 + n) % n]; // operate in shared, sync tricky
            out[i] = sum > 2;                                                                                       // if majority is true (sum in [3,5]), out is true
        }

        if (threadIdx.x == 0)
        {
            atomicAdd(blockCounter, 1);

            while (*blockCounter < gridDim.x && *blockCounter != 0) // if blockCounter is 0, then all blocks have finished and one has initialized the counter to 0
                __threadfence_block();                              // Ensure I have the latest value of blockCounter

            *blockCounter = 0;
        }
        __syncthreads();
        memcpy(&in[start], &out[start], sizeof(uint8_t) * (end - start));
    }
}

void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const size_t n, const uint32_t k, uint32_t blocks, uint32_t threads)
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

    uint32_t *blockCounter;
    cudaMalloc((void **)&blockCounter, sizeof(uint32_t));
    cudaMemset(&blockCounter, 0, sizeof(uint32_t));

    // Copy the input to the device
    cudaMemcpy(d_in, in.data(), n * n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    size_t n2 = n * n;
    uint32_t blockChunk = n2 / blocks;
    blocks = blocks * blockChunk == n2 ? blocks : blocks++;

    if (threads > MAX_THREADS_PER_BLOCK)
    {
        std::cout << "Error: too many threads per block. Using 1024 threads per block" << std::endl;
        threads = MAX_THREADS_PER_BLOCK;
    }

    // Launch the kernel
    isingModel<<<blocks, threads>>>(d_out, d_in, n, k, blockChunk, blockCounter);
    cudaDeviceSynchronize();

    // Copy the output back to the host
    cudaMemcpy(out.data(), d_out, n * n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(blockCounter);
}