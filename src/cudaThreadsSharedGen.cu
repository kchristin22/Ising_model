#include <iostream>
#include "cudaThreadsShared.cuh"

__global__ void isingModelGen(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk, uint32_t *blockCounter)
{
    size_t n2 = n * n;
    size_t threadChunk = blockChunk / blockDim.x; // not ceil to ensure that the total number of elements processed per block is not greater than blockChunk
    size_t blockChunkStart = blockIdx.x * blockChunk;
    size_t start = blockChunkStart + threadIdx.x * threadChunk;
    size_t end = threadIdx.x == blockDim.x - 1 ? start + threadChunk + (blockChunk - blockDim.x * threadChunk) : start + threadChunk; // last thread of the block processes the remaining elements
    if (end > n2)
        end = n2;

    __shared__ uint8_t s[MAX_SHARED_PER_BLOCK]; // max shared memory macro is defined in bytes which is the size of each element
    uint8_t *s_in = s;
    uint8_t *s_out = &s[blockChunk]; // s_in has blockChunk elements

    // printf("blockChunk: %d, start: %ld, end: %ld\n", blockChunk, start, end);
    // printf("blockChunkStart: %ld\n", blockChunkStart);

    if (blockChunkStart == 0)
        blockChunkStart = end; // needed for the indexing of the shared memory
    // printf("start: %ld, blockChunkStart: %ld, start mod blockChunkStart: %ld\n", start, blockChunkStart, start % blockChunkStart);

    memcpy(&s_in[start % blockChunkStart], &in[start], sizeof(uint8_t) * (end - start)); // needed for the inter-block communication

    __syncthreads(); // other threads of the block wait for thread 0

    for (size_t i = start; i < end; i++)
    {
        // printf("up: %d, down: %d, left: %d, right: %d\n", (i - n + n2) % n2, (i + n) % n2, (i / n * n) + (i - 1 + n) % n2, (i / n * n) + (i + 1) % n2);
        size_t up = (i - n + n2) % n2;
        size_t in_up = up >= start && up < end ? s_in[up % blockChunkStart] : in[up];
        size_t down = (i + n) % n2;
        size_t in_down = down >= start && down < end ? s_in[down % blockChunkStart] : in[down];
        size_t row = i / n;
        size_t left = row * n + (i - 1 + n) % n;
        size_t in_left = left >= start && left < end ? s_in[left % blockChunkStart] : in[left];
        size_t right = row * n + (i + 1) % n;
        size_t in_right = right >= start && right < end ? s_in[right % blockChunkStart] : in[right];
        uint8_t sum = s_in[i % blockChunkStart] + in_up + in_down + in_left + in_right;
        s_out[i % blockChunkStart] = sum > 2; // assign the majority
    }

    memcpy(&out[start], &s_out[start % blockChunkStart], sizeof(uint8_t) * (end - start)); // copy the result to the global memory (the last swap is not needed)
}

void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks, uint32_t threads)
{
    size_t n2 = in.size();
    // check if `in` vector has a perfect square size
    if (ceil(sqrt(n2)) != floor(sqrt(n2)))
    {
        std::cout << "Error: input vector has wrong dimensions" << std::endl;
        return;
    }
    out.resize(n2);

    // Allocate memory on the device (GPU)
    uint8_t *d_in, *d_out;
    cudaError_t error = cudaMalloc((void **)&d_in, n2 * sizeof(uint8_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of d_in failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    error = cudaMalloc((void **)&d_out, n2 * sizeof(uint8_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of d_out failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    uint32_t *blockCounter; // used to sync the blocks
    error = cudaMalloc((void **)&blockCounter, sizeof(uint32_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }
    error = cudaMemset(blockCounter, 0, sizeof(uint32_t)); // initialize block counter to 0
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memset of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }

    // Copy the input from CPU to the device
    error = cudaMemcpy(d_in, in.data(), n2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of d_in failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    uint32_t blockChunk = n2 / blocks;                // number of elements each block will process
    blocks = (uint32_t)ceil((double)n2 / blockChunk); // the actual number of blocks may change but the total number of elements
                                                      // processed per block will be as expected

    if (blockChunk > MAX_SHARED_PER_BLOCK)
    {
        std::cout << "Error: too many elements per block. Use more blocks." << std::endl;
        return;
    }

    if (threads > MAX_THREADS_PER_BLOCK)
    {
        std::cout << "Error: too many threads per block. Using " << MAX_THREADS_PER_BLOCK << " threads per block" << std::endl;
        threads = MAX_THREADS_PER_BLOCK;
    }

    for (size_t iter = 0; iter < k; iter++)
    {
        // Launch the kernel
        isingModelGen<<<blocks, threads>>>(d_out, d_in, (size_t)sqrt(n2), blockChunk, blockCounter);
        error = cudaGetLastError(); // Since no error was returned from all the previous cuda calls,
                                    // the last error must be from the kernel launch
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
            printf("Error: %d\n", error);
        }

        // Wait for the kernel to finish to avoid exiting the program prematurely
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(error));
            printf("Error: %d\n", error);
            return;
        }

        // swap the pointers
        uint8_t *temp = d_in;
        d_in = d_out;
        d_out = temp;

        // printf("d_out: %p, d_in: %p\n", d_out, d_in);
    }

    // Copy the output back to the host
    error = cudaMemcpy(out.data(), d_in, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost); // the last swap is not needed, so d_in is the final result
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of device's output to host failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    // Free the memory on the device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(blockCounter);
}