#include <iostream>
#include <cooperative_groups.h>
#include "cudaBlocks.cuh"

__global__ void isingModelBlocks(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter, bool *allBlocksFinished)
{
    size_t n2 = n * n;
    size_t start = blockIdx.x * blockChunk;
    size_t end = start + blockChunk;
    if (end > n2)
        end = n2;

    cooperative_groups::grid_group g = cooperative_groups::this_grid(); // used for the inter-block communication

    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = start; i < end; i++)
        {
            size_t up = (i - n + n2) % n2;
            size_t down = (i + n) % n2;
            size_t row = i / n;
            size_t left = row * n + (i - 1 + n) % n;
            size_t right = row * n + (i + 1) % n;
            uint8_t sum = in[i] + in[up] + in[down] + in[left] + in[right];
            out[i] = sum > 2; // assign the majority
        }

        // sync the running blocks before swapping the pointers
        g.sync();

        /* Without cooperative groups version

        atomicAdd(blockCounter, 1); // this block has finished
        __threadfence();            // ensure that threads reading the value of blockCounter from now on cannot see the previous value

        *allBlocksFinished = false;
        __threadfence();
        while (!(*allBlocksFinished))
        {
            __threadfence(); // rest of the blocks load the new value of allBlocksFinished

            if (*blockCounter == gridDim.x)
            {
                *allBlocksFinished = true;
                __threadfence(); // update the value of allBlocksFinished
            }
        }
        *blockCounter = 0; // re-set this block's value to 0
        __threadfence();

        */

        // swap the pointers
        uint8_t *temp = in;
        in = out;
        out = temp;
    }
}

void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks)
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

    uint32_t blockChunk;
    if (blocks > n2)
    {
        std::cout << "No need for that many blocks. Using " << n2 << " blocks" << std::endl;
        blocks = n2;
        blockChunk = 1;
    }
    else
    {
        blockChunk = n2 / blocks;                         // number of elements each block will process
        blocks = (uint32_t)ceil((double)n2 / blockChunk); // the actual number of blocks may change but the total number of elements
                                                          // processed per block will be as expected
    }

    if (blocks > MAX_BLOCKS)
    {
        std::cout << "Error: too many blocks. Using " << MAX_BLOCKS << " blocks" << std::endl;
        blocks = MAX_BLOCKS;
        blockChunk = (uint32_t)ceil((double)n2 / blocks);
    }

    // Allocate memory for the flag that indicates if all blocks have finished
    bool *allBlocksFinished;
    error = cudaMalloc((void **)&allBlocksFinished, sizeof(bool));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of allBlocksFinished failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }
    error = cudaMemset(allBlocksFinished, false, sizeof(bool));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memset of allBlocksFinished failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }

    // Set arguments for the kernel
    size_t n = (size_t)sqrt(n2);
    void *kernelArgs[] = {&d_out, &d_in, &n, (void *)&k, &blockChunk, &blockCounter, &allBlocksFinished};

    // Launch the kernel
    error = cudaLaunchCooperativeKernel((void *)isingModelBlocks, blocks, 1, (void **)kernelArgs);

    /* Or if your device doesn't support cooperative groups

    isingModelBlocks<<<blocks, 1>>>(d_out, d_in, (size_t)sqrt(n2), k, blockChunk, blockCounter, allBlocksFinished);
    error = cudaGetLastError(); // Since no error was returned from all the previous cuda calls,
                                // the last error must be from the kernel launch

    */

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

    // Copy the output back to the host
    if (k % 2 == 0)
        error = cudaMemcpy(out.data(), d_in, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    else
        error = cudaMemcpy(out.data(), d_out, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
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
    cudaFree(allBlocksFinished);
}