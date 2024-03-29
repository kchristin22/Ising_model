#include <iostream>
#include <cooperative_groups.h>
#include "cudaThreadsShared.cuh"

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter, bool *allBlocksFinished)
{
    size_t n2 = n * n;
    // elements per thread
    size_t threadChunk = blockChunk / blockDim.x; // not ceil to ensure that the total number of elements processed per block is not greater than blockChunk
    // offset in the global memory compared to the shared memory
    size_t blockStart = blockIdx.x * blockChunk; // needed for the indexing of the shared memory
    // start and end indices of the elements that this thread will process (in the global memory)
    size_t start = blockStart + threadIdx.x * threadChunk;
    size_t end = threadIdx.x == blockDim.x - 1 ? start + threadChunk + (blockChunk - blockDim.x * threadChunk) : start + threadChunk; // last thread of the block processes the remaining elements

    if (end > n2)
        end = n2;

    // size of s is 2 * blockChunk bytes, configured in the kernel launch
    extern __shared__ uint8_t s[]; // max shared memory macro is defined in bytes which is the size of each element
    uint8_t *s_in = s;
    uint8_t *s_out = &s[blockChunk]; // s_in has blockChunk elements

    // load from the global memory to the shared memory
    memcpy(&s_in[start - blockStart], &in[start], sizeof(uint8_t) * (end - start));

    __syncthreads(); // ensure that all threads have finished copying

    cooperative_groups::grid_group g = cooperative_groups::this_grid(); // used for the inter-block communication

    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = start; i < end; i++)
        {
            size_t up = (i - n + n2) % n2;
            uint8_t in_up = up >= blockStart && up < (blockStart + blockChunk) ? s_in[up - blockStart] : in[up]; // if the element is in the shared memory, read it from there
            size_t down = (i + n) % n2;
            uint8_t in_down = down >= blockStart && down < (blockStart + blockChunk) ? s_in[down - blockStart] : in[down];
            size_t row = i / n;
            size_t left = row * n + (i - 1 + n) % n;
            uint8_t in_left = left >= blockStart && left < (blockStart + blockChunk) ? s_in[left - blockStart] : in[left];
            size_t right = row * n + (i + 1) % n;
            uint8_t in_right = right >= blockStart && right < (blockStart + blockChunk) ? s_in[right - blockStart] : in[right];
            uint8_t sum = s_in[i - blockStart] + in_up + in_down + in_left + in_right;
            s_out[i - blockStart] = sum > 2; // assign the majority
        }

        // copy to output vector to avoid updating the in pointer before all threads have finished reading from it (and avoid the need for synchronization)
        memcpy(&out[start], &s_out[start - blockStart], sizeof(uint8_t) * (end - start)); // needed for the inter-block communication

        // ensure that all threads have finished copying to global memory
        g.sync();

        /* Without cooperative groups version

        if (threadIdx.x == 0) // each block has at least one thread
        {
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
        }
        __syncthreads(); // other threads wait for thread 0 to finish

        */

        // swap the pointers
        uint8_t *temp = s_in;
        s_in = s_out;
        s_out = temp;

        temp = in;
        in = out;
        out = temp;
    }
}

void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks, uint32_t threads)
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
    cudaError_t error = cudaMallocAsync((void **)&d_in, n2 * sizeof(uint8_t), 0);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of d_in failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    error = cudaMallocAsync((void **)&d_out, n2 * sizeof(uint8_t), 0);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of d_out failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Copy the input from CPU to the device
    error = cudaMemcpyAsync(d_in, in.data(), n2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
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
    if (2 * blockChunk > MAX_SHARED_PER_BLOCK)
    {
        std::cout << "Error: too many elements per block. Use more blocks." << std::endl;
        return;
    }

    uint32_t *blockCounter; // used to sync the blocks
    error = cudaMallocAsync((void **)&blockCounter, sizeof(uint32_t), 0);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }
    error = cudaMemsetAsync(blockCounter, 0, sizeof(uint32_t)); // initialize block counter to 0
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memset of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }

    if (threads > MAX_THREADS_PER_BLOCK)
    {
        std::cout << "Error: too many threads per block. Using " << MAX_THREADS_PER_BLOCK << " threads per block" << std::endl;
        threads = MAX_THREADS_PER_BLOCK;
    }

    // Allocate memory for the flag that indicates if all blocks have finished
    bool *allBlocksFinished;
    error = cudaMallocAsync((void **)&allBlocksFinished, sizeof(bool), 0);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of allBlocksFinished failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }
    error = cudaMemsetAsync(allBlocksFinished, false, sizeof(bool));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memset of allBlocksFinished failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }

    // Set arguments for the kernel
    size_t n = (size_t)sqrt(n2);
    void *kernelArgs[] = {&d_out, &d_in, &n, (void *)&k, &blockChunk, &blockCounter, &allBlocksFinished};

    // Launch the kernel
    error = cudaLaunchCooperativeKernel((void *)isingModel, blocks, threads, (void **)kernelArgs, blockChunk * 2 * sizeof(uint8_t));

    /* Or if your device doesn't support cooperative groups

    isingModel<<<blocks, threads, blockChunk * 2 * sizeof(uint8_t)>>>(d_out, d_in, (size_t)sqrt(n2), k, blockChunk, blockCounter, allBlocksFinished);
    error = cudaGetLastError(); // Since no error was returned from all the previous cuda calls,
                                // the last error must be from the kernel launch

    */

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }

    // Wait for the kernel to finish to avoid exiting the program prematurely
    error = cudaStreamSynchronize(0);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Copy the output back to the host
    if (k % 2 == 0)
        error = cudaMemcpyAsync(out.data(), d_in, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    else
        error = cudaMemcpyAsync(out.data(), d_out, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of device's output to host failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    // Free the memory on the device
    cudaFreeAsync(d_in, 0);
    cudaFreeAsync(d_out, 0);
    cudaFreeAsync(blockCounter, 0);
    cudaFreeAsync(allBlocksFinished, 0);
}