#include <iostream>
#include <cmath>
#include "cudaThreads.cuh"

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, uint32_t *blockCounter, uint8_t *allBlocksFinished)
{
    size_t n2 = n * n;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n2) // more threads than elements may have been created
        return;

    size_t up = (index - n + n2) % n2;
    size_t down = (index + n) % n2;
    size_t row = index / n;
    size_t left = row * n + (index - 1 + n) % n;
    size_t right = row * n + (index + 1) % n;

    for (size_t iter = 0; iter < k; iter++)
    {
        uint8_t sum = in[index] + in[up] + in[down] + in[left] + in[right];
        out[index] = sum > 2; // assign the majority

        // sync the running blocks before swapping the pointers
        if (threadIdx.x == 0) // each block has at least one thread
        {
            blockCounter[blockIdx.x] = 1; // this block has finished
            __threadfence();              // ensure that threads reading the value of blockCounter from now on cannot see the previous value

            *allBlocksFinished = 0;
            __threadfence();
            while (!(*allBlocksFinished))
            {
                if (blockIdx.x == 0) // there's at least one block
                {
                    for (size_t i = 0; i < gridDim.x; i++)
                    {
                        if (blockCounter[i] == 0)
                            break;

                        if (i == gridDim.x - 1)
                        {
                            *allBlocksFinished = 1;
                            __threadfence(); // update the value of allBlocksFinished
                        }
                    }
                }
                __threadfence(); // rest of the blocks load the new value of allBlocksFinished
            }
            blockCounter[blockIdx.x] = 0; // re-set this block's value to 0
            __threadfence();
        }
        __syncthreads(); // other threads of the block wait for thread 0

        // swap input and output
        uint8_t *temp = in;
        in = out;
        out = temp;
    }
}

void isingCuda(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k)
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

    // Copy the input from CPU to the device
    error = cudaMemcpy(d_in, in.data(), n2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of d_in failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Calculate the number of blocks and threads needed to assign a single element to each thread
    uint32_t blocks = (uint32_t)ceil((double)n2 / MAX_THREADS_PER_BLOCK);
    if (blocks > MAX_BLOCKS)
    {
        std::cout << "Error: too many blocks needed for this input array" << std::endl;
        return;
    }
    uint32_t threads = (uint32_t)ceil((double)n2 / blocks); // distribute the elements evenly among the blocks

    uint32_t *blockCounter; // used to sync the blocks
    error = cudaMalloc((void **)&blockCounter, blocks * sizeof(uint32_t));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Malloc of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }
    error = cudaMemset(blockCounter, 0, blocks * sizeof(uint32_t)); // initialize block counter to 0
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memset of blockCounter failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
    }

    // Allocate memory for the flag that indicates if all blocks have finished
    uint8_t *allBlocksFinished;
    cudaMalloc((void **)&allBlocksFinished, sizeof(uint8_t));
    cudaMemset(allBlocksFinished, 0, sizeof(uint8_t));

    // Run the kernel
    isingModel<<<blocks, threads>>>(d_out, d_in, (size_t)sqrt(n2), k, blockCounter, allBlocksFinished);
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
}