#include <iostream>
#include "cudaThreadsSharedGen.cuh"

__global__ void isingModelGen(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk, uint32_t *blockCounter)
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
    memcpy(&s_in[start - blockStart], &in[start], sizeof(uint8_t) * (end - start)); // needed for the inter-block communication

    __syncthreads(); // ensure that all threads have finished copying

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

    memcpy(&out[start], &s_out[start - blockStart], sizeof(uint8_t) * (end - start)); // copy the result to the global memory (the last swap is not needed)
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

    if (threads > MAX_THREADS_PER_BLOCK)
    {
        std::cout << "Error: too many threads per block. Using " << MAX_THREADS_PER_BLOCK << " threads per block" << std::endl;
        threads = MAX_THREADS_PER_BLOCK;
    }

    for (size_t iter = 0; iter < k; iter++)
    {
        // Launch the kernel
        isingModelGen<<<blocks, threads, 2 * blockChunk * sizeof(uint8_t)>>>(d_out, d_in, (size_t)sqrt(n2), blockChunk, blockCounter);
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