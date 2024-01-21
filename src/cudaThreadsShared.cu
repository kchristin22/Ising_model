#include <iostream>
#include "cudaThreadsShared.cuh"

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k, const uint32_t blockChunk, uint32_t *blockCounter)
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

    if (blockChunkStart == 0)
        blockChunkStart = end; // needed for the indexing of the shared memory

    memcpy(&s_in[start % blockChunkStart], &in[start], sizeof(uint8_t) * (end - start)); // needed for the inter-block communication

    __syncthreads(); // ensure that all threads have finished copying

    // printf("blockChunkStart: %ld, start: %ld, end: %ld\n", blockChunkStart, start, end);
    printf("blockIdx.x: %d, threadIdx.x: %d, start: %ld, end: %ld\n", blockIdx.x, threadIdx.x, start, end);

    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = start; i < end; i++)
        {
            size_t up = (i - n + n2) % n2;
            // if (i == 33)
            // printf("up: %ld, in[%ld] = %d\n", up, up, in[up]);
            uint8_t in_up = up >= start && up < end ? s_in[up % blockChunkStart] : in[up];
            // if (i == 33)
            // printf("up >= start && up < end = %d, in_up: %d, in[up]: %d\n", up >= start && up < end, in_up, in[up]);
            size_t down = (i + n) % n2;
            uint8_t in_down = down >= start && down < end ? s_in[down % blockChunkStart] : in[down];
            size_t row = i / n;
            size_t left = row * n + (i - 1 + n) % n;
            uint8_t in_left = left >= start && left < end ? s_in[left % blockChunkStart] : in[left];
            size_t right = row * n + (i + 1) % n;
            uint8_t in_right = right >= start && right < end ? s_in[right % blockChunkStart] : in[right];
            if (i == 33)
                printf("up: %d, down: %d, left: %d, right: %d\n", in_up, in_down, in_left, in_right);
            uint8_t sum = s_in[i % blockChunkStart] + in_up + in_down + in_left + in_right;
            if (i == 33)
                printf("sum: %d\n", sum);
            s_out[i % blockChunkStart] = sum > 2; // assign the majority
        }

        // ensure that all threads have finished reading before copying
        if (threadIdx.x == 0) // each block has at least one thread
        {
            atomicAdd(blockCounter, 1); // this block has finished
            __threadfence();            // ensure that threads reading the value of blockCounter from now on cannot see the previous value
            // printf("blockCounter: %d, gridDim.x: %d\n", *blockCounter, gridDim.x);

            while (*blockCounter < gridDim.x && *blockCounter != 0)
                ; // if blockCounter is 0, then all blocks have finished and one has initialized the counter to 0

            *blockCounter = 0;
            // printf("blockCounter set to zero\n");
        }
        __syncthreads(); // other threads wait for thread 0 to finish

        memcpy(&in[start], &s_out[start % blockChunkStart], sizeof(uint8_t) * (end - start)); // needed for the inter-block communication

        // ensure that all threads have finished reading before copying
        if (threadIdx.x == 0) // each block has at least one thread
        {
            atomicAdd(blockCounter, 1); // this block has finished
            __threadfence();            // ensure that threads reading the value of blockCounter from now on cannot see the previous value

            while (*blockCounter < gridDim.x && *blockCounter != 0)
                ; // if blockCounter is 0, then all blocks have finished and one has initialized the counter to 0

            *blockCounter = 0;
        }

        __syncthreads(); // other threads wait for thread 0 to finish

        // swap the pointers
        uint8_t *temp = s_in;
        s_in = s_out;
        s_out = temp;

        __syncthreads(); // ensure that all threads have finished swapping
    }
    // for (size_t i = start; i < end; i++)
    // printf("in[%ld]: %d\n", i, in[i]);

    memcpy(&out[start], &s_in[start % blockChunkStart], sizeof(uint8_t) * (end - start)); // copy the result to the global memory (the last swap is not needed)
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

    if (blocks > MAX_BLOCKS)
    {
        std::cout << "Error: too many blocks. Using " << MAX_BLOCKS << " blocks" << std::endl;
        blocks = MAX_BLOCKS;
        blockChunk = (uint32_t)ceil((double)n2 / blocks);
    }
    if (blockChunk > MAX_SHARED_PER_BLOCK)
    {
        std::cout << "Error: too many elements per block. Use more blocks." << std::endl;
        return;
    }

    printf("blocks: %d, blockChunk: %d\n", blocks, blockChunk);

    if (threads > MAX_THREADS_PER_BLOCK)
    {
        std::cout << "Error: too many threads per block. Using " << MAX_THREADS_PER_BLOCK << " threads per block" << std::endl;
        threads = MAX_THREADS_PER_BLOCK;
    }

    // Launch the kernel
    isingModel<<<blocks, threads>>>(d_out, d_in, (size_t)sqrt(n2), k, blockChunk, blockCounter);
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
    std::cout << "Kernel finished" << std::endl;

    // Copy the output back to the host
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