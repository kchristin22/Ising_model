#include <iostream>
#include <cooperative_groups.h>
#include "cudaBlocks.cuh"

__device__ inline size_t upOffset(size_t n, size_t i)
{
    return (i - n + n * n) % (n * n);
}

__device__ inline size_t downOffset(size_t n, size_t i)
{
    return (i + n) % (n * n);
}

__device__ inline size_t leftOffset(size_t n, size_t i)
{
    return i / n * n + (i - 1 + n) % n;
}

__device__ inline size_t rightOffset(size_t n, size_t i)
{
    return i / n * n + (i + 1) % n;
}

__device__ inline size_t centerOffset(size_t n, size_t i)
{
    return i;
}

__device__ funcP upP = upOffset;
__device__ funcP downP = downOffset;
__device__ funcP leftP = leftOffset;
__device__ funcP rightP = rightOffset;
__device__ funcP centerP = centerOffset;

// auxiliary function for atomicAdd on uint8_t
__device__ static inline char atomicAdd(char *address, char val)
{
    // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
    size_t long_address_modulo = (size_t)address & 3;
    // the 32-bit address that overlaps the same memory
    auto *base_address = (unsigned int *)((char *)address - long_address_modulo);
    // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
    // The "4" signifies the position where the first byte of the second argument will end up in the output.
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;

    long_old = *base_address;

    do
    {
        long_assumed = long_old;
        // replace bits in long_old that pertain to the char address with those from val
        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}

__global__ void addValue(uint8_t *out, const uint8_t *in, const size_t n, const uint32_t blockChunk, const funcP calcOffset)
{
    size_t n2 = n * n;
    size_t start = blockIdx.x * blockChunk;
    size_t end = start + blockChunk;
    if (end > n2)
        end = n2;

    for (size_t i = start; i < end; i++)
    {
        size_t offset = (*calcOffset)(n, i); // alocate memory for calcOffset
        atomicAdd((char *)&out[offset], (uint8_t)in[i]);
        __threadfence();
    }
}

__global__ void assignClearValue(uint8_t *out, uint8_t *in, const size_t n, const uint32_t blockChunk)
{
    size_t n2 = n * n;
    size_t start = blockIdx.x * blockChunk;
    size_t end = start + blockChunk;
    if (end > n2)
        end = n2;

    for (size_t i = start; i < end; i++)
    {
        out[i] = out[i] > 2;
        in[i] = 0;
    }
}

void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k, uint32_t blocks)
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

    // Set arguments for the kernel
    size_t n = (size_t)sqrt(n2);

    cudaStream_t up, down, left, right, center;
    cudaStreamCreate(&up);
    cudaStreamCreate(&down);
    cudaStreamCreate(&left);
    cudaStreamCreate(&right);
    cudaStreamCreate(&center);

    funcP upFunc, downFunc, leftFunc, rightFunc, centerFunc;

    error = cudaMemcpyFromSymbol(&upFunc, upP, sizeof(funcP));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of upFunc failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    error = cudaMemcpyFromSymbol(&downFunc, downP, sizeof(funcP));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of downFunc failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    error = cudaMemcpyFromSymbol(&leftFunc, leftP, sizeof(funcP));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of leftFunc failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    error = cudaMemcpyFromSymbol(&rightFunc, rightP, sizeof(funcP));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of rightFunc failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }
    error = cudaMemcpyFromSymbol(&centerFunc, centerP, sizeof(funcP));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of centerFunc failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    for (size_t iter = 0; iter < k; iter++)
    {
        // Launch the kernel
        addValue<<<blocks, 1, 0, up>>>(d_out, d_in, n, blockChunk, upFunc);
        addValue<<<blocks, 1, 0, down>>>(d_out, d_in, n, blockChunk, downFunc);
        addValue<<<blocks, 1, 0, left>>>(d_out, d_in, n, blockChunk, leftFunc);
        addValue<<<blocks, 1, 0, right>>>(d_out, d_in, n, blockChunk, rightFunc);
        addValue<<<blocks, 1, 0, center>>>(d_out, d_in, n, blockChunk, centerFunc);
        error = cudaGetLastError(); // Since no error was returned from all the previous cuda calls,
                                    // the last error must be from the kernel launches

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
            printf("Error: %d\n", error);
        }

        assignClearValue<<<blocks, 1>>>(d_out, d_in, n, blockChunk); // no sync needed because default stream is synchronous to the others
        error = cudaGetLastError();                                  // Since no error was returned from all the previous cuda calls,
        // the last error must be from the kernel launches
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
            printf("Error: %d\n", error);
        }

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
            printf("Error: %d\n", error);
        }

        // Wait for the default stream to finish to launch next iteration
        error = cudaStreamSynchronize(0);
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
    }

    cudaStreamDestroy(up);
    cudaStreamDestroy(down);
    cudaStreamDestroy(left);
    cudaStreamDestroy(right);
    cudaStreamDestroy(center);

    // Copy the output back to the host
    error = cudaMemcpy(out.data(), d_in, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of device's output to host failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Free the memory on the device
    cudaFree(d_in);
    cudaFree(d_out);
}