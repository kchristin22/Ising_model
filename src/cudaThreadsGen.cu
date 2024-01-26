#include <iostream>
#include <cmath>
#include <cooperative_groups.h>
#include "cudaThreads.cuh"

__global__ void addValue(uint8_t *out, const uint8_t *in, const size_t n)
{
    size_t n2 = n * n;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n2)
        return;

    size_t up = (index - n + n2) % n2;
    size_t down = (index + n) % n2;
    size_t row = index / n;
    size_t left = row * n + (index - 1 + n) % n;
    size_t right = row * n + (index + 1) % n;
    out[index] = (in[index] + in[up] + in[down] + in[left] + in[right]) > 2;
}

__global__ void assignClearValue(uint8_t *out, uint8_t *in, const size_t n)
{
    size_t n2 = n * n;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n2)
        return;

    in[index] = out[index]; // swap the pointers and assign the final value of this iteration
}

void isingCudaGen(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k)
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

    // Calculate the number of blocks and threads needed to assign a single element to each thread
    uint32_t blocks = (uint32_t)ceil((double)n2 / MAX_THREADS_PER_BLOCK);
    if (blocks > MAX_BLOCKS)
    {
        std::cout << "Error: too many blocks needed for this input array" << std::endl;
        return;
    }
    uint32_t threads = (uint32_t)ceil((double)n2 / blocks); // distribute the elements evenly among the blocks

    // Set arguments for the kernel
    size_t n = (size_t)sqrt(n2);

    for (size_t iter = 0; iter < k; iter++)
    {
        // Launch the kernel
        addValue<<<blocks, threads>>>(d_out, d_in, n);
        error = cudaGetLastError(); // Since no error was returned from all the previous cuda calls,
                                    // the last error must be from the kernel launches

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
            printf("Error: %d\n", error);
        }

        // Swap the pointers to prepare for the next iteration
        uint8_t *temp = d_in;
        d_in = d_out;
        d_out = temp;
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
    error = cudaMemcpyAsync(out.data(), d_in, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost); // output is in d_in
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of device's output to host failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Free the memory on the device
    cudaFreeAsync(d_in, 0);
    cudaFreeAsync(d_out, 0);
}

void isingCudaGenGraph(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k)
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

    // Calculate the number of blocks and threads needed to assign a single element to each thread
    uint32_t blocks = (uint32_t)ceil((double)n2 / MAX_THREADS_PER_BLOCK);
    if (blocks > MAX_BLOCKS)
    {
        std::cout << "Error: too many blocks needed for this input array" << std::endl;
        return;
    }
    uint32_t threads = (uint32_t)ceil((double)n2 / blocks); // distribute the elements evenly among the blocks

    // Set arguments for the kernel
    size_t n = (size_t)sqrt(n2);

    cudaStream_t stream; // stream capture is only supported on non-default streams
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (size_t iter = 0; iter < k; iter++)
    {
        addValue<<<blocks, threads, 0, stream>>>(d_out, d_in, n);

        // use the kernel equivalent of pointer swapping to take advantage of graph instantiation
        assignClearValue<<<blocks, threads, 0, stream>>>(d_out, d_in, n);
    }

    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    error = cudaGraphLaunch(instance, stream);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Graph launch failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Wait for the kernel to finish to avoid exiting the program prematurely
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    cudaStreamDestroy(stream);
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);

    // Copy the output back to the host
    error = cudaMemcpyAsync(out.data(), d_in, n2 * sizeof(uint8_t), cudaMemcpyDeviceToHost); // output is in d_in
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Memcpy of device's output to host failed: %s\n", cudaGetErrorString(error));
        printf("Error: %d\n", error);
        return;
    }

    // Free the memory on the device
    cudaFreeAsync(d_in, 0);
    cudaFreeAsync(d_out, 0);
}