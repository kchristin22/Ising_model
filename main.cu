#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "seq.hpp"
#include "cudaSeq.cuh"

__global__ void cuda_hello(int *a)
{
    printf("Hello World from GPU!\n");
    *a = 1;
}

int main()
{
    // Allocate memory for the variable 'a' on the CPU
    int *a_cpu = (int *)malloc(sizeof(int));

    // Allocate memory for the variable 'a' on the GPU
    int *a_gpu;
    cudaMalloc((void **)&a_gpu, sizeof(int));

    // Launch the CUDA kernel
    cuda_hello<<<1, 1>>>(a_gpu);

    // Copy the result from the GPU to the CPU
    cudaMemcpy(a_cpu, a_gpu, sizeof(int), cudaMemcpyDeviceToHost);

    // Print messages
    printf("Hello World from CPU!\n");
    printf("a_cpu = %d\n", *a_cpu);

    // Free allocated memory
    free(a_cpu);
    cudaFree(a_gpu);

    std::vector<uint8_t> in(16);
    std::vector<uint8_t> out(16);
    for (size_t i = 0; i < in.size(); i++)
    {
        in[i] = rand() % 2;
        if (i != 0 && i % ((size_t)sqrt(in.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(in[i]) << " ";
    }
    std::cout << std::endl
              << "out:" << std::endl;

    // isingSeq(out, in, 4, 1);
    isingCudaSeq(out, in, 4, 2, 1);

    for (size_t i = 0; i < out.size(); i++)
    {
        if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(out[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}