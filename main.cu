#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>


__global__ void cuda_hello(int *a)
{
    printf("Hello World from GPU!\n");
    *a = 1;
}

int main()
{
    // Allocate memory for the variable 'a' on the CPU
    int *a_cpu = (int*)malloc(sizeof(int));

    // Allocate memory for the variable 'a' on the GPU
    int *a_gpu;
    cudaMalloc((void**)&a_gpu, sizeof(int));

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

    return 0;
}