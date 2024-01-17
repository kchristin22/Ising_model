#include <iostream>
#include "cudaSeq.cuh"

__global__ void isingModel(uint8_t *out, uint8_t *in, const size_t n, const uint32_t k)
{
    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                // calculate the index of the current element
                size_t index = i * n + j;

                uint8_t sum = in[index] + in[((i + 1) % n) * n + j] + in[((i - 1 + n) % n) * n + j] + in[i * n + (j + 1) % n] + in[i * n + (j - 1 + n) % n];
                out[index] = sum > 2; // if majority is true (sum in [3,5]), out is true
            }
        }
        memcpy(in, out, sizeof(out));
    }
}

void isingCudaSeq(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const size_t n, const uint32_t k, const uint32_t kThread)
{
    // check if in vector has the right dimensions
    if (in.size() != n * n)
    {
        std::cout << "Error: input vector has wrong dimensions" << std::endl;
        return;
    }
    out.resize(n * n);

    // Allocate memory on the device
    uint8_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, n * n * sizeof(uint8_t));
    cudaMalloc((void **)&d_out, n * n * sizeof(uint8_t));

    // Copy the input to the device
    cudaMemcpy(d_in, in.data(), n * n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    uint32_t chunk = k / kThread;
    uint32_t threads = chunk * kThread == k ? chunk : chunk + 1;

    // Run the kernel
    for (uint32_t i = 1; i <= threads; i++)
    {
        size_t iter = i == threads ? k - (i - 1) * kThread : kThread;
        isingModel<<<1, 1>>>(d_out, d_in, n, iter);
        cudaDeviceSynchronize();
    }

    // Copy the output back to the host
    cudaMemcpy(out.data(), d_out, n * n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_in);
    cudaFree(d_out);
}