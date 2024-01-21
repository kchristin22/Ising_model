#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>
#include "seq.hpp"
#include "cudaThreads.cuh"
#include "cudaBlocks.cuh"
#include "cudaThreadsShared.cuh"
#include "cudaThreadsSharedGen.cuh"

int main(int argc, char **argv)
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // printf("Max grid size memory per block: %d bytes\n", prop.maxGridSize[0]);

    uint8_t version; // enum
    size_t n;
    uint32_t k, blocks, threadsPerBlock;
    switch (argc)
    {
    case 1:
        std::cout << "Usage: " << argv[0] << " <version> <n> <k> <number of blocks> <number of threads per block>" << std::endl;
        return 0;
    case 2:
        version = atoi(argv[1]);
        std::cout << "You need to specify the array dimension" << std::endl;
        return 0;
    case 3:
        version = atoi(argv[1]);
        n = atoi(argv[2]);
        if (version > 1)
        {
            std::cout << "You need to specify the number of blocks for this version" << std::endl;
            return 0;
        }
        std::cout << "Num of iterations not specified. Setting k = 1" << std::endl;
        k = 1;
        break;
    case 4:
        version = atoi(argv[1]);
        if (version > 1)
        {
            std::cout << "You need to specify the number of blocks for this version" << std::endl;
            return 0;
        }
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        break;
    case 5:
        version = atoi(argv[1]);
        if (version == 3)
        {
            std::cout << "You need to specify the number of threads for this version" << std::endl;
            return 0;
        }
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        blocks = atoi(argv[4]);
        break;
    case 6:
        version = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        blocks = atoi(argv[4]);
        threadsPerBlock = atoi(argv[5]);
        if (version > 2 && threadsPerBlock > MAX_THREADS_PER_BLOCK)
        {
            std::cout << "Max number of threads per block is " << MAX_THREADS_PER_BLOCK << std::endl;
            return 0;
        }
        break;
    }

    // srand(time(NULL));
    std::vector<uint8_t> in(n * n);
    std::vector<uint8_t> out(n * n);
    for (size_t i = 0; i < in.size(); i++)
    {
        in[i] = rand() % 2;
        if (i != 0 && i % ((size_t)sqrt(in.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(in[i]) << " ";
    }
    std::cout << std::endl;

    std::vector<uint8_t> in2(n * n);
    std::vector<uint8_t> out2(n * n);
    in2 = in;

    isingSeq(out, in, k);
    std::cout << "out:" << std::endl;
    for (size_t i = 0; i < out.size(); i++)
    {
        if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(out[i]) << " ";
    }
    std::cout << std::endl;

    in = in2;
    isingCuda(out2, in, k, blocks);

    std::cout << "Seq and Cuda blocks are equal: " << (out == out2) << std::endl;
    in = in2;

    std::cout << "out:" << std::endl;
    for (size_t i = 0; i < out.size(); i++)
    {
        if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(out2[i]) << " ";
    }
    std::cout << std::endl;

    isingCuda(out2, in, k);
    std::cout << "Seq and Cuda threads are equal: " << (out == out2) << std::endl;
    in = in2;

    isingCudaGen(out2, in, k, blocks, threadsPerBlock);
    std::cout << "Seq and Cuda threads shared gen are equal: " << (out == out2) << std::endl;
    in = in2;
    // std::cout << "out:" << std::endl;
    // for (size_t i = 0; i < out2.size(); i++)
    // {
    //     if (i != 0 && i % ((size_t)sqrt(out2.size())) == 0)
    //         std::cout << std::endl;
    //     std::cout << unsigned(out2[i]) << " ";
    // }
    // std::cout << std::endl;

    isingCuda(out2, in, k, blocks, threadsPerBlock);
    std::cout << "Seq and Cuda threads shared are equal: " << (out == out2) << std::endl;
    in = in2;

    std::cout << "out:" << std::endl;
    for (size_t i = 0; i < out2.size(); i++)
    {
        if (i != 0 && i % ((size_t)sqrt(out2.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(out2[i]) << " ";
    }
    std::cout << std::endl;

    // struct timeval start, end;
    // gettimeofday(&start, NULL);
    // isingCudaGen(out, in, k, blocks, threadsPerBlock);
    // gettimeofday(&end, NULL);
    // double time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000; // ms
    // std::cout << "Time gen: " << time << " ms" << std::endl;
    // std::cout << "out:" << std::endl;
    // for (size_t i = 0; i < out.size(); i++)
    // {
    //     if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
    //         std::cout << std::endl;
    //     std::cout << unsigned(out[i]) << " ";
    // }
    // std::cout << std::endl;

    // gettimeofday(&start, NULL);
    // isingCuda(out, in2, k, blocks, threadsPerBlock);
    // gettimeofday(&end, NULL);
    // time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000; // ms
    // std::cout << "Time: " << time << " ms" << std::endl;

    // std::cout << "out:" << std::endl;
    // for (size_t i = 0; i < out.size(); i++)
    // {
    //     if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
    //         std::cout << std::endl;
    //     std::cout << unsigned(out[i]) << " ";
    // }
    // std::cout << std::endl;

    return 0;
}