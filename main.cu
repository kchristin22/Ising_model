#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "seq.hpp"
#include "cudaThreads.cuh"
#include "cudaBlocks.cuh"
#include "cudaThreadsShared.cuh"

int main()
{
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
    isingCuda(out, in, 1);
    // isingCuda(out, in, 10, 1, 1, 5);
    // isingCuda(out, in, 10, 1, 5);

    for (size_t i = 0; i < out.size(); i++)
    {
        if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(out[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}