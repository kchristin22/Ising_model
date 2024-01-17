#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "seq.hpp"
#include "cudaSeq.cuh"
#include "cudaBlocks.cuh"
#include "cudaThreads.cuh"

int main()
{
    std::vector<uint8_t> in(100);
    std::vector<uint8_t> out(100);
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
    // isingCudaSeq(out, in, 10, 2, 2);
    isingCuda(out, in, 10, 1, 5, 1);

    for (size_t i = 0; i < out.size(); i++)
    {
        if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
            std::cout << std::endl;
        std::cout << unsigned(out[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}