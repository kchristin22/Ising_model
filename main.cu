#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>
#include <map>
#include <functional>
#include "seq.hpp"
#include "cudaThreads.cuh"
#include "cudaBlocks.cuh"
#include "cudaThreadsShared.cuh"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

std::map<uint8_t, std::string> VersionsMap =
    {
        {0, "SEQ"},
        {1, "CUDA_THREADS"},
        {2, "CUDA_BLOCKS"},
        {3, "CUDA_THREADS_SHARED"},
        {4, "CUDA_THREADS_GEN"},
        {5, "CUDA_BLOCKS_GEN"},
        {6, "CUDA_THREADS_SHARED_GEN"},
        {7, "CUDA_BLOCKS_GEN_GRAPH"},
        {8, "CUDA_THREADS_GEN_GRAPH"},
        {9, "CUDA_THREADS_SHARED_GEN_GRAPH"},
        {10, "CUDA_BLOCKS_GEN_STREAMS"},
        {11, "CUDA_BLOCKS_GEN_GRAPH_STREAMS"},
        {12, "ALL_GEN"},
        {13, "ALL_GEN_GRAPH"}};

int main(int argc, char **argv)
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Needed to use cudaStreamSynchronize instead of cudaDeviceSynchronize which is slower
    std::cout << "cudaDeviceScheduleBlockingSync flag: " << prop.kernelExecTimeoutEnabled << std::endl;

    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    std::cout << "Cooperative launch support: " << supportsCoopLaunch << std::endl;

    printf("Max grid size of dimesion x: %d bytes\n", prop.maxGridSize[0]); // change macro of MAX_BLOCKS if necessary

    uint8_t version;
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

    srand(time(NULL));
    std::vector<uint8_t> in(n * n);
    std::vector<uint8_t> out(n * n);
    for (size_t i = 0; i < in.size(); i++)
    {
        in[i] = rand() % 2;
        // if (i != 0 && i % ((size_t)sqrt(in.size())) == 0)
        //     std::cout << std::endl;
        // std::cout << unsigned(in[i]) << " ";
    }
    // std::cout << std::endl;

    std::vector<uint8_t> in_copy(n * n);
    in_copy = in;
    std::vector<uint8_t> outSeq(n * n);

    std::map<uint8_t, std::function<void()>> run = {
        {0, [&]()
         { isingSeq(out, in, k); }},
        {1, [&]()
         { isingCuda(out, in, k); }},
        {2, [&]()
         { isingCuda(out, in, k, blocks); }},
        {3, [&]()
         { isingCuda(out, in, k, blocks, threadsPerBlock); }},
        {4, [&]()
         { isingCudaGen(out, in, k); }},
        {5, [&]()
         { isingCudaGen(out, in, k, blocks); }},
        {6, [&]()
         { isingCudaGen(out, in, k, blocks, threadsPerBlock); }},
        {7, [&]()
         { isingCudaGenGraph(out, in, k, blocks); }},
        {8, [&]()
         { isingCudaGenGraph(out, in, k); }},
        {9, [&]()
         { isingCudaGenGraph(out, in, k, blocks, threadsPerBlock); }},
        {10, [&]()
         { isingCudaGenStreams(out, in, k, blocks); }},
        {11, [&]()
         { isingCudaGenGraphStreams(out, in, k, blocks); }}};

    std::cout << "Running version " << VersionsMap[version] << std::endl;

    if (version < 12)
    {
        char *filename = new char[17];
        sprintf(filename, "version_%d.json", version);
        std::fstream file(filename, std::ios::out);

        // run and benchmark the version specified
        ankerl::nanobench::Bench()
            .minEpochIterations(100)
            .epochs(5)
            .run(filename, [&]
                 { run[version](); })
            .render(ankerl::nanobench::templates::pyperf(), file);

        if (VersionsMap[version] == "SEQ")
            return 0;

        cudaDeviceSynchronize();

        in = in_copy; // reset input

        isingSeq(outSeq, in, k);
        // std::cout << "out:" << std::endl;
        // for (size_t i = 0; i < out.size(); i++)
        // {
        //     if (i != 0 && i % ((size_t)sqrt(out.size())) == 0)
        //         std::cout << std::endl;
        //     std::cout << unsigned(out[i]) << " ";
        // }
        // std::cout << std::endl;
        std::cout << "Seq and Cuda are equal: " << (out == outSeq) << std::endl;

        return 0;
    }

    // run all versions

    isingSeq(outSeq, in, k);

    cudaDeviceSynchronize();
    in = in_copy;

    if (VersionsMap[version] == "ALL_GEN")
    {
        std::fstream blocksFile("blocks_gen", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("blocks_gen", [&]
                 { isingCudaGen(out, in, k, blocks); })
            .render(ankerl::nanobench::templates::pyperf(), blocksFile);

        std::cout << "Seq and Cuda blocks gen are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
        in = in_copy;

        std::fstream blocksStreamsFile("blocks_gen_streams", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("blocks_gen_streams", [&]
                 { isingCudaGenStreams(out, in, k, blocks); })
            .render(ankerl::nanobench::templates::pyperf(), blocksStreamsFile);

        std::cout << "Seq and Cuda blocks gen streams are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
        in = in_copy;

        std::fstream threadsFile("threads_gen", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(100)
            .epochs(5)
            .run("threads_gen", [&]
                 { isingCudaGen(out, in, k); })
            .render(ankerl::nanobench::templates::pyperf(), threadsFile);

        std::cout << "Seq and Cuda threads gen are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
        in = in_copy;

        std::fstream threadsSharedFile("threads_shared_gen", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(100)
            .epochs(5)
            .run("threads_shared_gen", [&]
                 { isingCudaGen(out, in, k, blocks, threadsPerBlock); })
            .render(ankerl::nanobench::templates::pyperf(), threadsSharedFile);

        std::cout << "Seq and Cuda threads shared gen are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
    }
    else
    {
        std::fstream blocksFile("blocks_gen_graph", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("blocks_gen_graph", [&]
                 { isingCudaGenGraph(out, in, k, blocks); })
            .render(ankerl::nanobench::templates::pyperf(), blocksFile);

        std::cout << "Seq and Cuda blocks gen graph are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
        in = in_copy;

        std::fstream blocksStreamsFile("blocks_gen_graph_streams", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("blocks_gen_graph_streams", [&]
                 { isingCudaGenGraphStreams(out, in, k, blocks); })
            .render(ankerl::nanobench::templates::pyperf(), blocksStreamsFile);

        std::cout << "Seq and Cuda blocks gen graph streams are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
        in = in_copy;

        std::fstream threadsFile("threads_gen_graph", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(100)
            .epochs(5)
            .run("threads_gen_graph", [&]
                 { isingCudaGenGraph(out, in, k); })
            .render(ankerl::nanobench::templates::pyperf(), threadsFile);

        std::cout << "Seq and Cuda threads gen graph are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
        in = in_copy;

        std::fstream threadsSharedFile("threads_shared_gen_graph", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(100)
            .epochs(5)
            .run("threads_shared_gen_graph", [&]
                 { isingCudaGenGraph(out, in, k, blocks, threadsPerBlock); })
            .render(ankerl::nanobench::templates::pyperf(), threadsSharedFile);

        std::cout << "Seq and Cuda threads shared gen graph are equal: " << (out == outSeq) << std::endl;
        cudaDeviceSynchronize();
    }

    return 0;
}