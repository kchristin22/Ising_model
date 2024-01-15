#include <iostream>
#include "seq.hpp"

void isingSeq(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const size_t n, const uint32_t k)
{
    // check if in vector has the right dimensions
    if (in.size() != n * n)
    {
        std::cout << "Error: input vector has wrong dimensions" << std::endl;
        return;
    }
    out.resize(n * n);

    for (size_t i = 0; i < k; i++)
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
        in = out;
    }
}