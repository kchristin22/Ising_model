#include <iostream>
#include "seq.hpp"

void isingSeq(std::vector<uint8_t> &out, std::vector<uint8_t> &in, const uint32_t k)
{
    size_t n2 = in.size();
    size_t n = (size_t)sqrt(n2);
    // check if `in` vector has a perfect square size
    if (ceil(sqrt(n2)) != floor(sqrt(n2)))
    {
        std::cout << "Error: input vector has wrong dimensions" << std::endl;
        return;
    }
    out.resize(n2);

    for (size_t iter = 0; iter < k; iter++)
    {
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                // calculate the index of the current element
                size_t index = i * n + j;
                size_t up = ((i - 1 + n) % n) * n + j;
                size_t down = ((i + 1) % n) * n + j;
                size_t left = i * n + (j - 1 + n) % n;
                size_t right = i * n + (j + 1) % n;

                uint8_t sum = in[index] + in[up] + in[down] + in[left] + in[right];
                out[index] = sum > 2; // assign majority
            }
        }
        in = out; // copy the output to the new input
    }
}