cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER nvcc) # change this to your nvcc path and make sure you have linked the architecture lib
message("${CMAKE_CXX_COMPILER}")
project(ISING LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda -std=c++20 -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SOURCES "src/*.cu")

add_executable(output main.cu ${SOURCES})
target_include_directories(output PUBLIC inc)
# target_link_libraries(output PUBLIC -lcudart)
