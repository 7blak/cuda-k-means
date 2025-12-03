cmake_minimum_required(VERSION 4.0)
project(cuda_k_means LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

set(HOST_SOURCES src/main.cpp
        src/kmeans_cpu.cpp)
set(CUDA_SOURCES src/kmeans.cu)

add_executable(cuda_k_means ${HOST_SOURCES} ${CUDA_SOURCES})

target_include_directories(cuda_k_means PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}}/src)