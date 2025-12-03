# CUDA K-Means Clustering

A high-performance implementation of the K-Means clustering algorithm using CUDA. This project generates random data on the GPU and compares two different parallelization strategies for centroid updates.

### To skip CPU K-Means implementation, use flag --no-cpu

## Prerequisites
* **NVIDIA GPU** with CUDA support.
* **CUDA Toolkit** (installed and configured in your PATH).
* **C++ Compiler** with C++17 support
* **CMake** version 4.0 or higher.

## Setup and Build

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/7blak/cuda-k-means
    cd cuda-k-means
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Generate build files with CMake:**
    ```bash
    cmake ..
    ```

4.  **Compile the project:**
    ```bash
    make
    ```

## Execution

Run the executable generated in the build directory:

```bash
./cuda_k_means
```

## Configuration

K-Means configuration parameters may be changed by using the config.txt file without having to recompile the program.
