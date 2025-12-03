#include "kmeans.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

// Macro for checking CUDA errors
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// --- RANDOM NUMBER GENERATION KERNELS ---

// Initialized a random state for EACH thread. This is computationally expensive, so it is only done once (best practice)
__global__ void initRNG(curandState *states, unsigned long seed, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        curand_init(seed, idx, 0, &states[idx]);
}

// Uses the initialized states to generate random floats
__global__ void generatePointsKernel(float *d_points, curandState *states, int n, int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState localState = states[idx];

        const int point_start_index = idx * d;

        for (int i = 0; i < d; i++) {
            const float rand_val = curand_uniform(&localState);
            d_points[point_start_index + i] = rand_val * 100.0f;
        }

        states[idx] = localState;
    }
}

// --- K-MEANS PART 1: CLUSTER ASSIGNMENT ---

// For every point, find the nearest centroid
__global__ void assignClusters(const float *d_points, const float *d_centroids, int *d_labels, const int n, const int d,
                               const int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) // Bounds check
        return;

    float min_dist = MAXFLOAT;
    int best_cluster = -1;
    const int point_start_index = idx * d;

    // Loop through all K centroids
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        const int centroid_start_index = c * d;

        // Euclidean Distance Calculation
        for (int i = 0; i < d; i++) {
            const float p_val = d_points[point_start_index + i];
            const float c_val = d_centroids[centroid_start_index + i];
            const float diff = p_val - c_val;
            dist += diff * diff; // Squared distance is sufficient for comparison (avoiding using sqrt())
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    d_labels[idx] = best_cluster;
}

// --- K-MEANS PART 2: UPDATE CENTROIDS ---

// Method A: Global Atomics - Simple but slow due to usage of global memory
__global__ void updateCentroids_MethodA(const float *d_points, const int *d_labels, float *d_newCentroids,
                                        int *d_counts, const int n, const int d, const int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    const int cluster_id = d_labels[idx];

    // Critical point, thousands of threads trying to write to the same address, will cause delays
    atomicAdd(&d_counts[cluster_id], 1);

    const int point_start = idx * d;
    const int centroid_start = cluster_id * d;

    for (int i = 0; i < d; i++) {
        const float val = d_points[point_start + i];
        atomicAdd(&d_newCentroids[centroid_start + i], val);
    }
}

// Method B: Shared Memory - accumulate locally in Shared Memory first, then write to Global
__global__ void updateCentroids_MethodB(const float *d_points, const int *d_labels, float *d_newCentroids,
                                        int *d_counts, int n, int d, int k) {
    // Dynamic Shared Memory: Size decided at runtime launch
    extern __shared__ float s_mem[];

    // Organize shared memory: [Centroid Sums (floats) | Counts (ints)]
    float *s_centroids = s_mem;
    int *s_counts = reinterpret_cast<int *>(&s_centroids[k * d]);

    // 1. Initialize Shared Memory to 0 (Collaborative reset by thread block)
    int s_len_floats = k * d;
    int s_len_inits = k;
    for (int i = threadIdx.x; i < s_len_floats; i += blockDim.x) {
        s_centroids[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < s_len_inits; i += blockDim.x) {
        s_counts[i] = 0;
    }

    __syncthreads(); // Wait for all threads to finish initialization

    // 2. Accumulate into Shared Memory
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const int cluster_id = d_labels[idx];
        atomicAdd(&s_counts[cluster_id], 1);
        const int point_start = idx * d;
        const int s_centroid_start = cluster_id * d;

        for (int i = 0; i < d; i++) {
            const float val = d_points[point_start + i];
            atomicAdd(&s_centroids[s_centroid_start + i], val);
        }

        __syncthreads(); // Wait for all threads to finish accumulation

        // 3. Write Shared Memory totals to Global Memory
        // Only the first K threads do the writing to reduce global traffic
        if (threadIdx.x < k) {
            const int thread_cluster_id = threadIdx.x;
            atomicAdd(&d_counts[thread_cluster_id], s_counts[thread_cluster_id]);

            const int start_idx = thread_cluster_id * d;
            for (int i = 0; i < d; i++) {
                atomicAdd(&d_newCentroids[start_idx + i], s_centroids[start_idx + i]);
            }
        }
    }
}

// Wrapper for Data Generation
void generateDataCUDA(float *h_points, const int n, const int d) {
    size_t data_size = n * d * sizeof(float);

    float *d_points;
    curandState *d_states;

    CHECK_CUDA(cudaMalloc(&d_points, data_size));
    CHECK_CUDA(cudaMalloc(&d_states, n * sizeof(curandState)));

    const int threadCount = 256;
    const int blockCount = (n + threadCount - 1) / threadCount;

    unsigned long seed = time(NULL);
    initRNG<<<blockCount, threadCount>>>(d_states, seed, n);
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    generatePointsKernel<<<blockCount, threadCount>>>(d_points, d_states, n, d);
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(h_points, d_points, data_size, cudaMemcpyDeviceToHost));

    cudaFree(d_points);
    cudaFree(d_states);
}

// Wrapper for running K-Means
void runKMeansCUDA(const float *h_points, float *h_centroids, int *h_labels, const KMeansConfig &config) {
    const size_t points_size = config.n_points * config.n_dims * sizeof(float);
    const size_t centroids_size = config.k_clusters * config.n_dims * sizeof(float);
    const size_t labels_size = config.n_points * sizeof(int);

    const int threadCount = 256;
    const int blockCount = (config.n_points * config.n_dims + threadCount - 1) / threadCount;

    float *d_points, *d_centroids, *d_newCentroids_sum;
    int *d_labels, *d_newCentroids_count;

    CHECK_CUDA(cudaMalloc(&d_points, points_size));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA(cudaMalloc(&d_newCentroids_sum, centroids_size));
    CHECK_CUDA(cudaMalloc(&d_labels, labels_size));
    CHECK_CUDA(cudaMalloc(&d_newCentroids_count, config.k_clusters * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // -- MAIN LOOP --
    for (int iter = 0; iter < config.max_iterations; iter++) {
        // Step 1: Assign Clusters
        assignClusters<<<blockCount, threadCount>>>(d_points, d_centroids, d_labels, config.n_points, config.n_dims,
                                                    config.k_clusters);
        CHECK_CUDA(cudaGetLastError());
        cudaDeviceSynchronize();

        // Step 2: Reset Accumulators
        CHECK_CUDA(cudaMemset(d_newCentroids_sum, 0, centroids_size));
        CHECK_CUDA(cudaMemset(d_newCentroids_count, 0, config.k_clusters * sizeof(int)));

        // Step 3: Update Centroids
        size_t shared_mem_size = config.k_clusters * config.n_dims * sizeof(float) + config.k_clusters * sizeof(int);

        // updateCentroids_MethodA<<<blockCount, threadCount>>>(d_points, d_labels, d_newCentroids_sum, d_newCentroids_count, config.n_points, config.n_dims, config.k_clusters);
        updateCentroids_MethodB<<<blockCount, threadCount, shared_mem_size>>>(
            d_points, d_labels, d_newCentroids_sum, d_newCentroids_count, config.n_points, config.n_dims,
            config.k_clusters);
        CHECK_CUDA(cudaGetLastError());
        cudaDeviceSynchronize();

        // Step 4: Convergence Check (CPU)
        // Copies sums and counts back to CPU for averaging
        std::vector<float> h_temp_sums(config.k_clusters * config.n_dims);
        std::vector<int> h_temp_counts(config.k_clusters);

        CHECK_CUDA(cudaMemcpy(h_temp_sums.data(), d_newCentroids_sum, centroids_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(
            cudaMemcpy(h_temp_counts.data(), d_newCentroids_count, config.k_clusters * sizeof(int),
                cudaMemcpyDeviceToHost));

        float total_movement = 0.0f;

        for (int i = 0; i < config.k_clusters; i++) {
            const int count = h_temp_counts[i];

            if (count > 0) {
                for (int j = 0; j < config.n_dims; j++) {
                    const int idx = i * config.n_dims + j;

                    const float new_val = h_temp_sums[idx] / count;
                    const float old_val = h_centroids[idx];
                    h_centroids[idx] = new_val;
                    float diff = new_val - old_val;
                    total_movement += diff * diff;
                }
            }
        }

        if (total_movement < config.threshold) {
            std::cout << "Converged at iteration " << iter << " with movement: " << total_movement << std::endl;
            break;
        }

        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));

        // Code for showing changes among the first 10 points
        std::vector<int> debug_labels(10);
        cudaMemcpy(debug_labels.data(), d_labels, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Debug Iteration " << iter << "\t| Movement: " << total_movement << "\tLabels:\t";
        for (int i = 0; i < 10; i++)
            std::cout << debug_labels[i] << " ";
        std::cout << std::endl;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Total Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Average Time per Iteration: " << milliseconds / config.max_iterations << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_newCentroids_sum);
    cudaFree(d_labels);
    cudaFree(d_newCentroids_count);
}
