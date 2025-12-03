#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "kmeans.cuh"

int main() {
    // 1. Configuration
    // Problem size (n_points) and algorithm parameters.
    KMeansConfig config{};
    config.n_points = 100'000'000;
    config.n_dims = 4;
    config.k_clusters = 16;
    config.max_iterations = 1'000;
    config.threshold = 0.0001f;

    // 2. Host memory allocation
    std::vector<float>  h_points_a;
    std::vector<float>  h_centroids_a;
    std::vector<int>  h_labels_a;
    std::vector<float>  h_points_b;
    std::vector<float>  h_centroids_b;
    std::vector<int>  h_labels_b;

    // 3. Initialization
    // Needed for GPU writing data back into this array
    h_points_a.resize(config.n_points * config.n_dims);
    h_points_b.resize(config.n_points * config.n_dims);
    // Prepare host arrays for centroids and labels
    h_centroids_a.resize(config.k_clusters * config.n_dims);
    h_centroids_b.resize(config.k_clusters * config.n_dims);
    h_labels_a.resize(config.n_points);
    h_labels_b.resize(config.n_points);

    // 4. Data Generation (GPU)
    // Generates random points on the GPU using curand functionality.
    std::cout << "Generating " << config.n_points << " points on GPU..." << std::endl;
    generateDataCUDA(h_points_a.data(), config.n_points, config.n_dims);
    copy_n(h_points_a.begin(), config.n_points * config.n_dims, h_points_b.begin());
    std::cout << "Data generation complete." << std::endl;

    // Naive initialization: Pick the first K points as the initial centroids
    for (int d = 0; d < config.n_dims; d++) {
        for (int c = 0; c < config.k_clusters; c++) {
            h_centroids_a[d * config.k_clusters + c] = h_points_a[d * config.n_points + c];
        }
    }

    // 5. Run K-Means - Method A - Atomic Add
    std::cout << ">>>>>> [METHOD A - ATOMIC ADD] Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points_a.data(), h_centroids_a.data(), h_labels_a.data(), config, true);
    std::cout << ">>>>>> [METHOD A - ATOMIC ADD] K-Means finished." << std::endl << std::endl;

    // 6. Run K-Means - Method B - Shared Memory
    std::cout << ">>>>>> [METHOD B - SHARED MEMORY] Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points_b.data(), h_centroids_b.data(), h_labels_b.data(), config, false);
    std::cout << ">>>>>> [METHOD B - SHARED MEMORY] K-Means finished." << std::endl;

    return 0;
}