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
    config.n_dims = 5;
    config.k_clusters = 10;
    config.max_iterations = 10'000;
    config.threshold = 0.001f;

    // 2. Host memory allocation
    std::vector<float>  h_points;
    std::vector<float>  h_centroids;
    std::vector<int>  h_labels;

    // Needed for GPU writing data back into this array
    h_points.resize(config.n_points * config.n_dims);

    // 3. Data Generation (GPU)
    // Generates random points on the GPU using curand functionality.
    std::cout << "Generating " << config.n_points << " points on GPU..." << std::endl;
    generateDataCUDA(h_points.data(), config.n_points, config.n_dims);
    std::cout << "Data generation complete." << std::endl;

    // 4. Initialization
    // Prepare host arrays for centroids and labels
    h_centroids.resize(config.k_clusters * config.n_dims);
    h_labels.resize(config.n_points);

    // Naive initialization: Pick the first K points as the initial centroids
    std::copy_n(h_points.begin(), config.k_clusters * config.n_dims, h_centroids.begin());

    // 5. Run K-Means
    std::cout << "Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points.data(), h_centroids.data(), h_labels.data(), config);
    std::cout << "K-Means finished." << std::endl;

    return 0;
}