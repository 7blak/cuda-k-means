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
    std::vector<float>  h_points;
    std::vector<float>  h_centroids;
    std::vector<int>  h_labels;

    // 3. Initialization
    // Needed for GPU writing data back into this array
    h_points.resize(config.n_points * config.n_dims);
    // Prepare host arrays for centroids and labels
    h_centroids.resize(config.k_clusters * config.n_dims);
    h_labels.resize(config.n_points);

    // 4. Data Generation (GPU)
    // Generates random points on the GPU using curand functionality.
    std::cout << "Generating " << config.n_points << " points on GPU..." << std::endl;
    generateDataCUDA(h_points.data(), config.n_points, config.n_dims);
    std::cout << "Data generation complete." << std::endl;

    // Naive initialization: Pick the first K points as the initial centroids
    for (int d = 0; d < config.n_dims; d++) {
        for (int c = 0; c < config.k_clusters; c++) {
            h_centroids[d * config.k_clusters + c] = h_points[d * config.n_points + c];
        }
    }

    // 5. Run K-Means
    std::cout << "Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points.data(), h_centroids.data(), h_labels.data(), config);
    std::cout << "K-Means finished." << std::endl;

    return 0;
}