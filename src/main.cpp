#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "kmeans.cuh"

int main() {
    KMeansConfig config{};
    config.n_points = 100'000'000;
    config.n_dims = 3;
    config.k_clusters = 5;
    config.max_iterations = 100;
    config.threshold = 0.001f;

    std::vector<float>  h_points;
    std::vector<float>  h_centroids;
    std::vector<int>  h_labels;

    h_points.resize(config.n_points * config.n_dims);

    std::cout << "Generating " << config.n_points << " points on GPU..." << std::endl;
    generateDataCUDA(h_points.data(), config.n_points, config.n_dims);
    std::cout << "Data generation complete." << std::endl;

    h_centroids.resize(config.k_clusters * config.n_dims);
    h_labels.resize(config.n_points);

    std::copy_n(h_points.begin(), (config.k_clusters * config.n_dims), h_centroids.begin());

    std::cout << "Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points.data(), h_centroids.data(), h_labels.data(), config);

    std::cout << "K-Means finished." << std::endl;

    return 0;
}