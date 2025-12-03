#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include "kmeans.cuh"
#include "kmeans_cpu.h"

// Helper to trim whitespace
std::string trim(const std::string &str) {
    const size_t first = str.find_first_not_of(" \t");
    if (std::string::npos == first) return str;
    const size_t last = str.find_last_not_of(" \t");
    return str.substr(first, last - first + 1);
}

KMeansConfig loadConfig(const std::string &filename) {
    KMeansConfig config{};
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << ". Using default." << std::endl;
        // Default fallback
        config.n_points = 1'000'000;
        config.n_dims = 3;
        config.k_clusters = 5;
        config.max_iterations = 100;
        config.threshold = 0.001f;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream is_line(line);
        std::string key;
        if (std::getline(is_line, key, '=')) {
            std::string value;
            if (std::getline(is_line, value)) {
                key = trim(key);
                value = trim(value);

                if (key == "n_points") config.n_points = std::stoi(value);
                else if (key == "n_dims") config.n_dims = std::stoi(value);
                else if (key == "k_clusters") config.k_clusters = std::stoi(value);
                else if (key == "max_iterations") config.max_iterations = std::stoi(value);
                else if (key == "threshold") config.threshold = std::stof(value);
            }
        }
    }

    std::cout << "Loaded Configuration from " << filename << ":" << std::endl;
    std::cout << "  Points: " << config.n_points << std::endl;
    std::cout << "  Dims:   " << config.n_dims << std::endl;
    std::cout << "  K:      " << config.k_clusters << std::endl;
    std::cout << "  Iter:   " << config.max_iterations << std::endl;
    std::cout << "  Thresh: " << config.threshold << std::endl << std::endl;

    return config;
}

int main(int argc, char** argv) {
    bool runCPU = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-cpu") {
            runCPU = false;
            std::cout << "Flag --no-cpu detected: Skipping CPU implementation." << std::endl;
        }
    }

    // 1. Load Configuration
    KMeansConfig config = loadConfig("config.txt");

    // 2. Host memory allocation
    std::vector<float> h_points;

    std::vector<float> h_centroids_a;
    std::vector<int> h_labels_a;

    std::vector<float> h_centroids_b;
    std::vector<int> h_labels_b;

    std::vector<float> h_centroids_cpu;
    std::vector<int> h_labels_cpu;

    // 3. Initialization
    // Needed for GPU writing data back into this array
    h_points.resize(config.n_points * config.n_dims);
    // Prepare host arrays for centroids and labels
    h_centroids_a.resize(config.k_clusters * config.n_dims);
    h_labels_a.resize(config.n_points);
    h_labels_b.resize(config.n_points);
    h_labels_cpu.resize(config.n_points);

    // 4. Data Generation (GPU)
    // Generates random points on the GPU using curand functionality.
    std::cout << "Generating " << config.n_points << " points on GPU..." << std::endl;
    generateDataCUDA(h_points.data(), config.n_points, config.n_dims);
    std::cout << "Data generation complete." << std::endl;

    // Naive initialization: Pick the first K points as the initial centroids
    for (int d = 0; d < config.n_dims; d++) {
        for (int c = 0; c < config.k_clusters; c++) {
            h_centroids_a[d * config.k_clusters + c] = h_points[d * config.n_points + c];
        }
    }

    h_centroids_b = h_centroids_a;
    h_centroids_cpu = h_centroids_a;

    // 5. Run K-Means - Method A - Atomic Add
    std::cout << ">>>>>> [METHOD A - ATOMIC ADD] Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points.data(), h_centroids_a.data(), h_labels_a.data(), config, true);
    std::cout << ">>>>>> [METHOD A - ATOMIC ADD] K-Means finished." << std::endl << std::endl;

    // 6. Run K-Means - Method B - Shared Memory
    std::cout << ">>>>>> [METHOD B - SHARED MEMORY] Starting CUDA K-Means..." << std::endl;
    runKMeansCUDA(h_points.data(), h_centroids_b.data(), h_labels_b.data(), config, false);
    std::cout << ">>>>>> [METHOD B - SHARED MEMORY] K-Means finished." << std::endl << std::endl;

    // 7. Run K-Means - CPU
    if (runCPU) {
        std::cout << ">>>>>> [METHOD C - CPU] Starting CPU K-Means..." << std::endl;
        runKMeansCPU(h_points.data(), h_centroids_cpu.data(), h_labels_cpu.data(), config);
        std::cout << ">>>>>> [METHOD C - CPU] K-Means finished." << std::endl;
    } else {
        std::cout << ">>>>>> [METHOD C - CPU] Skipped (--no-cpu flag present)." << std::endl;
    }

    return 0;
}
