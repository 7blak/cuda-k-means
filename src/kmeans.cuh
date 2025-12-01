#pragma once

struct KMeansConfig {
    int n_points;
    int n_dims;
    int k_clusters;
    int max_iterations;
    float threshold;
};

void runKMeansCUDA(const float* h_points, float* h_centroids, int* h_labels, const KMeansConfig& config);
void generateDataCUDA(float* h_points, int n, int d);