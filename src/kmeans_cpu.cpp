#include "kmeans_cpu.h"

#include <chrono>
#include <iostream>
#include <cmath>
#include <ostream>
#include <vector>

void assignClustersCPU(const float *points, const float *centroids, int *labels, const int n, const int d, const int k) {
    for (int i = 0; i < n; i++) {
        float min_dist = MAXFLOAT;
        int best_cluster = -1;

        for (int c = 0; c < k; c++) {
            float dist = 0.0f;

            for (int dim = 0; dim < d; dim++) {
                const float p_val = points[dim * n + i];
                const float c_val = centroids[dim * k + c];
                const float diff = p_val - c_val;
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        labels[i] = best_cluster;
    }
}

float updateCentroidsCPU(const float *points, const int *labels, float *centroids, const int n, const int d, const int k) {
    std::vector<float> sums(k * d, 0.0f);
    std::vector<int> counts(k, 0);

    for (int i = 0; i < n; i++) {
        const int cluster_id = labels[i];
        counts[cluster_id]++;

        for (int dim = 0; dim < d; dim++) {
            sums[dim * k + cluster_id] += points[dim * n + i];
        }
    }

    float total_movement = 0.0f;

    for (int c = 0; c < k; c++) {
        const int count = counts[c];
        if (count > 0) {
            for (int dim = 0; dim < d; dim++) {
                int idx = dim * k + c;
                float new_val = sums[idx] / count;
                float old_val = centroids[idx];

                centroids[idx] = new_val;

                float diff = new_val - old_val;
                total_movement += diff * diff;
            }
        }
    }

    return total_movement;
}

void runKMeansCPU(const float *points, float *centroids, int *labels, const KMeansConfig &config) {
    std::cout << "Starting CPU K-Means Loop..." << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();

    double total_assign_ms = 0.0;
    double total_update_ms = 0.0;
    int iterations = 0;

    for (int iter = 0; iter < config.max_iterations; iter++) {
        iterations++;

        auto t1 = std::chrono::high_resolution_clock::now();

        assignClustersCPU(points, centroids, labels, config.n_points, config.n_dims, config.k_clusters);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dt_assign = t2 - t1;
        total_assign_ms += dt_assign.count();

        auto t3 = std::chrono::high_resolution_clock::now();

        float movement = updateCentroidsCPU(points, labels, centroids, config.n_points, config.n_dims, config.k_clusters);

        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dt_update = t4 - t3;
        total_update_ms += dt_update.count();

        if (movement < config.threshold) {
            std::cout << "Converged at iteration " << iter << " with movement: " << movement << std::endl;
            break;
        }

        std::cout << "Iteration: " << iterations << std::endl;
    }

    auto stop_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = stop_total - start_total;

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Total Execution Time (CPU):      " << total_duration.count() << " ms" << std::endl;
    std::cout << "Iterations executed:             " << iterations << std::endl;
    std::cout << "Average Time per Iteration:      " << total_duration.count() / iterations << " ms" << std::endl;
    std::cout << "Average AssignClusters Time:     " << total_assign_ms / iterations << " ms" << std::endl;
    std::cout << "Average UpdateCentroids Time:    " << total_update_ms / iterations << " ms" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
}
