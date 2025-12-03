#pragma once
#include "kmeans.cuh"

void runKMeansCPU(const float *points, float *centroids, int *labels, const KMeansConfig &config);
