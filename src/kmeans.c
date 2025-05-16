#include "kmeans.h"
#include <stdlib.h>
#include <math.h>

KMeans* kmeans_create(int k, int n_features) {
    KMeans *km = (KMeans*)malloc(sizeof(KMeans));
    km->k = k;
    km->centroids = matrix_create(k, n_features);
    km->assignments = (int*)calloc(1, sizeof(int));  // Dynamically resized in fit
    return km;
}

void kmeans_free(KMeans *km) {
    if (km) {
        matrix_free(km->centroids);
        free(km->assignments);
        free(km);
    }
}

void kmeans_fit(KMeans *km, const Matrix *data, int max_iters) {
    int n_samples = data->rows;
    int n_features = data->cols;

    // Resize assignments array
    km->assignments = (int*)realloc(km->assignments, n_samples * sizeof(int));

    // Initialize centroids randomly (simplified: first k points)
    for (int i = 0; i < km->k; i++) {
        for (int j = 0; j < n_features; j++) {
            km->centroids->data[i * n_features + j] = data->data[i * n_features + j];
        }
    }

    // K-means loop
    for (int iter = 0; iter < max_iters; iter++) {
        int changed = 0;

        // Assign points to nearest centroid
        for (int i = 0; i < n_samples; i++) {
            double min_dist = INFINITY;
            int best_cluster = 0;
            for (int j = 0; j < km->k; j++) {
                double dist = 0.0;
                for (int f = 0; f < n_features; f++) {
                    double diff = data->data[i * n_features + f] - km->centroids->data[j * n_features + f];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            if (km->assignments[i] != best_cluster) {
                km->assignments[i] = best_cluster;
                changed = 1;
            }
        }

        // Update centroids
        Matrix *counts = matrix_create(km->k, 1);  // Count of points per cluster
        matrix_free(km->centroids);
        km->centroids = matrix_create(km->k, n_features);
        for (int i = 0; i < n_samples; i++) {
            int c = km->assignments[i];
            counts->data[c] += 1.0;
            for (int f = 0; f < n_features; f++) {
                km->centroids->data[c * n_features + f] += data->data[i * n_features + f];
            }
        }
        for (int i = 0; i < km->k; i++) {
            if (counts->data[i] > 0) {
                for (int f = 0; f < n_features; f++) {
                    km->centroids->data[i * n_features + f] /= counts->data[i];
                }
            }
        }
        matrix_free(counts);

        // Early stopping if no changes
        if (!changed) break;
    }
}

Matrix* kmeans_predict(const KMeans *km, const Matrix *data) {
    Matrix *labels = matrix_create(data->rows, 1);
    for (int i = 0; i < data->rows; i++) {
        double min_dist = INFINITY;
        int best_cluster = 0;
        for (int j = 0; j < km->k; j++) {
            double dist = 0.0;
            for (int f = 0; f < data->cols; f++) {
                double diff = data->data[i * data->cols + f] - km->centroids->data[j * data->cols + f];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels->data[i] = (double)best_cluster;
    }
    return labels;
}