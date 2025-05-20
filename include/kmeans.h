#ifndef KMEANS_H
#define KMEANS_H
#include "matrix.h"

typedef struct {
    Matrix *centroids;  // k x n matrix (k clusters, n features)
    int *assignments;   // Cluster assignment for each data point
    int k;              // Number of clusters
} KMeans;

KMeans* kmeans_create(int k, int n_features);
void kmeans_free(KMeans *km);
void kmeans_fit(KMeans *km, const Matrix *data, int max_iters);
Matrix* kmeans_predict(const KMeans *km, const Matrix *data);

#endif