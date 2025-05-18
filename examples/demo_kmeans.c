#include "kmeans.h"
#include "dataset.h"
#include "visualize.h"
#include <stdio.h>

int main() {
    Matrix *data = load_csv("examples/iris.csv");
    if (!data) {
        printf("Failed to load iris.csv\n");
        return 1;
    }
    normalize_matrix(data);

    KMeans *km = kmeans_create(3, data->cols);
    if (!km) {
        printf("Failed to initialize K-means\n");
        matrix_free(data);
        return 1;
    }
    kmeans_fit(km, data, 100);
    Matrix *labels = kmeans_predict(km, data);

    if (labels) {
        visualize_clusters(data, labels, km->centroids);
        matrix_free(labels);
    }

    kmeans_free(km);
    matrix_free(data);
    return 0;
}