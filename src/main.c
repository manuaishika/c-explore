#include "kmeans.h"
#include "dataset.h"
#include "visualize.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <csv_file>\n", argv[0]);
        return 1;
    }

    Matrix *data = load_csv(argv[1]);
    if (!data) {
        printf("Failed to load dataset\n");
        return 1;
    }
    normalize_matrix(data);

    // run kmeans
    KMeans *km = kmeans_create(3, data->cols);  // 3 clusters
    kmeans_fit(km, data, 100);  // Max 100 iterations
    Matrix *labels = kmeans_predict(km, data);

    visualize_clusters(data, labels, km->centroids);

    matrix_free(labels);
    kmeans_free(km);
    matrix_free(data);
    return 0;
}