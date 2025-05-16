#include "visualize.h"
#include <stdio.h>

void visualize_clusters(const Matrix *data, const Matrix *labels, const Matrix *centroids) {
    if (data->cols != 2) {
        printf("Visualization requires 2D data\n");
        return;
    }
    char grid[20][20] = {0};  // 20x20 grid
    for (int i = 0; i < data->rows; i++) {
        int x = (int)(data->data[i * 2] * 19);  
        int y = (int)(data->data[i * 2 + 1] * 19);
        if (x >= 0 && x < 20 && y >= 0 && y < 20) {
            grid[y][x] = '0' + (int)labels->data[i];  // cluster label
        }
    }
    for (int i = 0; i < centroids->rows; i++) {
        int x = (int)(centroids->data[i * 2] * 19);
        int y = (int)(centroids->data[i * 2 + 1] * 19);
        if (x >= 0 && x < 20 && y >= 0 && y < 20) {
            grid[y][x] = '*';  // centroid marker
        }
    }
    for (int y = 19; y >= 0; y--) {
        for (int x = 0; x < 20; x++) {
            printf("%c ", grid[y][x] ? grid[y][x] : '.');
        }
        printf("\n");
    }
}