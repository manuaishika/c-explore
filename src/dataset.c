#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix* load_csv(const char *filename) {
    // simplified CSV parser 
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;

    // count rows and cols
    int rows = 0, cols = 0;
    char line[1024];
    if (fgets(line, 1024, file)) {
        cols = 1;
        for (char *p = line; *p; p++) if (*p == ',') cols++;
    }
    while (fgets(line, 1024, file)) rows++;
    rows++;  
    fseek(file, 0, SEEK_SET);

    // load data
    Matrix *m = matrix_create(rows, cols);
    int r = 0;
    while (fgets(line, 1024, file) && r < rows) {
        char *token = strtok(line, ",");
        int c = 0;
        while (token && c < cols) {
            m->data[r * cols + c] = atof(token);
            token = strtok(NULL, ",");
            c++;
        }
        r++;
    }
    fclose(file);
    return m;
}

void normalize_matrix(Matrix *m) {
    // min max normalization 
    for (int j = 0; j < m->cols; j++) {
        double min_val = m->data[j], max_val = m->data[j];
        for (int i = 0; i < m->rows; i++) {
            double val = m->data[i * m->cols + j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        double range = max_val - min_val;
        if (range > 0) {
            for (int i = 0; i < m->rows; i++) {
                m->data[i * m->cols + j] = (m->data[i * m->cols + j] - min_val) / range;
            }
        }
    }
}