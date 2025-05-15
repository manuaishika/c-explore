#include "matrix.h"
#include <stdlib.h>
#include <string.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));  // zero initialized
    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_add(const Matrix *a, const Matrix *b, Matrix *result) {
    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols) {
        return;  // errors
    }
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result) {
    if (a->cols != b->rows || a->rows != result->rows || b->cols != result->cols) {
        return;  // errors
    }
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
}

void matrix_transpose(const Matrix *m, Matrix *result) {
    if (m->rows != result->cols || m->cols != result->rows) {
        return;  // errors
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }
}

double matrix_dot(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != 1 || b->cols != 1) {
        return 0.0;  // errors
    }
    double sum = 0.0;
    for (int i = 0; i < a->rows; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}