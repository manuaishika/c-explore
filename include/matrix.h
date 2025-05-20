#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    double *data;  
    int rows;
    int cols;
} Matrix;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix *m);

void matrix_add(const Matrix *a, const Matrix *b, Matrix *result);
void matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result);
void matrix_transpose(const Matrix *m, Matrix *result);
double matrix_dot(const Matrix *a, const Matrix *b);  // For vectors
void matrix_print(const Matrix *m);  // Debugging

#endif