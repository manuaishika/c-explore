#ifndef LINREG_H
#define LINREG_H
#include "matrix.h"

typedef struct {
    Matrix *weights;  // Weight vector
    double bias;      // Bias term
} LinearRegression;

LinearRegression* linreg_create(int n_features);
void linreg_free(LinearRegression *lr);
void linreg_fit(LinearRegression *lr, const Matrix *X, const Matrix *y, double lr, int max_iters);
Matrix* linreg_predict(const LinearRegression *lr, const Matrix *X);

#endif