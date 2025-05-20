#include "linreg.h"
#include <stdlib.h>

LinearRegression* linreg_create(int n_features) {
    if (n_features <= 0) return NULL;
    LinearRegression *lr = (LinearRegression*)malloc(sizeof(LinearRegression));
    if (!lr) return NULL;
    lr->weights = matrix_create(n_features, 1);
    if (!lr->weights) {
        free(lr);
        return NULL;
    }
    lr->bias = 0.0;
    return lr;
}

void linreg_free(LinearRegression *lr) {
    if (lr) {
        matrix_free(lr->weights);
        free(lr);
    }
}

void linreg_fit(LinearRegression *lr, const Matrix *X, const Matrix *y, double learning_rate, int max_iters) {
    if (!lr || !X || !y || X->rows != y->rows || y->cols != 1 || X->cols != lr->weights->rows) return;

    Matrix *pred = matrix_create(X->rows, 1);
    Matrix *error = matrix_create(X->rows, 1);
    Matrix *X_t = matrix_create(X->cols, X->rows);  // Transpose of X
    Matrix *grad = matrix_create(X->cols, 1);

    for (int iter = 0; iter < max_iters; iter++) {
        // Predict: y_pred = X * weights + bias
        matrix_multiply(X, lr->weights, pred);
        for (int i = 0; i < pred->rows; i++) {
            pred->data[i] += lr->bias;
        }

        // Compute error: error = pred - y
        for (int i = 0; i < pred->rows; i++) {
            error->data[i] = pred->data[i] - y->data[i];
        }

        // Compute gradient for weights: grad = (X^T * error) / n
        matrix_transpose(X, X_t);
        matrix_multiply(X_t, error, grad);
        for (int i = 0; i < grad->rows; i++) {
            grad->data[i] /= X->rows;
        }

        // Update weights: weights -= learning_rate * grad
        for (int i = 0; i < lr->weights->rows; i++) {
            lr->weights->data[i] -= learning_rate * grad->data[i];
        }

        // Compute gradient for bias: mean of error
        double bias_grad = 0.0;
        for (int i = 0; i < error->rows; i++) {
            bias_grad += error->data[i];
        }
        bias_grad /= X->rows;

        // Update bias
        lr->bias -= learning_rate * bias_grad;
    }

    matrix_free(pred);
    matrix_free(error);
    matrix_free(X_t);
    matrix_free(grad);
}

Matrix* linreg_predict(const LinearRegression *lr, const Matrix *X) {
    if (!lr || !X || X->cols != lr->weights->rows) return NULL;
    Matrix *pred = matrix_create(X->rows, 1);
    if (!pred) return NULL;

    // pred = X * weights + bias
    matrix_multiply(X, lr->weights, pred);
    for (int i = 0; i < pred->rows; i++) {
        pred->data[i] += lr->bias;
    }

    return pred;
}