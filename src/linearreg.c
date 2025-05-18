void linreg_fit(LinearRegression *lr, const Matrix *X, const Matrix *y, double lr_rate, int max_iters) {
    Matrix *pred = matrix_create(X->rows, 1);
    Matrix *error = matrix_create(X->rows, 1);
    for (int iter = 0; iter < max_iters; iter++) {
        // pred = X * weights + bias
        matrix_multiply(X, lr->weights, pred);
        for (int i = 0; i < pred->rows; i++) {
            pred->data[i] += lr->bias;
        }
        
        matrix_add(pred, y, error);
        for (int i = 0; i < error->rows; i++) {
            error->data[i] = -error->data[i];
        }
        //weights -= lr_rate * (X^T * error) / n
        Matrix *grad = matrix_create(X->cols, 1);
        matrix_transpose(X, X);
        matrix_multiply(X, error, grad);
        for (int i = 0; i < grad->rows; i++) {
            lr->weights->data[i] -= lr_rate * grad->data[i] / X->rows;
        }

        double bias_grad = 0.0;
        for (int i = 0; i < error->rows; i++) {
            bias_grad += error->data[i];
        }
        lr->bias -= lr_rate * bias_grad / X->rows;
        matrix_free(grad);
    }
    matrix_free(pred);
    matrix_free(error);
}