#include "matrix.h"
#include <time.h>

// Matrix creation and destruction
Matrix* matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        ML_DEBUG_PRINT("Invalid matrix dimensions: %dx%d", rows, cols);
        return NULL;
    }
    
    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->data = calloc(rows * cols, sizeof(double));
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    m->rows = rows;
    m->cols = cols;
    m->is_view = false;
    return m;
}

Matrix* matrix_create_from_array(const double *data, int rows, int cols) {
    if (!data || rows <= 0 || cols <= 0) return NULL;
    
    Matrix *m = matrix_create(rows, cols);
    if (!m) return NULL;
    
    memcpy(m->data, data, rows * cols * sizeof(double));
    return m;
}

Matrix* matrix_create_zeros(int rows, int cols) {
    return matrix_create(rows, cols);  // calloc already zeros memory
}

Matrix* matrix_create_ones(int rows, int cols) {
    Matrix *m = matrix_create(rows, cols);
    if (!m) return NULL;
    
    matrix_fill(m, 1.0);
    return m;
}

Matrix* matrix_create_identity(int size) {
    Matrix *m = matrix_create_zeros(size, size);
    if (!m) return NULL;
    
    for (int i = 0; i < size; i++) {
        m->data[i * size + i] = 1.0;
    }
    return m;
}

Matrix* matrix_create_random(int rows, int cols, double min_val, double max_val) {
    Matrix *m = matrix_create(rows, cols);
    if (!m) return NULL;
    
    static bool seeded = false;
    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }
    
    matrix_fill_random(m, min_val, max_val);
    return m;
}

Matrix* matrix_copy(const Matrix *src) {
    if (!matrix_is_valid(src)) return NULL;
    
    return matrix_create_from_array(src->data, src->rows, src->cols);
}

Matrix* matrix_view(Matrix *src, int start_row, int start_col, int rows, int cols) {
    if (!matrix_is_valid(src) || start_row < 0 || start_col < 0 ||
        start_row + rows > src->rows || start_col + cols > src->cols) {
        return NULL;
    }
    
    Matrix *view = malloc(sizeof(Matrix));
    if (!view) return NULL;
    
    view->data = src->data + start_row * src->cols + start_col;
    view->rows = rows;
    view->cols = cols;
    view->is_view = true;
    return view;
}

void matrix_free(Matrix *m) {
    if (m) {
        if (!m->is_view) {
            ML_SAFE_FREE(m->data);
        }
        free(m);
    }
}

// Matrix properties
bool matrix_is_valid(const Matrix *m) {
    return m && m->data && m->rows > 0 && m->cols > 0;
}

bool matrix_same_size(const Matrix *a, const Matrix *b) {
    return matrix_is_valid(a) && matrix_is_valid(b) &&
           a->rows == b->rows && a->cols == b->cols;
}

bool matrix_can_multiply(const Matrix *a, const Matrix *b) {
    return matrix_is_valid(a) && matrix_is_valid(b) && a->cols == b->rows;
}

double matrix_get(const Matrix *m, int row, int col) {
    if (!matrix_is_valid(m) || row < 0 || row >= m->rows || col < 0 || col >= m->cols) {
        return NAN;
    }
    return m->data[row * m->cols + col];
}

ml_error_t matrix_set(Matrix *m, int row, int col, double value) {
    ML_CHECK_NULL(m);
    if (row < 0 || row >= m->rows || col < 0 || col >= m->cols) {
        return ML_ERROR_INVALID_PARAMETER;
    }
    m->data[row * m->cols + col] = value;
    return ML_SUCCESS;
}

// Basic operations
ml_error_t matrix_add(const Matrix *a, const Matrix *b, Matrix *result) {
    ML_CHECK_NULL(a);
    ML_CHECK_NULL(b);
    ML_CHECK_NULL(result);
    
    if (!matrix_same_size(a, b) || !matrix_same_size(a, result)) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return ML_SUCCESS;
}

ml_error_t matrix_subtract(const Matrix *a, const Matrix *b, Matrix *result) {
    ML_CHECK_NULL(a);
    ML_CHECK_NULL(b);
    ML_CHECK_NULL(result);
    
    if (!matrix_same_size(a, b) || !matrix_same_size(a, result)) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return ML_SUCCESS;
}

ml_error_t matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result) {
    ML_CHECK_NULL(a);
    ML_CHECK_NULL(b);
    ML_CHECK_NULL(result);
    
    if (!matrix_can_multiply(a, b) || result->rows != a->rows || result->cols != b->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    // Initialize result to zero
    matrix_fill(result, 0.0);
    
    // Perform multiplication
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                result->data[i * result->cols + j] += 
                    a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
    return ML_SUCCESS;
}

ml_error_t matrix_multiply_scalar(const Matrix *m, double scalar, Matrix *result) {
    ML_CHECK_NULL(m);
    ML_CHECK_NULL(result);
    
    if (!matrix_same_size(m, result)) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = m->data[i] * scalar;
    }
    return ML_SUCCESS;
}

ml_error_t matrix_transpose(const Matrix *m, Matrix *result) {
    ML_CHECK_NULL(m);
    ML_CHECK_NULL(result);
    
    if (m->rows != result->cols || m->cols != result->rows) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }
    return ML_SUCCESS;
}

ml_error_t matrix_hadamard(const Matrix *a, const Matrix *b, Matrix *result) {
    ML_CHECK_NULL(a);
    ML_CHECK_NULL(b);
    ML_CHECK_NULL(result);
    
    if (!matrix_same_size(a, b) || !matrix_same_size(a, result)) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return ML_SUCCESS;
}

// Vector operations
double matrix_dot_product(const Matrix *a, const Matrix *b) {
    if (!matrix_is_valid(a) || !matrix_is_valid(b)) return NAN;
    if (a->rows * a->cols != b->rows * b->cols) return NAN;
    
    double sum = 0.0;
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

double matrix_norm(const Matrix *m) {
    if (!matrix_is_valid(m)) return NAN;
    return sqrt(matrix_norm_squared(m));
}

double matrix_norm_squared(const Matrix *m) {
    if (!matrix_is_valid(m)) return NAN;
    
    double sum = 0.0;
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        sum += m->data[i] * m->data[i];
    }
    return sum;
}

ml_error_t matrix_normalize(Matrix *m) {
    ML_CHECK_NULL(m);
    
    double norm = matrix_norm(m);
    if (norm < ML_EPSILON) {
        return ML_ERROR_INVALID_DATA;
    }
    
    return matrix_multiply_scalar(m, 1.0 / norm, m);
}

// Statistical operations
double matrix_mean(const Matrix *m) {
    if (!matrix_is_valid(m)) return NAN;
    
    double sum = 0.0;
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        sum += m->data[i];
    }
    return sum / size;
}

double matrix_std(const Matrix *m) {
    if (!matrix_is_valid(m)) return NAN;
    
    double mean = matrix_mean(m);
    double sum_sq_diff = 0.0;
    int size = m->rows * m->cols;
    
    for (int i = 0; i < size; i++) {
        double diff = m->data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    return sqrt(sum_sq_diff / size);
}

double matrix_min(const Matrix *m) {
    if (!matrix_is_valid(m)) return NAN;
    
    double min_val = m->data[0];
    int size = m->rows * m->cols;
    for (int i = 1; i < size; i++) {
        if (m->data[i] < min_val) {
            min_val = m->data[i];
        }
    }
    return min_val;
}

double matrix_max(const Matrix *m) {
    if (!matrix_is_valid(m)) return NAN;
    
    double max_val = m->data[0];
    int size = m->rows * m->cols;
    for (int i = 1; i < size; i++) {
        if (m->data[i] > max_val) {
            max_val = m->data[i];
        }
    }
    return max_val;
}

// Utility functions
void matrix_print(const Matrix *m) {
    if (!matrix_is_valid(m)) {
        printf("Invalid matrix\n");
        return;
    }
    
    printf("Matrix [%dx%d]:\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("[");
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

void matrix_print_shape(const Matrix *m) {
    if (matrix_is_valid(m)) {
        printf("Matrix shape: (%d, %d)\n", m->rows, m->cols);
    } else {
        printf("Invalid matrix\n");
    }
}

ml_error_t matrix_fill(Matrix *m, double value) {
    ML_CHECK_NULL(m);
    
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = value;
    }
    return ML_SUCCESS;
}

ml_error_t matrix_fill_random(Matrix *m, double min_val, double max_val) {
    ML_CHECK_NULL(m);
    
    if (min_val >= max_val) {
        return ML_ERROR_INVALID_PARAMETER;
    }
    
    double range = max_val - min_val;
    int size = m->rows * m->cols;
    
    for (int i = 0; i < size; i++) {
        m->data[i] = min_val + (double)rand() / RAND_MAX * range;
    }
    return ML_SUCCESS;
}

// Row and column operations
ml_error_t matrix_get_row(const Matrix *m, int row, Matrix *result) {
    ML_CHECK_NULL(m);
    ML_CHECK_NULL(result);
    
    if (row < 0 || row >= m->rows || result->rows != 1 || result->cols != m->cols) {
        return ML_ERROR_INVALID_PARAMETER;
    }
    
    memcpy(result->data, m->data + row * m->cols, m->cols * sizeof(double));
    return ML_SUCCESS;
}

ml_error_t matrix_get_col(const Matrix *m, int col, Matrix *result) {
    ML_CHECK_NULL(m);
    ML_CHECK_NULL(result);
    
    if (col < 0 || col >= m->cols || result->rows != m->rows || result->cols != 1) {
        return ML_ERROR_INVALID_PARAMETER;
    }
    
    for (int i = 0; i < m->rows; i++) {
        result->data[i] = m->data[i * m->cols + col];
    }
    return ML_SUCCESS;
}