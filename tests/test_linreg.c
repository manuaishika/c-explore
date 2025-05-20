#include "linreg.h"
#include <stdio.h>
#include <assert.h>

int tests_run = 0;
int tests_passed = 0;

#define TEST(name) do { printf("Running %s...\n", #name); tests_run++; name(); } while (0)
#define ASSERT(cond) do { if (!(cond)) { printf("FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); } else { tests_passed++; } } while (0)

void test_linreg_create_free() {
    LinearRegression *lr = linreg_create(3);
    ASSERT(lr != NULL);
    ASSERT(lr->weights->rows == 3);
    ASSERT(lr->weights->cols == 1);
    ASSERT(lr->bias == 0.0);
    linreg_free(lr);
}

void test_linreg_fit_predict() {
    // Create a simple dataset: y = 2 * x1 + 3 * x2 + 1
    Matrix *X = matrix_create(4, 2);
    Matrix *y = matrix_create(4, 1);
    X->data[0] = 1.0; X->data[1] = 1.0; y->data[0] = 6.0;  // 2*1 + 3*1 + 1
    X->data[2] = 2.0; X->data[3] = 0.0; y->data[1] = 5.0;  // 2*2 + 3*0 + 1
    X->data[4] = 0.0; X->data[5] = 2.0; y->data[2] = 7.0;  // 2*0 + 3*2 + 1
    X->data[6] = 1.0; X->data[7] = 2.0; y->data[3] = 9.0;  // 2*1 + 3*2 + 1

    LinearRegression *lr = linreg_create(2);
    linreg_fit(lr, X, y, 0.01, 1000);
    Matrix *pred = linreg_predict(lr, X);

    ASSERT(pred != NULL);
    ASSERT(pred->rows == 4);
    ASSERT(pred->cols == 1);
    for (int i = 0; i < 4; i++) {
        ASSERT(pred->data[i] > y->data[i] - 1.0 && pred->data[i] < y->data[i] + 1.0);  // Within 1.0 of actual
    }

    matrix_free(pred);
    linreg_free(lr);
    matrix_free(X);
    matrix_free(y);
}

int main() {
    TEST(test_linreg_create_free);
    TEST(test_linreg_fit_predict);
    printf("Ran %d tests, %d passed\n", tests_run, tests_passed);
    return tests_run != tests_passed;
}