#ifndef SVM_H
#define SVM_H
#include "matrix.h"

typedef struct {
    Matrix *support_vectors;
    Matrix *weights;
    double bias;
} SVM;

SVM* svm_create(int n_features);
void svm_free(SVM *svm);
void svm_fit(SVM *svm, const Matrix *X, const Matrix *y, double C, int max_iters);
Matrix* svm_predict(const SVM *svm, const Matrix *X);

#endif