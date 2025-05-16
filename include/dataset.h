#ifndef DATASET_H
#define DATASET_H
#include "matrix.h"

Matrix* load_csv(const char *filename);
void normalize_matrix(Matrix *m);

#endif