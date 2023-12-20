#ifndef CML_MATRIX_H
#define CML_MATRIX_H

#include <stddef.h>

typedef struct {
    float* data;
    size_t rows;
    size_t cols;
} cml_Matrix;

cml_Matrix cml_createMatrix(const size_t rows, const size_t cols);
void cml_deleteMatrix(cml_Matrix matrix);

#endif // CML_MATRIX_H