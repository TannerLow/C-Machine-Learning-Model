#ifndef CML_MATRIX_ARRAY_H
#define CML_MATRIX_ARRAY_H

#include "../matrix/Matrix.h"

typedef struct {
    cml_Matrix** matrices;
    size_t count;
} cml_MatrixArray;

cml_MatrixArray cml_createMatrixArray(const size_t count);

// TODO add delete function

#endif // CML_MATRIX_ARRAY_H