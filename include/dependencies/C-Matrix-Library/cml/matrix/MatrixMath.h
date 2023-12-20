#ifndef CML_MATRIX_MATH_H
#define CML_MATRIX_MATH_H

#include "Matrix.h"

// By default operations are done on CPU, use GPU alternatives for hardware acceleration

//Dont think these need to be pointers since we just care about the data pointer
void cml_matrixAdd(const cml_Matrix* a, const cml_Matrix* b, cml_Matrix* c);
void cml_matrixMultiply(const cml_Matrix a, const cml_Matrix b, cml_Matrix* c);
// Adds each rows of the matrix to the given row
void cml_matrixAddRow(const cml_Matrix a, const cml_Matrix row, cml_Matrix* out);
void cml_matrixLinear(const cml_Matrix* x, cml_Matrix* y);
void cml_matrixLinearDerivative(const cml_Matrix* x, cml_Matrix* y);
void cml_matrixRelu(const cml_Matrix* x, cml_Matrix* y);
void cml_matrixReluDerivative(const cml_Matrix* x, cml_Matrix* y);

#endif // CML_MATRIX_MATH_H