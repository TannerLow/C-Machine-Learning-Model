#include <cml/ActivationFunction.h>

#include <assert.h>
#include <string.h>

cml_ActivationFunction cml_getActivation(const enum cml_ActivationID activationID) {
    switch(activationID) {
        case CML_LINEAR: return (cml_ActivationFunction) CML_BUILTIN_LINEAR;
        case CML_RELU  : return (cml_ActivationFunction) CML_BUILTIN_RELU;
        default        : return (cml_ActivationFunction) CML_BUILTIN_NONE;
    }
}

// bool relu(const cml_Matrix* x, cml_Matrix* y);
// bool dRelu(const cml_Matrix* x, cml_Matrix* y);

// bool softmax(const cml_Matrix* x, cml_Matrix* y);
// bool dSoftmax(const cml_Matrix* x, cml_Matrix* y);

// bool cml_linear(const cml_Matrix* x, cml_Matrix* y) {
//     assert(x != NULL);
//     assert(y != NULL);
//     assert(x->rows == y->rows && x->cols == y->cols);

//     memcpy(y->data, x->data, x->rows * x->cols * sizeof(float));

//     return true;
// }

// bool cml_dLinear(const cml_Matrix* x, cml_Matrix* y) {
//     assert(x != NULL);
//     assert(y != NULL);
//     assert(x->rows == y->rows && x->cols == y->cols);

//     for(size_t row = 0; row < x->rows; row++) {
//         for(size_t col = 0; col < x->cols; col++) {
//             y->data[row * y->cols + col] = 1.0f;
//         }
//     }

//     return true;
// }