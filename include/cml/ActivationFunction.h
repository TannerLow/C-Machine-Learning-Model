#ifndef CML_ACTIVATIONS_H
#define CML_ACTIVATIONS_H

#include <cml/matrix/Matrix.h>
#include <cml/matrix/MatrixMath.h>

#include <stdbool.h>

enum cml_ActivationID {CML_LINEAR, CML_RELU, CML_CUSTOM, CML_NONE};

typedef struct {
    void (*function)(const cml_Matrix*, cml_Matrix*);
    void (*derivative)(const cml_Matrix*, cml_Matrix*);
    bool elementWiseEligible;
} cml_ActivationFunction;


// builtin activation functions

// bool relu(const cml_Matrix* x, cml_Matrix* y);
// bool dRelu(const cml_Matrix* x, cml_Matrix* y);

// bool softmax(const cml_Matrix* x, cml_Matrix* y);
// bool dSoftmax(const cml_Matrix* x, cml_Matrix* y);

// bool cml_linear(const cml_Matrix* x, cml_Matrix* y);
// bool cml_dLinear(const cml_Matrix* x, cml_Matrix* y);

// macros for more readable code when using the functions provided above
#define CML_BUILTIN_LINEAR {cml_matrixLinear, cml_matrixLinearDerivative, true}
#define CML_BUILTIN_RELU {cml_matrixRelu, cml_matrixReluDerivative, true}
#define CML_BUILTIN_SOFTMAX {softmax, dSoftmax, false} // DO NOT USE YET
#define CML_BUILTIN_NONE {NULL, NULL, false}

cml_ActivationFunction cml_getActivation(const enum cml_ActivationID activationID);

#endif // CML_ACTIVATIONS_H