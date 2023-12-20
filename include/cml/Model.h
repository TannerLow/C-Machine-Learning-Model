#ifndef CML_MODEL_H
#define CML_MODEL_H

#include <cml/matrix/Matrix.h>
#include <cml/util/String.h>
#include <cml/util/ActivationFnMetadata.h>
#include <cml/device/GPU.h>
#include <intdefs.h>

#include <stddef.h>

// Assumption: system serializing has equal sizeof(float) as system deserializing

typedef struct {
    float* data;
    uint64* layerSizes;
    size_t layerCount;
    size_t scale;
    cml_ActivationFnMetadata* activationFunctions; // array, count = layerCount - 1
} cml_Model;

typedef struct {
    cml_Matrix* activationInputs;
    cml_Matrix* activationOutputs;
    cml_Matrix* weights;
    cml_Matrix* biases;
} cml_ModelMatrices;

// ActivationFnMetadata array uses original data
cml_Model cml_createModel(const size_t numOfLayers, const uint64* layerSizes, const cml_ActivationFnMetadata* activationFunctions);
// Scales the layers by the scale, weight matrices remain the same size
// ActivationFnMetadata array uses original data
cml_Model cml_createScaledModel(const size_t numOfLayers, const uint64* layerSizes, const size_t scale, const cml_ActivationFnMetadata* activationFunctions);
// Does not delete activationFunctions
void cml_deleteModel(cml_Model* model);

cml_String cml_serializeModel(const cml_Model model);
cml_Model cml_deserializeModel(const char* serializedModel);

void cml_predictCPU(const cml_Model model, float* in, float* out);
void cml_predictGPU(const cml_Model model, float* in, float* out, const cml_GPU gpu);

// cml_ModelMatrices is heap allocated and needs to be deleted after use
cml_ModelMatrices cml_getModelMatrices(const cml_Model model);
// Do not use with a cml_ModelMatrices containing stack allocated pointers
void cml_deleteModelMatrices(cml_ModelMatrices modelMatrices);

#endif // CML_MODEL_H