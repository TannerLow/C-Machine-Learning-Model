#include <cml/Model.h>
#include <cml/matrix/MatrixMathGPU.h>
#include <cml/matrix/MatrixMath.h>

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static size_t cml_getDataCellCount(const cml_Model model) {
    size_t cellCount = 0;
    for(size_t i = 0; i < model.layerCount; i++) {
        // layers after the first are different sicne they have weights, biases, and activations
        if(i > 0) {
            // weight and bias matrices
            cellCount += model.layerSizes[i-1] * model.layerSizes[i] + model.layerSizes[i];
            
            // copy of below to store values before and after activation function is applied
            cellCount += model.layerSizes[i] * model.scale;
        }

        // layers scaled by given scale
        cellCount += model.layerSizes[i] * model.scale;
    }

    return cellCount;
}

static size_t cml_getModelDataSize(const cml_Model model) {
    return sizeof(float) * cml_getDataCellCount(model);
}

static size_t cml_getModelSize(const cml_Model model) {
    size_t sizeBytes = 0;
    sizeBytes += cml_getModelDataSize(model); //data
    sizeBytes += sizeof(uint64) * model.layerCount; // layerSizes
    sizeBytes += sizeof(model.layerCount); // layerCount
    sizeBytes += sizeof(model.scale); // scale
    for(size_t i = 0; i < model.layerCount-1; i++) {
        sizeBytes += cml_getActivationFnMetadataSize(model.activationFunctions[i]);
    }
    
    return sizeBytes;
}

cml_Model cml_createModel(
    const size_t numOfLayers, 
    const uint64* layerSizes, 
    const cml_ActivationFnMetadata* activationFunctions) {
        
    return cml_createScaledModel(numOfLayers, layerSizes, 1, activationFunctions);
}

cml_Model cml_createScaledModel(
    const size_t numOfLayers, 
    const uint64* layerSizes, 
    const size_t scale, 
    const cml_ActivationFnMetadata* activationFunctions) {

    assert(numOfLayers > 0);
    assert(layerSizes != NULL);
    
    cml_Model model;
    model.scale = scale;
    model.layerCount = numOfLayers;

    size_t layerSizesSize = sizeof(uint64) * numOfLayers;
    model.layerSizes = (uint64*)malloc(layerSizesSize);
    memcpy(model.layerSizes, layerSizes, layerSizesSize);

    size_t activationFunctionsSize = sizeof(cml_ActivationFnMetadata) * (numOfLayers - 1);
    model.activationFunctions = (cml_ActivationFnMetadata*)malloc(activationFunctionsSize);
    for(size_t i = 0; i < model.layerCount-1; i++) {
        model.activationFunctions[i] = cml_duplicateActivationFnMetadata(activationFunctions[i]);
    }

    size_t modelDataSizeBytes = cml_getModelDataSize(model);
    model.data = (float*)malloc(modelDataSizeBytes);
    memset(model.data, 0, modelDataSizeBytes);

    return model;
}

void cml_deleteModel(cml_Model* model) {
    assert(model != NULL);
    assert(model->data != NULL);
    assert(model->layerSizes != NULL);
    assert(model->activationFunctions != NULL);

    free(model->data);
    free(model->layerSizes);
    for(size_t i = 0; i < model->layerCount-1; i++) {
        cml_deleteActivationFnMetadata(&model->activationFunctions[i]);
    }
    free(model->activationFunctions);

    model->data = NULL;
    model->layerSizes = NULL;
    model->layerCount = 0;
    model->scale = 1;
    model->activationFunctions = NULL;
}

cml_String cml_serializeModel(const cml_Model model) {
    size_t dataSizeBytes = cml_getModelDataSize(model);
    size_t modelSizeBytes = cml_getModelSize(model);
    // header will consist of sizeof size_t
    size_t headerSizeBytes = 1;
    char* serializedModel = (char*)malloc(headerSizeBytes + modelSizeBytes);

    // need to know sizeof size_t since other data uses this type
    // the PC architecture deserializing may not align with PC architecture that serialized it
    serializedModel[0] = (unsigned char)sizeof(size_t);

    size_t offset = 1;
    memcpy(serializedModel + offset, &model.layerCount, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(serializedModel + offset, &model.scale, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(serializedModel + offset, model.layerSizes, model.layerCount * sizeof(uint64));
    offset += model.layerCount * sizeof(uint64);
    for(size_t i = 0; i < model.layerCount-1; i++) {
        cml_String serializedActivationFn = cml_serializeActivationFnMetadata(model.activationFunctions[i]);
        memcpy(serializedModel + offset, serializedActivationFn.data, serializedActivationFn.size);
        offset += serializedActivationFn.size;
        cml_deleteString(&serializedActivationFn);
    }
    memcpy(serializedModel + offset, model.data, dataSizeBytes);

    cml_String string = cml_createString(serializedModel, headerSizeBytes + modelSizeBytes);
    return string;
}

cml_Model cml_deserializeModel(const char* serializedModel) {
    size_t layerCount;
    uint64* layerSizes = NULL;
    size_t scale;
    cml_ActivationFnMetadata* activationFunctions = NULL;

    uint8 sizeofSize_t = serializedModel[0];

    size_t offset = 1;
    memcpy(&layerCount, serializedModel + offset, sizeofSize_t);
    offset += sizeofSize_t;
    memcpy(&scale, serializedModel + offset, sizeofSize_t);
    offset += sizeofSize_t;
    size_t layerSizesBytes = sizeof(uint64) * layerCount;
    layerSizes = (uint64*)malloc(layerSizesBytes);
    memcpy(layerSizes, serializedModel + offset, layerSizesBytes);
    offset += layerSizesBytes;
    size_t activationFunctionsSizeBytes = sizeof(cml_ActivationFnMetadata) * layerCount-1;
    activationFunctions = (cml_ActivationFnMetadata*)malloc(activationFunctionsSizeBytes);
    for(size_t i = 0; i < layerCount-1; i++) {
        activationFunctions[i] = cml_deserializeActivationFnMetadata(serializedModel + offset, sizeofSize_t);
        offset += cml_getActivationFnMetadataSize(activationFunctions[i]);
    }

    cml_Model model = cml_createScaledModel(layerCount, layerSizes, scale, activationFunctions);
    size_t modelDataSizeBytes = cml_getModelDataSize(model);
    memcpy(model.data, serializedModel + offset, modelDataSizeBytes);

    return model;
}

static void cml_predictCopyInput(const cml_Model model, float* in) {
    // Copy over the input
    size_t inputCellCount = model.layerSizes[0] * model.scale;
    memcpy(model.data, in, inputCellCount * sizeof(float));
    printf("input to model: %g %g %g\n", model.data[0], model.data[1], model.data[2]);
}

static void cml_predictCopyOutput(const cml_Model model, float* out) {
    // Copy over the output
    size_t dataCellCount = cml_getDataCellCount(model);
    size_t outputCellCount = model.layerSizes[model.layerCount-1] * model.scale;
    float* outputOffset = model.data + dataCellCount - outputCellCount;
    memcpy(out, outputOffset, outputCellCount * sizeof(float));
}

void cml_predictCPU(const cml_Model model, float* in, float* out) {
    // Copy over the input
    cml_predictCopyInput(model, in);

    // Run model calculation
    cml_ModelMatrices modelMatrices = cml_getModelMatrices(model);

    for(size_t i = 0; i < model.layerCount-1; i++) {
        cml_Matrix* layerInputMatrix = (i == 0)? &modelMatrices.activationInputs[i] : &modelMatrices.activationOutputs[i-1];
        cml_matrixMultiply(*layerInputMatrix, modelMatrices.weights[i], &modelMatrices.activationInputs[i+1]);
        printf("layer 2 pre bias: %g %g\n", modelMatrices.activationInputs[i+1].data[0], modelMatrices.activationInputs[i+1].data[1]);
        cml_matrixAddRow(modelMatrices.activationInputs[i+1], modelMatrices.biases[i], &modelMatrices.activationInputs[i+1]);
        printf("layer 2 post bias: %g %g\n", modelMatrices.activationInputs[i+1].data[0], modelMatrices.activationInputs[i+1].data[1]);
        cml_ActivationFunction activation = cml_getActivation(model.activationFunctions[i].activationID);
        activation.function(&modelMatrices.activationInputs[i+1], &modelMatrices.activationOutputs[i]);
    }

    // Clean up modelMatrices pointers
    cml_deleteModelMatrices(modelMatrices);

    // Copy over the output
    cml_predictCopyOutput(model, out);
}


void _printMatrix(const cml_Matrix* matrix) {
    for(size_t row = 0; row < matrix->rows; row++) {
        for(size_t col = 0; col < matrix->cols; col++) {
            printf("%0.4f ", matrix->data[row * matrix->cols + col]);
        }
        printf("\n");
    }
    printf("\n");
}
void cml_predictGPU(const cml_Model model, float* in, float* out, const cml_GPU gpu) {
    // Copy over the input
    cml_predictCopyInput(model, in);

    // Run model calculation
    cml_ModelMatrices modelMatrices = cml_getModelMatrices(model);

    for(size_t i = 0; i < model.layerCount-1; i++) {
        cml_Matrix* layerInputMatrix = (i == 0)? &modelMatrices.activationInputs[i] : &modelMatrices.activationOutputs[i-1];
        printf("pre cml_matrixMultiplyGPU\n");
        _printMatrix(layerInputMatrix);
        _printMatrix(&modelMatrices.weights[i]);
        _printMatrix(&modelMatrices.activationInputs[i+1]);
        cml_matrixMultiplyGPU(&gpu, layerInputMatrix, &modelMatrices.weights[i], &modelMatrices.activationInputs[i+1]);
        _printMatrix(&modelMatrices.activationInputs[i+1]);
        printf("layer 2 pre bias: %g %g\n", modelMatrices.activationInputs[i+1].data[0], modelMatrices.activationInputs[i+1].data[1]);
        cml_matrixAddRowGPU(&gpu, modelMatrices.activationInputs[i+1], modelMatrices.biases[i], &modelMatrices.activationInputs[i+1]);
        printf("layer 2 post bias: %g %g\n", modelMatrices.activationInputs[i+1].data[0], modelMatrices.activationInputs[i+1].data[1]);
        cml_ActivationFunction activation = cml_getActivation(model.activationFunctions[i].activationID);
        activation.function(&modelMatrices.activationInputs[i+1], &modelMatrices.activationOutputs[i]);
    }

    // Clean up modelMatrices pointers
    cml_deleteModelMatrices(modelMatrices);

    // Copy over the output
    cml_predictCopyOutput(model, out);
}

// TODO Refactor to treat layer 1 as outputs instead of inputs, will affect cml_predict
cml_ModelMatrices cml_getModelMatrices(const cml_Model model) {
    cml_ModelMatrices modelMatrices;
    modelMatrices.activationInputs = (cml_Matrix*)malloc(sizeof(cml_Matrix) * model.layerCount);
    modelMatrices.activationOutputs = (cml_Matrix*)malloc(sizeof(cml_Matrix) * model.layerCount-1);
    modelMatrices.biases = (cml_Matrix*)malloc(sizeof(cml_Matrix) * model.layerCount-1);
    modelMatrices.weights = (cml_Matrix*)malloc(sizeof(cml_Matrix) * model.layerCount-1);

    size_t cellOffset = 0;
    for(size_t i = 0; i < model.layerCount; i++) {
        modelMatrices.activationInputs[i].data = model.data + cellOffset;
        modelMatrices.activationInputs[i].rows = model.scale;
        modelMatrices.activationInputs[i].cols = model.layerSizes[i];
        cellOffset += model.scale * model.layerSizes[i];

        // Layer 1 (index 0) has no outputs
        if(i > 0) {
            modelMatrices.activationOutputs[i-1].data = model.data + cellOffset;
            modelMatrices.activationOutputs[i-1].rows = model.scale;
            modelMatrices.activationOutputs[i-1].cols = model.layerSizes[i]; // output is the input of next layer
            cellOffset += model.scale * model.layerSizes[i];
        }

        // need to guard since cardinality of weights & biases is layerCount-1
        if(i < model.layerCount-1) {
            modelMatrices.weights[i].data = model.data + cellOffset;
            modelMatrices.weights[i].rows = model.layerSizes[i];
            modelMatrices.weights[i].cols = model.layerSizes[i+1];
            cellOffset += model.layerSizes[i] * model.layerSizes[i+1];

            modelMatrices.biases[i].data = model.data + cellOffset;
            modelMatrices.biases[i].rows = 1;
            modelMatrices.biases[i].cols = model.layerSizes[i+1];
            cellOffset += model.layerSizes[i+1];
        }
    }

    return modelMatrices;
}

void cml_deleteModelMatrices(cml_ModelMatrices modelMatrices) {
    free(modelMatrices.activationInputs);
    free(modelMatrices.activationOutputs);
    free(modelMatrices.biases);
    free(modelMatrices.weights);
    modelMatrices.activationInputs = NULL;
    modelMatrices.activationInputs = NULL;
    modelMatrices.biases = NULL;
    modelMatrices.weights = NULL;
}
