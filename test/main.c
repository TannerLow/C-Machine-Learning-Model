#include <cml/Logger.h>
#include <cml/Model.h>
#include <cml/util/String.h>
#include <intdefs.h>
#include <stdio.h>
#include <cml/ActivationFunction.h>
#include <cml/util/ActivationFnMetadata.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

void printMatrix(const cml_Matrix* matrix) {
    for(size_t row = 0; row < matrix->rows; row++) {
        for(size_t col = 0; col < matrix->cols; col++) {
            printf("%0.4f ", matrix->data[row * matrix->cols + col]);
        }
        printf("\n");
    }
    printf("\n");
}

bool test_createAndSerializeModel();
bool test_modelPredictCPULinear();
bool test_modelPredictCPURelu();
bool test_modelPredictGPULinear();
bool cml_withinMarginOfError(const float actual, const float expected, const float acceptableDeviation);

int main() {
    cml_logStream = stdout;
    return 
        test_createAndSerializeModel() &&
        test_modelPredictCPULinear() &&
        test_modelPredictCPURelu() &&
        test_modelPredictGPULinear();
}

bool test_createAndSerializeModel() {
    // Model Specs
    size_t numOflayers = 3;
    uint64 layerSizes[] = {3,4,2};
    cml_ActivationFnMetadata* activations = (cml_ActivationFnMetadata*)malloc(sizeof(cml_ActivationFnMetadata) * 2);
    for(size_t i = 0; i < numOflayers-1; i++) {
        activations[i] = cml_createActivationFnMetadataWithID(NULL, NULL, CML_RELU);
    }

    // Model
    cml_Model model = cml_createModel(numOflayers, layerSizes, activations);
    
    // Use and test
    printf("Layer Count: %lld\n", model.layerCount);
    printf("Scale: %lld\n", model.scale);
    for(size_t i = 0; i < model.layerCount; i++) {
        printf("Layer Size: %lld\n", model.layerSizes[i]);
    }
    
    // Set model weights and biases manually
    int cells = 3 + 12 + 4 + 4 + 8 + 2 + 2;
    int value = 1;
    for(int i = 0; i < cells; i++) {
        model.data[i] = value++;
    }

    // Test serialization
    cml_String serializedModel = cml_serializeModel(model);
    for(size_t i = 0; i < serializedModel.size; i++) {
        printf("%d ", (int)serializedModel.data[i]);
    }
    printf("\n");
    printf("serialized size: %lld\n", serializedModel.size);
    FILE* file = fopen("model.dat", "w");
    printf("%lld\n", fwrite(serializedModel.data, 1, serializedModel.size, file));
    fclose(file);

    // Test Deserialization
    cml_Model newModel = cml_deserializeModel(serializedModel.data);

    // Use and test
    printf("Layer Count: %lld\n", newModel.layerCount);
    printf("Scale: %lld\n", newModel.scale);
    for(size_t i = 0; i < newModel.layerCount; i++) {
        printf("Layer Size: %lld\n", newModel.layerSizes[i]);
    }
    printf("%g\n", newModel.data[12]);

    cml_String serializedNewModel = cml_serializeModel(newModel);

    // Compare
    int cmp = memcmp(serializedModel.data, serializedNewModel.data, serializedModel.size);

    // clean up memory
    cml_deleteModel(&model);
    for(size_t i = 0; i < numOflayers-1; i++) {
        cml_deleteActivationFnMetadata(&activations[i]);
    }
    free(activations);

    cml_deleteModel(&newModel);
    cml_deleteString(&serializedModel);
    cml_deleteString(&serializedNewModel);

    printf("done\n");

    return cmp == 0;
}

bool test_modelPredictCPULinear() {
    // Model Specs
    size_t numOflayers = 3;
    uint64 layerSizes[] = {3,2,2};
    cml_ActivationFnMetadata* activations = (cml_ActivationFnMetadata*)malloc(sizeof(cml_ActivationFnMetadata) * 2);
    for(size_t i = 0; i < numOflayers-1; i++) {
        activations[i] = cml_createActivationFnMetadataWithID(NULL, NULL, CML_LINEAR);
    }

    // Model
    cml_Model model = cml_createModel(numOflayers, layerSizes, activations);
    
    // Use and test
    printf("Layer Count: %lld\n", model.layerCount);
    printf("Scale: %lld\n", model.scale);
    for(size_t i = 0; i < model.layerCount; i++) {
        printf("Layer Size: %lld\n", model.layerSizes[i]);
    }

    // Set model weights and biases manually
    float layer1Weights[] = {1,2,3,4,5,6};
    float layer2Biases[] = {1,2};
    float layer2Weights[] = {4,3,2,1};
    float layer3Biases[] = {2,1};
    // 3     6    2    2      2      4    2    2      2
    // L1 - W12 - B1 - A1 - Z1/L2 - W23 - B2 - A2 - Z2/L3
    
    // W12
    for(int i = 3; i < 9; i++) {
        model.data[i] = layer1Weights[i-3];
    }

    // B1
    for(int i = 9; i < 11; i++) {
        model.data[i] = layer2Biases[i-9];
    }

    // W23
    for(int i = 15; i < 19; i++) {
        model.data[i] = layer2Weights[i-15];
    }

    // B2
    for(int i = 19; i < 21; i++) {
        model.data[i] = layer3Biases[i-19];
    }

    // Run prediction
    float in[] = {0.5f, 0.2f, 0.3f};
    float out[2];
    cml_predictCPU(model, in, out);

    for(int i = 0; i < 2; i++) {
        printf("%0.4f ", out[i]);
    }
    printf("\n");

    return cml_withinMarginOfError(out[0], 27.6f, 0.125f) && cml_withinMarginOfError(out[1], 17.4f, 0.125f);
}

bool test_modelPredictCPURelu() {
    // Model Specs
    size_t numOflayers = 3;
    uint64 layerSizes[] = {3,2,2};
    cml_ActivationFnMetadata* activations = (cml_ActivationFnMetadata*)malloc(sizeof(cml_ActivationFnMetadata) * 2);
    for(size_t i = 0; i < numOflayers-1; i++) {
        activations[i] = cml_createActivationFnMetadataWithID(NULL, NULL, CML_RELU);
    }

    // Model
    cml_Model model = cml_createModel(numOflayers, layerSizes, activations);
    
    // Use and test
    printf("Layer Count: %lld\n", model.layerCount);
    printf("Scale: %lld\n", model.scale);
    for(size_t i = 0; i < model.layerCount; i++) {
        printf("Layer Size: %lld\n", model.layerSizes[i]);
    }

    // Set model weights and biases manually
    float layer1Weights[] = {-0.3f,1,2,3,-2,-3};
    float layer2Biases[] = {2,0};
    float layer2Weights[] = {1,2,3,4};
    float layer3Biases[] = {1,0.9f};
    // 3     6    2    2      2      4    2    2      2
    // L1 - W12 - B1 - A1 - Z1/L2 - W23 - B2 - A2 - Z2/L3
    
    // W12
    for(int i = 3; i < 9; i++) {
        model.data[i] = layer1Weights[i-3];
    }

    // B1
    for(int i = 9; i < 11; i++) {
        model.data[i] = layer2Biases[i-9];
    }

    // W23
    for(int i = 15; i < 19; i++) {
        model.data[i] = layer2Weights[i-15];
    }

    // B2
    for(int i = 19; i < 21; i++) {
        model.data[i] = layer3Biases[i-19];
    }

    // Run prediction
    float in[] = {0.5f, -0.2f, 0.7f};
    float out[2];
    cml_predictCPU(model, in, out);

    for(int i = 0; i < 2; i++) {
        printf("%0.4f ", out[i]);
    }
    printf("\n");

    return cml_withinMarginOfError(out[0], 1.05f, 0.001f) && cml_withinMarginOfError(out[1], 1.0f, 0.001f);
}

#include <cml/device/GPU.h>
bool test_modelPredictGPULinear() {
    printf("==[ GPU test ]==\n");
    // Model Specs
    size_t numOflayers = 3;
    uint64 layerSizes[] = {3,2,2};
    cml_ActivationFnMetadata* activations = (cml_ActivationFnMetadata*)malloc(sizeof(cml_ActivationFnMetadata) * 2);
    for(size_t i = 0; i < numOflayers-1; i++) {
        activations[i] = cml_createActivationFnMetadataWithID(NULL, NULL, CML_LINEAR);
    }

    cml_GPU gpu = cml_simpleSetupGPU();
    printf("GPU setup successful\n");

    // Model
    cml_Model model = cml_createModel(numOflayers, layerSizes, activations);
    
    // Use and test
    printf("Layer Count: %lld\n", model.layerCount);
    printf("Scale: %lld\n", model.scale);
    for(size_t i = 0; i < model.layerCount; i++) {
        printf("Layer Size: %lld\n", model.layerSizes[i]);
    }

    // Set model weights and biases manually
    float layer1Weights[] = {1,2,3,4,5,6};
    float layer2Biases[] = {1,2};
    float layer2Weights[] = {4,3,2,1};
    float layer3Biases[] = {2,1};
    // 3     6    2    2      2      4    2    2      2
    // L1 - W12 - B1 - A1 - Z1/L2 - W23 - B2 - A2 - Z2/L3
    
    // W12
    for(int i = 3; i < 9; i++) {
        model.data[i] = layer1Weights[i-3];
    }

    // B1
    for(int i = 9; i < 11; i++) {
        model.data[i] = layer2Biases[i-9];
    }

    // W23
    for(int i = 15; i < 19; i++) {
        model.data[i] = layer2Weights[i-15];
    }

    // B2
    for(int i = 19; i < 21; i++) {
        model.data[i] = layer3Biases[i-19];
    }

    // Run prediction
    float in[] = {0.5f, 0.2f, 0.3f};
    float out[2];
    cml_predictGPU(model, in, out, gpu);

    for(int i = 0; i < 2; i++) {
        printf("%0.4f ", out[i]);
    }
    printf("\n");

    cml_deleteModel(&model);
    for(size_t i = 0; i < numOflayers-1; i++) {
        cml_deleteActivationFnMetadata(&activations[i]);
    }
    cml_deleteGPU(&gpu);

    return cml_withinMarginOfError(out[0], 27.6f, 0.125f) && cml_withinMarginOfError(out[1], 17.4f, 0.125f);
}

bool cml_withinMarginOfError(const float actual, const float expected, const float acceptableDeviation) {
    return fabs(actual - expected) < acceptableDeviation;
}