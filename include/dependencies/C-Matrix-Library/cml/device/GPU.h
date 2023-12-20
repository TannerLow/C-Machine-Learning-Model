#ifndef CML_GPU_H
#define CML_GPU_H

#include "../util/Kernel.h"
#include "../util/DynamicArray.h"
#include "../util/Program.h"

#include <CL/cl.h>

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    cl_device_id* deviceIds;
    size_t count;
} cml_DeviceArray;

cml_DeviceArray cml_getGPUsWithCUDASupport(const uint8_t maxPlatforms, const uint8_t maxDevices);
void cml_deleteDeviceArray(cml_DeviceArray* deviceArray);

typedef struct {
    cl_kernel kernel;
    const char* kernalName;
} cml_KernelMapEntry;

typedef struct {
    cl_device_id gpu;
    cl_context context;
    cl_command_queue commands;
    // cl_program program;
    // bool programLoaded;
    cml_DynamicArray programMap; // of cl_program
    // cml_DynamicArray programLoadedMap; // of bool
    cml_DynamicArray kernelMap; // of cml_KernelMapEntry
    cml_DynamicArray buffers; // of cl_mem
} cml_GPU;

typedef cl_mem cml_GPUBuffer;

cml_GPU cml_createGPU(cl_device_id device);
void cml_deleteGPU(cml_GPU* gpu);
cl_program cml_loadGPUProgram(cml_GPU* gpu, const cml_Program program);
void cml_createGPUKernel(cml_GPU* gpu, const cml_Kernel kernel);
cml_GPUBuffer* cml_allocateGPUBuffer(cml_GPU* gpu, size_t size);
// void cml_freeGPUBuffer(cml_GPU* gpu, cml_GPUBuffer* buffer);

cml_GPU cml_simpleSetupGPU();

#endif // CML_GPU_H