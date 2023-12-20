#ifndef CML_MATRIX_MATH_GPU_H
#define CML_MATRIX_MATH_GPU_H

#include "Matrix.h"
#include "../device/GPU.h"

// enum cml_State{CML_UNINITIALIZED, CML_PRE_BUFFERS, CML_PRE_KERNEL_ARGS, CML_READY};

// // CML_UNINITIALIZED -> CML_PRE_BUFFERS
// void cml_setupGPU(cl_device_id gpu, cml_Kernel kernel);
// // CML_PRE_BUFFERS -> CML_PRE_KERNEL_ARGS
// void cml_setupGPUBuffers(cml_MatrixArray bufferData);
// // CML_PRE_KERNEL_ARGS -> CML_READY
// void cml_setKernelArgs(cml_KernelArg* kernelArgs, const size_t count);
// // Requires CML_READY
// void cml_matrixMultiplyGPU(const cml_Matrix* a, const cml_Matrix* b, cml_Matrix* c);

void cml_matrixMultiplyGPU(cml_GPU* gpu, const cml_Matrix* a, const cml_Matrix* b, cml_Matrix* out);
void cml_matrixAddRowGPU(cml_GPU* gpu, const cml_Matrix matrix, const cml_Matrix row, cml_Matrix* out);
void cml_matrixReluGPU(cml_GPU* gpu, const cml_Matrix matrix, cml_Matrix* out);

#endif // CML_MATRIX_MATH_GPU_H