#ifndef CML_KERNEL
#define CML_KERNEL

#include <CL/cl.h>

#include <stddef.h>
#include <stdio.h>

typedef struct {
    const char* kernelName;
    cl_program program;
} cml_Kernel;

cml_Kernel cml_createKernel(const char* kernelName, cl_program program);
void cml_deleteKernel(cml_Kernel* kernel);

#endif // CML_KERNEL