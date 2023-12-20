#ifndef CML_KERNEL_ARG_H
#define CML_KERNEL_ARG_H

#include <CL/cl.h>

#include <stddef.h>

#define CML_BUFFER_SIZE

typedef struct {
    void* data;
    size_t size;
} cml_KernelArg;

#endif // CML_KERNEL_ARG_H