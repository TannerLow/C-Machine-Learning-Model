#ifndef CML_DYNAMIC_ARRAY_H
#define CML_DYNAMIC_ARRAY_H

#include <stddef.h>

// Amount cml_DynamicArray will grow when pushing to a full capacity array.
extern size_t cml_dynamicArrayGrowAmount;

/* Dyanamically sized array. Only grows by a fixed amount when
 * pushing to a full capacity array. It is not intended to be
 * used as a C++ style vector; it is for simpler use cases where
 * optimizating for speed is not important.
 */
typedef struct {
    void* data;
    size_t itemSize;
    size_t size;
    size_t capacity;
} cml_DynamicArray;

cml_DynamicArray cml_createDynamicArray(size_t initialCapacity, size_t itemSize);
void cml_deleteDynamicArray(cml_DynamicArray* array);

size_t cml_dynamicArrayPush(cml_DynamicArray* array, void* element);
void* cml_dynamicArrayGet(cml_DynamicArray* array, size_t index);

#endif // CML_DYNAMIC_ARRAY_H