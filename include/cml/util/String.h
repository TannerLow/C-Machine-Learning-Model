#ifndef CML_STRING_H
#define CML_STRING_H

#include <stddef.h>

typedef struct {
    char* data;
    size_t size;
} cml_String;

cml_String cml_createString(const char* data, const size_t size);
cml_String cml_createNewString(const size_t size);

// Assumption: string.data is a malloc pointer that can be freed with free()
void cml_deleteString(cml_String* string);

// Serilization:
// bytes: [sizeof(size_t)][size]
// map  : [size][data]
cml_String cml_serializeString(cml_String string);

// You will need to know sizeof(size_t) that the string was serialized on
// in order to properly deserialize
cml_String cml_deserializeString(const char* serializedString, const size_t sizeofSizeT);

#endif // CML_STRING_H