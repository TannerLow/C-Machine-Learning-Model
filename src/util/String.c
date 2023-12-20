#include <cml/util/String.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>


cml_String cml_createString(const char* data, const size_t size) {
    assert(data != NULL);
    assert(size > 0);

    cml_String string;
    string.data = (char*)malloc(size);
    memcpy(string.data, data, size);
    string.size = size;
    return string;
}

cml_String cml_createNewString(const size_t size) {
    cml_String string;
    string.data = size > 0 ? (char*)malloc(size) : NULL;
    string.size = size;
    return string;
}

void cml_deleteString(cml_String* string) {
    assert(string != NULL);

    free(string->data);
    string->data = NULL;
    string->size = 0;
}

cml_String cml_serializeString(cml_String string) {
    cml_String serializedString = cml_createNewString(string.size + sizeof(size_t));
    memcpy(serializedString.data, &string.size, sizeof(size_t));
    memcpy(serializedString.data + sizeof(size_t), string.data, string.size);
    return serializedString;
}

cml_String cml_deserializeString(const char* serializedString, const size_t sizeofSizeT) {
    assert(serializedString != NULL);

    size_t size = 0;
    memcpy(&size, serializedString, sizeofSizeT);
    cml_String string = cml_createNewString(size);
    memcpy(string.data, serializedString + sizeofSizeT, size);
    return string;
}