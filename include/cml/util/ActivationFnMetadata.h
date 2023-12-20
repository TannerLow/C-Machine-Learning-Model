#ifndef CML_ACTIVATION_FN_METADATA_H
#define CML_ACTIVATION_FN_METADATA_H

#include "../ActivationFunction.h"

#include "String.h"

typedef struct {
    cml_String gpuProgramFilename;
    cml_String gpuKernelName;
    enum cml_ActivationID activationID;
} cml_ActivationFnMetadata;

// filenames should be null-terminated strings
cml_ActivationFnMetadata cml_createActivationFnMetadata(const char* gpuProgramFilename, const char* gpuKernelName);
// filenames should be null-terminated strings
cml_ActivationFnMetadata cml_createActivationFnMetadataWithID(const char* gpuProgramFilename, const char* gpuKernelName, enum cml_ActivationID id);
void cml_deleteActivationFnMetadata(cml_ActivationFnMetadata* metadata);

cml_ActivationFnMetadata cml_duplicateActivationFnMetadata(const cml_ActivationFnMetadata original);

// Assumption: activationID will be able to fit in 1 byte (unsigned char)
// Used to get number of bytes needed for given metadata during serialization process
size_t cml_getActivationFnMetadataSize(cml_ActivationFnMetadata metadata);

// Assumption: activationID will be able to fit in 1 byte (unsigned char)
// Serilization:
// bytes: [sizeof(size_t)][gpuProgramFilename.size][sizeof(size_t)][gpuKernelName.size][1]
// map  : [gpuProgramFilename.size][gpuProgramFilename][gpuKernelName.size][gpuKernelName][activationID]
cml_String cml_serializeActivationFnMetadata(cml_ActivationFnMetadata metadata);

// You will need to know sizeof(size_t) that the string was serialized on in order to properly deserialize
cml_ActivationFnMetadata cml_deserializeActivationFnMetadata(const char* metadata, const size_t sizeofSizeT);

#endif // CML_ACTIVATION_FN_METADATA_H