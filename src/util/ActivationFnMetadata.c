#include <cml/util/ActivationFnMetadata.h>

#include <string.h>
#include <assert.h>

cml_ActivationFnMetadata cml_createActivationFnMetadata(const char* gpuProgramFilename, const char* gpuKernelName) {
    return cml_createActivationFnMetadataWithID(gpuProgramFilename, gpuKernelName, CML_NONE);
}

cml_ActivationFnMetadata cml_createActivationFnMetadataWithID(const char* gpuProgramFilename, const char* gpuKernelName, enum cml_ActivationID id) {
    assert((gpuProgramFilename != NULL && gpuKernelName != NULL) || id != CML_NONE);

    cml_ActivationFnMetadata metadata;
    // need to 0 these out for size calculation in event where these strings are never created
    metadata.gpuProgramFilename.size = 0;
    metadata.gpuKernelName.size = 0;

    if(gpuProgramFilename != NULL) {
        metadata.gpuProgramFilename = cml_createString(gpuProgramFilename, strlen(gpuProgramFilename));
    }
    if(gpuKernelName != NULL) {
        metadata.gpuKernelName = cml_createString(gpuKernelName, strlen(gpuKernelName));
    }
    metadata.activationID = id;

    return metadata;
}

void cml_deleteActivationFnMetadata(cml_ActivationFnMetadata* metadata) {
    assert(metadata != NULL);
    
    if(metadata->gpuProgramFilename.size != 0) {
        cml_deleteString(&metadata->gpuKernelName);
    }
    if(metadata->gpuKernelName.size != 0) {
        cml_deleteString(&metadata->gpuProgramFilename);
    }
    metadata->activationID = CML_NONE;
}

cml_ActivationFnMetadata cml_duplicateActivationFnMetadata(const cml_ActivationFnMetadata original) {
    cml_ActivationFnMetadata metadata;

    size_t strSize = original.gpuProgramFilename.size;
    metadata.gpuProgramFilename = cml_createNewString(strSize);
    strncpy(metadata.gpuProgramFilename.data, original.gpuProgramFilename.data, strSize);

    strSize = original.gpuKernelName.size;
    metadata.gpuKernelName = cml_createNewString(strSize);
    strncpy(metadata.gpuKernelName.data, original.gpuKernelName.data, strSize);

    metadata.activationID = original.activationID;
    
    return metadata;
}

size_t cml_getActivationFnMetadataSize(cml_ActivationFnMetadata metadata) {
    size_t sizeBytes = 0;
    sizeBytes += metadata.gpuProgramFilename.size + sizeof(size_t); // serialized size of cml_String
    sizeBytes += metadata.gpuKernelName.size + sizeof(size_t);      // serialized size of cml_String
    // this size fn is for serialization purposes so only 1 byte needed for activation id
    sizeBytes += 1;

    return sizeBytes;
}

cml_String cml_serializeActivationFnMetadata(cml_ActivationFnMetadata metadata) {
    cml_String gpuProgramFilename = cml_serializeString(metadata.gpuProgramFilename);
    cml_String gpuKernelName = cml_serializeString(metadata.gpuKernelName);
    cml_String serializedMetadata = cml_createNewString(gpuProgramFilename.size + gpuKernelName.size + 1);

    memcpy(serializedMetadata.data, gpuProgramFilename.data, gpuProgramFilename.size);
    memcpy(serializedMetadata.data + gpuProgramFilename.size, gpuKernelName.data, gpuKernelName.size);
    unsigned char activationID = (unsigned char)metadata.activationID;
    serializedMetadata.data[serializedMetadata.size-1] = activationID;

    cml_deleteString(&gpuProgramFilename);
    cml_deleteString(&gpuKernelName);

    return serializedMetadata;
}

cml_ActivationFnMetadata cml_deserializeActivationFnMetadata(const char* serializedMetadata, const size_t sizeofSizeT) {
    cml_ActivationFnMetadata metadata;
    metadata.gpuProgramFilename = cml_deserializeString(serializedMetadata, sizeofSizeT);
    metadata.gpuKernelName = cml_deserializeString(serializedMetadata + metadata.gpuProgramFilename.size + sizeof(size_t), sizeofSizeT);
    metadata.activationID = (enum cml_ActivationID)serializedMetadata[metadata.gpuProgramFilename.size + metadata.gpuKernelName.size + 2*sizeof(size_t)];
    return metadata;
}