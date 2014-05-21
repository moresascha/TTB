#pragma once

enum CT_RESULT
{
    CT_SUCCESS = 0,
    CT_INVALID_VALUE = 1,
    CT_MEMORY_ALLOC_FAILURE = 2,
    CT_INVALID_ENUM = 3,
    CT_NOT_YET_IMPLEMENTED = 4,
    CT_INVALID_OPERATION = 5,
    CT_INTERFACE_NOT_FOUND = 6,
    CT_OPERATION_NOT_SUPPORTED = 7
};

#define CT_API __stdcall
#define CT_EXPORT __declspec(dllexport)

#define CT_INLINE __forceinline

//CT Init flags
#define CT_ENABLE_CUDA_ACCEL (1 << 0)
#define CT_TREE_ENABLE_DEBUG_LAYER (1 << 1)

#define CT_CREATE_TREE_CPU (1 << 0)
#define CT_CREATE_TREE_GPU (1 << 1)

#define __uuidof(clazz) __uuid<##clazz>()

#define add_uuid_header(clazz) static CTuuid uuid(void) { return #clazz; }