#include "ct_error.h"

const char* CT_API CTGetErrorString(CT_RESULT error)
{
    switch(error)
    {
    case CT_SUCCESS : return "CT_SUCCESS";
    case CT_INVALID_VALUE : return "CT_INVALID_VALUE";
    case CT_MEMORY_ALLOC_FAILURE : return "CT_MEMORY_ALLOC_FAILURE";
    case CT_INVALID_ENUM : return "CT_INVALID_ENUM";
    case CT_NOT_YET_IMPLEMENTED : return "CT_NOT_YET_IMPLEMENTED";
    case CT_INTERFACE_NOT_FOUND : return "CT_INTERFACE_NOT_FOUND";
    case CT_INVALID_OPERATION : return "CT_INVALID_OPERATION";
    case CT_OPERATION_NOT_SUPPORTED : return "CT_OPERATION_NOT_SUPPORTED";
    default : break;
    }
    return "Unknown Error";
}