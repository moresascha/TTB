#pragma once
#include "ct_def.h"
#include "ct.h"

#define DebugOut OutputDebugStringA

CT_EXPORT const char* CT_API CTGetErrorString
    (
    CT_RESULT error
    );

#define CT_SAFE_CALL(call) \
{ \
    CT_RESULT error = call; \
    if(CT_SUCCESS != error) { \
        const char* errStr = CTGetErrorString(error); \
        DebugOut("CT_ERROR: "); \
        DebugOut(errStr); \
        DebugOut(", "); \
        DebugOut(__FILE__); \
        DebugOut(" "); \
        DebugOut(#call); \
        DebugOut("\n"); \
        \
    } \
}