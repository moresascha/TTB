#pragma once
#include "ct_def.h"
#include "ct.h"
#include <stdlib.h>

#define DebugOut OutputDebugStringA

CT_EXPORT const char* CT_API CTGetErrorString
    (
    CT_RESULT error
    );

/*char lineBuffer[2048]; \
        _itoa(__LINE__, lineBuffer, 10); \
        DebugOut(lineBuffer); \
        DebugOut(" "); \*/

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
        exit(-1); \
        \
    } \
}