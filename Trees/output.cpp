#include "ct.h"
#include <assert.h>
#include <iostream>

#define PRINTF OutputDebugStringA

void __ct_printf(char* format, ...)
{
    int len = (int)strlen(format);
    char buffer[2048];

    va_list ap;
    va_start(ap, format);

    vsprintf_s(buffer, format, ap);

    va_end(ap);
    PRINTF(buffer);
}
