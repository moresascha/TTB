#include <windows.h>
#include <assert.h>
#include <iostream>

#define PRINTF OutputDebugStringA

void __printf(char* format, ...)
{
    char buffer[2048];

    va_list ap;
    va_start(ap, format);

    vsprintf_s(buffer, format, ap);

    va_end(ap);
    PRINTF(buffer);
}
