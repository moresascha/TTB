#pragma once
#include "ct.h"

#define MAX_ELEMENTS_PER_LEAF 64
#define INVALID_SAH FLT_MAX
#define IS_INVALD_SAH(sah) (sah == FLT_MAX)

__forceinline CTuint GenerateDepth(CTuint N)
{
    return N > 16 ? (CTuint)(8.5 + 1.3 * log(N)) : 1; //PBR
}

struct GeometryRange
{
    CTuint start;
    CTuint end;
};