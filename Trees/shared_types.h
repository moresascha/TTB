#pragma once
#include "ct.h"

#define MAX_ELEMENTS_PER_LEAF 16

__forceinline CTuint GenerateDepth(CTuint N)
{
    return 2;//N > 16 ? (CTuint)(8.5 + 1.3 * log(N)) : 1; //PBR
}

struct GeometryRange
{
    CTuint start;
    CTuint end;
};