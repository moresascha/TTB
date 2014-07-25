#pragma once
#include "ct.h"
#include <DeviceBuffer.h>

#define make_real4 make_float4
#define make_real3 make_float3

struct _AABB;

extern "C" void cudaCreateTriangleAABBs(CTreal3* tris, _AABB* aabbs, CTuint N, cudaStream_t pStream = NULL);

//extern "C" void cudaGetSceneBBox(nutty::DeviceBuffer<_AABB>& aabbs, CTuint N, _AABB& aabb);

extern "C" void cudaTransformVector
    (
    nutty::DeviceBuffer<CTreal3>::iterator& v_in, 
    nutty::DeviceBuffer<CTreal3>::iterator& v_out, 
    const CTreal4* matrix, 
    CTuint N,
    cudaStream_t pStream = NULL);