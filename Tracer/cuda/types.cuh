#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cutil_inline.h>

#undef DOUBLE_PREC

#undef KEPLER

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#if defined DOUBLE_PREC
    typedef double Real;
    typedef double2 Real2;
    typedef double3 Real3;
    typedef double4 Real4;
    #define make_real3 make_double3
    #define make_real4 make_double4
#else
    typedef float Real;
    typedef float2 Real2;
    typedef float3 Real3;
    typedef float4 Real4;
    #define make_real3 make_float3
    #define make_real4 make_float4
#endif

typedef Real3 Position;
typedef Real3 Normal;
typedef Real2 TexCoord;

typedef unsigned char byte;
typedef unsigned int uint;

struct Material;
struct Triangles;
struct TraceResult;
struct Ray;
struct BBox;
struct cuTextureObj;

typedef void (*Shader)(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mats);

struct TreeNodes
{
    uint* contentCount;
    uint* content;
    uint* _left;
    uint* _right;
    uint* contentStart;
    uint* leafIndex;
    byte* isLeaf;
    byte* splitAxis;
    Real* split;

    __device__ uint left(uint index) const
    {
        return _left[index]; //index + 1;
    }

    __device__ uint right(uint index) const
    {
        return _right[index];
    }
};
