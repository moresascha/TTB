#pragma once
#include "globals.cuh"

__forceinline __device__ __host__ Real getAxis(const Real4& vec, byte axis)
{
    switch(axis)
    {
        case 0 : return vec.x;
        case 1 : return vec.y;
        case 2 : return vec.z;
        case 3 : return vec.w;
    }
    return 0;
}

__forceinline __device__ __host__ void setAxis(Real3& vec, byte axis, Real v)
{
    switch(axis)
    {
        case 0 : vec.x = v; break;
        case 1 : vec.y = v; break;
        case 2 : vec.z = v; break;
    }
}

__forceinline __device__ __host__ Real getAxis(const Real3& vec, byte axis)
{
    switch(axis)
    {
        case 0 : return vec.x;
        case 1 : return vec.y;
        case 2 : return vec.z;
    }
    return 0;
}