#pragma once
#include "globals.cuh"

Real dot(Real3 a, Real3 b);
Real3 cross(Real3 a, Real3 b);

enum RT_CullMode
{
    eRT_No = 0,
    eRT_Back = 1,
    eRT_Front = 2
};

__device__ __forceinline byte no_cull(const Real3& e0, const Real3& e1, const Real3& dir)
{
    return 0;
}

__device__ __forceinline byte cull_back(const Real3& e0, const Real3& e1, const Real3& dir)
{
    return dot(cross(e0, e1), dir) < 0;
}

__device__ __forceinline byte cull_front(const Real3& e0, const Real3& e1, const Real3& dir)
{
    return dot(cross(e0, e1), dir) >= 0;
}