#pragma once
#include "globals.cuh"

struct TraceResult
{
    bool isHit;
    Real2 bary;
    uint triIndex;
};

struct Ray
{
    uint2 screenCoord;
    Real4 origin_min;
    Real4 dir_max;
    Real rayWeight;
#ifdef _DEBUG
    Real3 tmp;
#endif

    __device__ void setScreenCoords(const uint2& sc)
    {
        screenCoord = sc;
    }

    __device__ void setOrigin(const Real3& o)
    {
        origin_min.x = o.x;
        origin_min.y = o.y;
        origin_min.z = o.z;
    }

    __device__ void setDir(const Real3& d)
    {
        dir_max.x = d.x;
        dir_max.y = d.y;
        dir_max.z = d.z;
    }

    __device__ void setMax(Real m)
    {
        dir_max.w = m;
    }

    __device__ void setMin(Real m)
    {
        origin_min.w = m < 0 ? 0 : m;
    }

    __device__ Real3 getOrigin(void) const
    {
        return make_real3(origin_min.x, origin_min.y, origin_min.z);
    }

    __device__ Real3 getDir(void) const
    {
        return make_real3(dir_max.x, dir_max.y, dir_max.z);
    }

    __device__ Real* getMax(void)
    {
        return &dir_max.w;
    }

    __device__ Real* getMin(void)
    {
        return &origin_min.w;
    }

    void __device__ clampToBBox(void);
};