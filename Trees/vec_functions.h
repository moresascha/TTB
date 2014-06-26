#pragma once
#include "ct.h"
#include "shared_types.h"

template<typename BBox>
__inline void __device__ __host__ splitAABB(const BBox* aabb, CTreal split, CTbyte axis, BBox* l, BBox* r)
{
    l->m_max = aabb->m_max;
    l->m_min = aabb->m_min;
    r->m_max = aabb->m_max;
    r->m_min = aabb->m_min;
    switch(axis)
    {
    case 0:
        {
            l->m_max.x = split; r->m_min.x = split; 
        } break;
    case 1:
        {
            l->m_max.y = split; r->m_min.y = split; 
        } break;
    case 2:
        {
            l->m_max.z = split; r->m_min.z = split; 
        } break;
    }
}

__forceinline __device__ __host__ int getLongestAxis(const CTreal3& mini, const CTreal3& maxi) 
{
    CTreal3 v = maxi - mini;
    float m = fmaxf(v.x, fmaxf(v.y, v.z));
    return m == v.x ? 0 : m == v.y ? 1 : 2;
    /*CTreal dx = maxi.x - mini.x;
    CTreal dy = maxi.y - mini.y;
    CTreal dz = maxi.z - mini.z;

    if(dy < dz && dx < dz)
    {
        return 2;
    }

    if(dy < dx && dz < dx)
    {
        return 0;
    }

    return 1;*/
}

__device__ __host__ __forceinline  int fls(int f)
{
    int order;
    for (order = 0; f != 0;f >>= 1, order++);
    return order;
}

__device__ __host__ __forceinline int ilog2(int f)
{
    return fls(f) - 1;
}

__device__ __host__ __forceinline enum CT_SPLIT_AXIS getLongestAxis(const CTreal3& v)
{
    float m = fmaxf(v.x, fmaxf(v.y, v.z));
    return m == v.x ? eCT_X : m == v.y ? eCT_Y : eCT_Z;
}

__device__ __host__ __forceinline float getAxis(const CTreal3& vec, CTbyte axis)
{
    switch(axis)
    {
    case 0 : return vec.x;
    case 1 : return vec.y;
    case 2 : return vec.z;
    }
    return 0;
}

__device__ __host__ __forceinline float getAxis(const CTreal4& vec, CTbyte axis)
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

__device__ __host__ __forceinline void setAxis(CTreal3& vec, CTbyte axis, CTreal v)
{
    switch(axis)
    {
    case 0 : vec.x = v; return;
    case 1 : vec.y = v; return;
    case 2 : vec.z = v; return;
    }
}

template <
    typename AABB
>
__device__ __host__ __forceinline float3 getAxisScale(const AABB& aabb)
{
    return make_float3(aabb.GetMax().x - aabb.GetMin().x, aabb.GetMax().y - aabb.GetMin().y, aabb.GetMax().z - aabb.GetMin().z);
}

template <
    typename __AABB
>
__device__ __host__ __forceinline  float getArea(const __AABB& aabb)
{
    float3 axisScale = getAxisScale(aabb);
    return 2 * axisScale.x * axisScale.y + 2 * axisScale.x * axisScale.z + 2 * axisScale.y * axisScale.z;
}

template <
    typename __AABB
>
__device__ __host__ CTreal __inline getSAH(const __AABB& node, CTint axis, CTreal split, CTint primBelow, CTint primAbove, CTreal traversalCost = 0.125f, CTreal isectCost = 1)
{
    CTreal cost = INVALID_SAH;

    if(split > getAxis(node.GetMin(), axis) && split < getAxis(node.GetMax(), axis))
    {
        CTreal3 axisScale = getAxisScale(node);
        CTreal invTotalSA = 1.0f / getArea(node);
        CTint otherAxis0 = (axis+1) % 3;
        CTint otherAxis1 = (axis+2) % 3;
        CTreal belowSA = 
            2 * 
            (getAxis(axisScale, otherAxis0) * getAxis(axisScale, otherAxis1) + 
            (split - getAxis(node.GetMin(), axis)) * 
            (getAxis(axisScale, otherAxis0) + getAxis(axisScale, otherAxis1)));

        CTreal aboveSA = 
            2 * 
            (getAxis(axisScale, otherAxis0) * getAxis(axisScale, otherAxis1) + 
            (getAxis(node.GetMax(), axis) - split) * 
            (getAxis(axisScale, otherAxis0) + getAxis(axisScale, otherAxis1)));    

        CTreal pbelow = belowSA * invTotalSA;
        CTreal pabove = aboveSA * invTotalSA;
        CTreal bonus = (primAbove == 0 || primBelow == 0) ? 1.0f : 0.0f;
        cost = traversalCost + isectCost * (1.0f - bonus) * (pbelow * primBelow + pabove * primAbove);
    }
    return cost;
}