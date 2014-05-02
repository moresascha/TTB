#pragma once
#include "globals.cuh"

struct Material
{
    float3 diffuseI;
    float3 ambientI;
    float3 specularI;
    float _specExp;
    int _texId;

    int _mirror;
    float _reflectionIndex;
    float _fresnel_r;
    float _fresnel_t;
    float _alpha;
    float _reflectivity;

    __device__ float reflectivity(void) const
    {
        return _reflectivity;
    }

    __device__ bool isMirror(void) const
    {
        return _mirror > 0;
    }
    
    __device__ float fresnel_r(void) const
    {
        return _fresnel_r;
    }

    __device__ float fresnel_t(void) const
    {
        return _fresnel_t;
    }

    __device__ float specularExp(void) const
    {
        return _specExp;
    }

    __device__ bool isTransp(void) const
    {
        return _alpha < 1;
    }

    __device__ float alpha(void) const
    {
        return _alpha;
    }

    __device__ int texId(void) const
    {
        return _texId;
    }

    __device__ float reflectionIndex(void) const
    {
        return _reflectionIndex;
    }
};
