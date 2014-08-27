#pragma once
#include "globals.cuh"

extern "C" __device__ Real3 _refract(const Real3& i, const Real3& n, Real eta);

extern "C" __device__ Real3 refract(const Real3& I, const Real3& N, Real n1, Real n2);

extern "C" __device__ Real reflectance(const Real3& n, const Real3& i, Real n1, Real n2);

//reflectance approxi
extern "C" __device__ Real schlick(const Real3& n, const Real3& i, Real n1, Real n2, Real fresnel);

extern "C" __device__ Real Reflectance(const Real3& i, const Real3& n, Real n1, Real n2, Real fresnel);

extern "C"__device__ float3 phongLighting(const float3& eye, const float3& world, const float3& lightPos, const float3& normal, const Material* material);

extern "C" __device__ float3 directionLighting(const float3& lightPos, const float3& normal);