#pragma once
#include "globals.cuh"

extern "C" __device__ void debugShadeTC(const TraceResult& result, uint id, float4* color, const Ray& r, const Material& mats);

extern "C" __device__ void debugShadeNormal(const TraceResult& result, uint id, float4* color, const Ray& r, const Material& mats);

extern "C" __device__ void phongShade(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat);

extern "C" __device__ void shade(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat);