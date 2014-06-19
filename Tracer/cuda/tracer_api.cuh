#pragma once
#include "globals.cuh"
#include "geometry.cuh"
#include "material.cuh"
#include "traversing.cuh"

//todo: find api name

extern "C" void RT_BindTextureAtlas(const cudaArray_t array);

extern "C" void RT_Init(unsigned int width, unsigned int height);

extern "C" void RT_Destroy(void);

extern "C" uint RT_GetLastRayCount(void);

extern "C" void RT_BindTree(TreeNodes& tree);

extern "C" void RT_SetViewPort(unsigned int width, unsigned int height);

extern "C" void RT_BindGeometry(Triangles& triangles);

extern "C" void RT_TransformNormals(Normal* normals, Normal* newNormals, Real4* matrix, size_t start, uint N);

extern "C" void RT_BindTextures(const cuTextureObj* textures, uint size);

extern "C" void RT_Trace(float4* colors, const float3* view, float3 eye, BBox& bbox);

extern "C" void RT_SetLightPos(const float3& light);

extern "C" void RT_GetRayInfo(std::string& info);

extern "C" void RT_SetRecDepth(int d);

extern "C" void RT_IncDepth(void);

extern "C" int RT_GetRecDepth(void);

extern "C" void RT_DecDepth(void);