#include <cutil_math.h>
#include "globals.cuh"
#include "geometry.cuh"
#include "shading.cuh"
#include "texturing.cuh"
#include "lighting.cuh"
#include "traversing.cuh"
#include "material.cuh"
#include "traversing.cuh"

__constant__ uint g_shader = 0;

__device__ void debugShadeTC(const TraceResult& result, uint id, float4* color, const Ray& r, const Material& mats)
{
    Real2 tc = getGeometry().getTrianglelTC(result.triIndex, result.bary);
    color[id] = make_float4(tc.x, tc.y, 0, 0);
}

__device__ void debugShadeNormal(const TraceResult& result, uint id, float4* color, const Ray& r, const Material& mats)
{
    Real3 n = normalize(getGeometry().getTrianglelNormal(result.triIndex, result.bary));
    color[id] = 0.5 + 0.5 * make_float4(n.x, n.y, n.z, 0);
}

__device__ void phongShade(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat)
{
    Real3 normal = getGeometry().getTrianglelNormal(result.triIndex, result.bary);
    Real3 pos = getGeometry().getTrianglelHitPos(result.triIndex, result.bary);

    float3 c = phongLighting(ray.getOrigin(), pos, SUN_POS, normal, &mat);

    if(mat.texId() != NO_TEXTURE)
    {
        Real2 texCoords = getGeometry().getTrianglelTC(result.triIndex, result.bary);
        float4 col = readTexture(mat.texId(), texCoords);

        c.x *= col.x;
        c.y *= col.y;
        c.z *= col.z;
    }

    c *= ray.rayWeight;

    color[id].x = color[id].x * (1 - ray.rayWeight) + c.x;
    color[id].y = color[id].y * (1 - ray.rayWeight) + c.y;
    color[id].z = color[id].z * (1 - ray.rayWeight) + c.z;
}

__constant__ Shader g_shaderPtr[3] = {&phongShade, &debugShadeNormal, &debugShadeTC};

extern "C" __device__ void shade(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat)
{
    g_shaderPtr[g_shader](result, id, color, ray, mat);
}