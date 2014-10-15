#include <cutil_math.h>
#include "globals.cuh"
#include "geometry.cuh"
#include "shading.cuh"
#include "texturing.cuh"
#include "lighting.cuh"
#include "traversing.cuh"
#include "material.cuh"
#include "traversing.cuh"

__device__ void texIndex(const TraceResult& result, uint id, float4* color, const Ray& r, const Material& mats)
{
    color[id] = make_float4(mats.texId() / 10.0f, 0,0,0);
}

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

__device__ void red(const TraceResult& result, uint id, float4* color, const Ray& r, const Material& mats)
{
    color[id] = make_float4(1,0,0,0);
}

__device__ void directionalLightOnly(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat)
{
    Real3 normal = getGeometry().getTrianglelNormal(result.triIndex, result.bary);
    Real3 pos = getGeometry().getTrianglelHitPos(result.triIndex, result.bary);

    float3 c = directionLighting(-SUN_POS, normal);

    c *= ray.rayWeight;

    color[id].x = color[id].x * (1 - ray.rayWeight) + c.x;
    color[id].y = color[id].y * (1 - ray.rayWeight) + c.y;
    color[id].z = color[id].z * (1 - ray.rayWeight) + c.z;
}

__device__ void phongShade(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat)
{
    Real3 normal = getGeometry().getTrianglelNormal(result.triIndex, result.bary);
    Real3 pos = getGeometry().getTrianglelHitPos(result.triIndex, result.bary);

    float3 c = make_float3(1,1,1); // * directionLighting(-SUN_POS, normal); //phongLighting(ray.getOrigin(), pos, SUN_POS, normal, &mat);

    if(mat.texId() != NO_TEXTURE)
    {
        Real2 texCoords = getGeometry().getTrianglelTC(result.triIndex, result.bary);
        float4 col = readTexture(mat.texId(), texCoords);
        c.x = col.x;
        c.y = col.y;
        c.z = col.z;
        c*= mat.diffuseI;
    }
    else
    {
        //c = mat.diffuseI;
    }

    //c *= directionLighting(-SUN_POS, normal);

    for(int i = 0; i < getLightCount(); ++i)
    {
        const Light& light = getLight(i);
        float3 d = light.position - pos;
        float distSquared = dot(d, d);

        if(distSquared > light.intensity.y * light.intensity.y) 
        {
            continue;
        }

        float in = fmaxf(0, 1 - distSquared / (light.intensity.y * light.intensity.y));
        c *= in * light.color * light.intensity.x * phongLighting(ray.getOrigin(), pos, light.position, normal, &mat);
        //c += light.color * phongLighting(ray.getOrigin(), pos, light.position, normal, &mat);
    }

    c *= ray.rayWeight;

    color[id].x = color[id].x * (1 - ray.rayWeight) + c.x;
    color[id].y = color[id].y * (1 - ray.rayWeight) + c.y;
    color[id].z = color[id].z * (1 - ray.rayWeight) + c.z;
}

__constant__ Shader g_shaderPtr[6] = {&phongShade, &directionalLightOnly, &debugShadeNormal, &debugShadeTC, &red, &texIndex};

extern "C" __device__ void shade(const TraceResult& result, uint id, float4* color, const Ray& ray, const Material& mat)
{
    g_shaderPtr[getCurrentShader()](result, id, color, ray, mat);
}