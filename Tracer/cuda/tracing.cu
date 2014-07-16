
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#include <cutil_math.h>
#include "globals.cuh"
#include <Nutty.h>
#include <DeviceBuffer.h>
#include <Scan.h>
#include <Copy.h>
#include <Fill.h>
#include <cuda/cuda_helper.h>
#include "../print.h"
#include "../double_math.h"
#include "../texture_array.h"
#include "lighting.cuh"
#include <list>
#include <queue>
#include <map>
#include "geometry.cuh"
#include "traversing.cuh"
#include "vector_functions.cuh"
#include "culling.cuh"
#include "material.cuh"
#include "texturing.cuh"
#include "shading.cuh"
#include "tracer_api.cuh"

__constant__ Real3 g_sunPos;

__constant__ float3 g_view[3];

__constant__ BBox g_bbox;

__constant__ TreeNodes g_tree;

__constant__ Triangles g_geometry;

extern "C" __device__ const Triangles& getGeometry(void)
{
    return g_geometry;
}

extern "C" __device__ const BBox& getBBox(void)
{
    return g_bbox;
}

extern "C" __device__ Real3 getSunPos(void)
{
    return g_sunPos;
}

#undef COMPUTE_SHADOW
#undef COMPUTE_REFRACTION
#define RECURSION 2
#define MAX_RECURSION 4
#define RAY_WEIGHT_THRESHOLD 0.01

#define AIR_RI 1.00029
#define GLASS_RI 1.52

#define GLASS_TO_AIR (GLASS_RI / AIR_RI)
#define AIR_TO_GLASS (AIR_RI / GLASS_RI)

__device__ __forceinline int intersectP(const Real3& eye, const Real3& ray, const Real3& boxmin, const Real3& boxmax, Real* tmin, Real* tmax) 
{
    Real t0 = *tmin; Real t1 = *tmax;

    Real3 invRay = 1.0 / ray;

#pragma unroll 3
    for(byte i = 0; i < 3; ++i) 
    {
        float tNear = (getAxis(boxmin, i) - getAxis(eye, i)) * getAxis(invRay, i);
        float tFar = (getAxis(boxmax, i) - getAxis(eye, i)) * getAxis(invRay, i);

        if(tNear > tFar) 
        {
            float tmp = tNear;
            tNear = tFar;
            tFar = tmp;
        }

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;

        if(t0 > t1) return 0;
    }
    *tmin = t0;
    *tmax = t1;
    return 1;
}

template <typename FC>
__device__  int hitsTriangle(const Real3& rayOrigin, Real min, Real max, const Real3& rayDirection, const Position* positions, Real* hitDepth, Real2& bary, FC cf, byte* isBackFace)
{
    Real3 p0 = positions[0];
    Real3 e1 = positions[1] - p0;
    Real3 e2 = positions[2] - p0;

    Real3 s1 = cross(rayDirection, e2);
    
    *isBackFace = (byte)(dot(cross(e1, e2), rayDirection) < 0);

    Real div = dot(e1, s1);
    if(div == 0) // || cf(e1, e2, rayDirection))
    {
        return 0;
    }
        
    Real invDiv = 1.0f / div;
        
    Real3 d = rayOrigin - p0;

    Real b1 = dot(d, s1) * invDiv;

    if(b1 < 0 || b1 > 1) 
    {
        return 0;
    }
        
    Real3 s2 = cross(d, e1);

    Real b2 = dot(rayDirection, s2) * invDiv;
        
    if(b2 < 0 || b2 + b1 > 1) 
    { 
        return false;
    }
        
    Real t = dot(e2, s2) * invDiv;
        
    if(t < min || t > max) 
    {
        return 0;
    }

    bary.x = b1;
    bary.y = b2;

    *hitDepth = t;

    return 1;
}

#define USE_SHORT_STACK

#ifdef USE_SHORT_STACK
struct ToDo 
{
    uint nodeIndex;
    Real tmax;
    Real tmin;
};
#endif

template <typename CF>
__device__ void traverse(const TreeNodes& n, const Real3& eye, const Real3& ray, TraceResult& hit, Real rayMin, Real rayMax, CF cf)
{ 
    hit.isHit = 0;

    Real tmin;
    Real tmax;
    tmin = tmax = rayMin;
    int pushDown = 1;
    uint nodeIndex = 0;

    uint root = 0;

#ifdef USE_SHORT_STACK
    int stackPos = 1;
    ToDo todo[16];
 
    todo[0].nodeIndex = 0;
    todo[0].tmin = rayMin;
    todo[0].tmax = rayMax;
#endif

    float t = FLT_MAX;

    while(tmax < rayMax) 
    {

#ifdef USE_SHORT_STACK
        if(stackPos == -1)
        {
            pushDown = 1;
            nodeIndex = root;
            tmin = tmax;
            tmax = rayMax;
        }
        else
        {
            stackPos--;
            nodeIndex = todo[stackPos].nodeIndex;
            tmin = todo[stackPos].tmin;
            tmax = todo[stackPos].tmax;
            pushDown = 0;
        }
#else
        uint nodeIndex = root;
        tmin = tmax;
        tmax = rayMax;
        pushDown = 1;
#endif
        
        while(!n.isLeaf[nodeIndex])
        {
            byte axis = n.splitAxis[nodeIndex];

            Real nsplit = n.split[nodeIndex];

            Real tsplit = (nsplit - getAxis(eye, axis)) / getAxis(ray, axis);

            int belowFirst = (getAxis(eye, axis) < nsplit) || ((getAxis(eye, axis) == nsplit) && (getAxis(ray, axis) >= 0));

            uint first, second;

            if(belowFirst)
            {
                first = n.left(nodeIndex);
                second = n.right(nodeIndex);
            } 
            else 
            {
                first = n.right(nodeIndex);
                second = n.left(nodeIndex);
            }

            if(tsplit > tmax || tsplit <= 0)
            {
                nodeIndex = first;
            } 
            else if(tsplit <= tmin) 
            {
                nodeIndex = second;
            }
            else
            {
#ifdef USE_SHORT_STACK
                todo[stackPos].nodeIndex = second;
                todo[stackPos].tmin = tsplit;
                todo[stackPos].tmax = tmax;
                stackPos++;
#endif
                nodeIndex = first;
                tmax = tsplit;
                pushDown = 0;
            }

            if(pushDown)
            {
                root = nodeIndex;
            }
        }

        uint leafIndex = n.leafIndex[nodeIndex];
        uint start = n.contentStart[leafIndex];
        uint prims = n.contentCount[leafIndex];
        Real depth = rayMax;

        for(uint i = start; i < prims + start; ++i)
        {
            Real2 bary;
            Real d;
            uint triId = n.content[i];
            byte bf = 0;
            if(hitsTriangle(eye, tmin, rayMax, ray, getGeometry().positions + 3 * triId, &d, bary, cf, &bf))
            {
                if(d < depth)
                {
                    t = d;
                    hit.bary = bary;
                    hit.triIndex = triId;
                    hit.isBackFace = bf;
                    depth = d;
                    hit.isHit = 1;
                }
            }
        }
        
        if(hit.isHit && (t < tmax))
        {
            return;
        }
    }
}

struct RayPair
{
    Ray* rays;
    uint* mask;
};

struct RayRange
{
    size_t begin;
    size_t end;

    RayRange (void)
    {

    }

    RayRange(const RayRange& cpy)
    {
        begin = cpy.begin;
        end = cpy.end;
    }

    size_t Length()
    {
        return end - begin;
    }
};

__device__ Real3 transform3f(float3* m3x3l, const Real3* vector)
{
    return make_real3(dot(m3x3l[0], *vector), dot(m3x3l[1], *vector), dot(m3x3l[2], *vector));
}

__device__ Real3 transform4f(float4* m3x3l, const Real4* vector)
{
    return make_real3(dot(m3x3l[0], *vector), dot(m3x3l[1], *vector), dot(m3x3l[2], *vector));
}

__device__ void addRay(RayPair& pair, uint id, Ray& r)
{
    if(r.rayWeight > RAY_WEIGHT_THRESHOLD)
    {
        pair.mask[id] = 1;
        pair.rays[id] = r;
    }
}

__global__ void _traceShadowRays(float4* color, Ray* rays, unsigned int width, unsigned int height, unsigned int N)
{
    uint rayIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if(rayIndex >= N)
    {
        return;
    }

    Ray r = rays[rayIndex];
    uint id = r.screenCoord.y * width + r.screenCoord.x;
    float4 c = color[id];

    Real3 d = getSunPos() - r.getOrigin();
    r.setDir(normalize(d));
    r.setMax(length(d));
    
    TraceResult hitRes;
    traverse(g_tree, r.getOrigin(), r.getDir(), hitRes, *r.getMin(), *r.getMax(), no_cull);

    if(hitRes.isHit)
    {
        Material mat = g_geometry.getMaterial(hitRes.triIndex);
        c *= (1 - r.rayWeight * mat.alpha());
        color[id] = c;
    }
}

__global__ void _traceRefractionRays(float4* color, Ray* refrRays, RayPair rays, unsigned int width, unsigned int height, unsigned int N)
{
    uint rayIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if(rayIndex >= N)
    {
        return;
    }

     Ray r = refrRays[rayIndex];
     //r.clampToBBox();
     addRay(rays, rayIndex, r);

//     r.clampToBBox();
//     float3 os = r.getOrigin() + 0.1 * r.getDir();
//     r.origin_min.x = os.x;
//     r.origin_min.y = os.y;
//     r.origin_min.z = os.z;
//     addRay(rays, rayIndex, r);
//     color[r.screenCoord.y * width + r.screenCoord.x] = make_float4(1,0,0,0);

//     TraceResult hitRes;
//     traverse(g_tree, r.getOrigin(), r.getDir(), hitRes, *r.getMin(), *r.getMax(), cull_front);
// 
//     if(hitRes.isHit)
//     {
//         Real3 hitPos = g_geometry.getTrianglelHitPos(hitRes.triIndex, hitRes.bary);
//         Real3 normal = g_geometry.getTrianglelNormal(hitRes.triIndex, hitRes.bary);
//         
//         Material mat = g_geometry.getMaterial(hitRes.triIndex);
//         //if(
//         r.setOrigin(hitPos + RAY_HIT_NORMAL_DELTA * normal);
//         r.setDir(normalize(refract(r.getDir(), -normal, mat.reflectionIndex(), AIR_RI)));
//         r.clampToBBox();
//         addRay(rays, rayIndex, r);
// 
//         shade(hitRes, rayIndex, color, r, mat);
//     }
}

template <int WRITE_OUT>
__global__ void _traceRays(
    float4* color,
    Ray* inputRays,
    RayPair newReflectionRays,
    RayPair newRefractionRays,
    RayPair newShadowRays,
    unsigned int width,
    unsigned int height,
    unsigned int N)
{
    uint rayIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if(rayIndex >= N)
    {
        return;
    }

    Ray r = inputRays[rayIndex];

    uint id = r.screenCoord.y * width + r.screenCoord.x;

    TraceResult hitRes;
    traverse(g_tree, r.getOrigin(), r.getDir(), hitRes, *r.getMin(), *r.getMax(), cull_back);

    if(hitRes.isHit)
    {
        Material mat = g_geometry.getMaterial(hitRes.triIndex);
        shade(hitRes, id, color, r, mat);
        //color[id].x = hitRes.isBackFace;
        Real3 hitPos = g_geometry.getTrianglelHitPos(hitRes.triIndex, hitRes.bary);
        Real3 normal = g_geometry.getTrianglelNormal(hitRes.triIndex, hitRes.bary);

        if(WRITE_OUT)
        {
            bool isTrans = mat.isTransp();
            bool isMirror = mat.isMirror();

            Real3 dir = r.getDir();
            Real weight = r.rayWeight;
                
            if(isTrans)
            {
                Real3 refraction;
                Real ratio;
                if(hitRes.isBackFace)
                {
                    refraction = refract(r.getDir(), -normal, mat.reflectionIndex(), AIR_RI);
                    ratio = Reflectance(r.getDir(), -normal, mat.reflectionIndex(), AIR_RI, mat.fresnel_t());
                }
                else
                {
                    refraction = refract(r.getDir(), normal, AIR_RI, mat.reflectionIndex());
                    ratio = Reflectance(r.getDir(), normal, AIR_RI, mat.reflectionIndex(), mat.fresnel_t());
                }

                if(abs(dot(refraction, refraction)) > 0)
                {
                    r.rayWeight = weight * ratio * (1 - mat.alpha()) * mat.reflectivity();
                    r.setDir(refraction);
                    r.setOrigin(hitPos + (hitRes.isBackFace ? +RAY_HIT_NORMAL_DELTA * normal : -RAY_HIT_NORMAL_DELTA * normal));
                    r.clampToBBox();
                    addRay(newRefractionRays, rayIndex, r);
                }
            }

            if(isMirror)
            {
                Real ratio = Reflectance(r.getDir(), normal, AIR_RI, mat.reflectionIndex(), mat.fresnel_r());
                Real3 reflection = reflect(r.getDir(), normal);
                r.rayWeight = ratio * weight * mat.reflectivity();
                r.setDir(reflection);
                r.setOrigin(hitPos + RAY_HIT_NORMAL_DELTA * normal);
                r.clampToBBox();
                addRay(newReflectionRays, rayIndex, r);
            }
        }

#if defined COMPUTE_SHADOW
        //spawn shadow ray forall lights
        if(dot(normal, normalize(SUN_POS - hitPos)) > 0)
        {
            r.setOrigin(hitPos + RAY_HIT_NORMAL_DELTA * normal);
            r.rayWeight = 0.5f;
            addRay(newShadowRays, rayIndex, r);
        }
#endif
    }
}

__global__ void computeInitialRays(float4* color, Ray* rays, uint* rayMask, float3 eye, unsigned int width, unsigned int height, unsigned int N)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;

    if(idx >= width || idy >= height)
    {
        return;
    }
    
    Real u = (Real)idx / (Real)width;
    Real v = (Real)idy / (Real)height;

    Real aspect = (Real)width / (Real)height;

    Real3 ray = normalize(make_real3(2 * u - 1, (2 * v - 1) / aspect, 1.0));

    ray = transform3f(g_view, &ray);

    color[id] = make_float4(0,0,0,0); //clear buffer

    Real min = 0, max = FLT_MAX;
    
    if(!intersectP(eye, ray, BBOX_MIN, BBOX_MAX, &min, &max))
    {
        rayMask[id] = 0;
        return;
    } 

    rayMask[id] = 1;

    Ray r;

    r.screenCoord.x = idx;
    r.screenCoord.y = idy;
    
    r.dir_max.x = ray.x;
    r.dir_max.y = ray.y;
    r.dir_max.z = ray.z;

    r.origin_min.x = eye.x;
    r.origin_min.y = eye.y;
    r.origin_min.z = eye.z;

    r.setMin(min);
    r.setMax(max);

    r.rayWeight = 1;

    rays[id] = r;
}

__constant__ Real4 g_matrix[4]; 
__global__ void transform(Normal* n, Normal* newNormals, uint N)
{
    uint id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= N)
    {
        return;
    }
    Normal _n = n[id];
    Real4 v = make_real4(_n.x, _n.y, _n.z, 0);
    newNormals[id] = transform4f(g_matrix, &v);
}

nutty::DeviceBuffer<Ray>* g_rays[2];
nutty::DeviceBuffer<uint>* g_rayMask;

nutty::DeviceBuffer<Ray>* g_refractionRays[2];
nutty::DeviceBuffer<uint>* g_refractionRayMask;

nutty::DeviceBuffer<Ray>* g_shadowRays[2];
nutty::DeviceBuffer<uint>* g_shadowRayMask;

nutty::DeviceBuffer<uint>* g_scannedRayMask;
nutty::DeviceBuffer<uint>* g_scannedSums;
nutty::DeviceBuffer<uint>* g_sums;

dim3 g_grid;
dim3 g_grp;
int g_recDepth = RECURSION;

uint g_width = 1;
uint g_height = 1;

extern "C" void RT_SetViewPort(unsigned int width, unsigned int height)
{
    g_width = width;

    g_height = height;

    g_grp.x = 32;
    g_grp.y = 32;
    g_grp.z = 1;
    
    g_grid.x = nutty::cuda::GetCudaGrid(width, g_grp.x);
    g_grid.y = nutty::cuda::GetCudaGrid(height, g_grp.y);
    g_grid.z = 1;
}

extern "C" void RT_TransformNormals(Normal* normals, Normal* newNormals, Real4* matrix, size_t start, uint N)
{
    dim3 grid; dim3 group(256);

    grid.x = nutty::cuda::GetCudaGrid(N, group.x);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_matrix, matrix, 4 * sizeof(Real4), 0, cudaMemcpyHostToDevice));
    transform<<<grid, group>>>(normals + start, newNormals + start, N);
}

extern "C" void RT_Init(unsigned int width, unsigned int height)
{
    g_shadowRays[0] = new nutty::DeviceBuffer<Ray>();
    g_shadowRays[1] = new nutty::DeviceBuffer<Ray>();

    g_rays[0] = new nutty::DeviceBuffer<Ray>();
    g_rays[1] = new nutty::DeviceBuffer<Ray>();

    g_refractionRays[0] = new nutty::DeviceBuffer<Ray>();
    g_refractionRays[1] = new nutty::DeviceBuffer<Ray>();

    g_rayMask = new nutty::DeviceBuffer<uint>();
    g_refractionRayMask = new nutty::DeviceBuffer<uint>();

    g_scannedRayMask = new nutty::DeviceBuffer<uint>();
    g_shadowRayMask = new nutty::DeviceBuffer<uint>();
    g_sums = new nutty::DeviceBuffer<uint>();
    g_scannedSums = new nutty::DeviceBuffer<uint>();

    unsigned int maxRaysLastBranch = width * height * (1 << (MAX_RECURSION - 1));
    unsigned int maxRaysPerNode = width * height;

    g_rays[0]->Resize(maxRaysLastBranch);
    g_rays[1]->Resize(maxRaysLastBranch);
    g_rayMask->Resize(maxRaysLastBranch);
    g_scannedRayMask->Resize(maxRaysPerNode);

    g_refractionRays[0]->Resize(maxRaysPerNode);
    g_refractionRays[1]->Resize(maxRaysPerNode);
    g_refractionRayMask->Resize(maxRaysPerNode);

    g_shadowRays[0]->Resize(maxRaysPerNode);
    g_shadowRays[1]->Resize(maxRaysPerNode);
    g_shadowRayMask->Resize(maxRaysPerNode);

    g_sums->Resize((2 * maxRaysPerNode) / 512);
    g_scannedSums->Resize((2 * maxRaysPerNode) / 512);

    RT_SetLightPos(make_float3(0,10,-10));
}

extern "C" void RT_Destroy(void)
{
    delete g_rays[0];
    delete g_rays[1];

    delete g_refractionRays[0];
    delete g_refractionRays[1];

    delete g_shadowRays[0];
    delete g_shadowRays[1];

    delete g_rayMask;
    delete g_refractionRayMask;

    delete g_scannedRayMask;
    delete g_sums;
    delete g_scannedSums;
    delete g_shadowRayMask;
}

uint g_lastRays = 0;
std::stringstream g_info;

extern "C" void RT_GetRayInfo(std::string& info)
{
    info.clear();
    info += "\n\n";
    info += g_info.str();
}

extern "C" uint RT_GetLastRayCount(void)
{
    return g_lastRays;
}

uint compactRays(nutty::DeviceBuffer<uint>::iterator& maskBegin,
                 nutty::DeviceBuffer<Ray>::iterator& rayDstBegin, 
                 nutty::DeviceBuffer<Ray>::iterator& raySrcBegin, 
                 uint rayCount)
{
    nutty::ZeroMem(*g_scannedRayMask);
    nutty::ZeroMem(*g_sums);

    nutty::ExclusivePrefixSumScan(maskBegin, maskBegin + rayCount, g_scannedRayMask->Begin(), g_sums->Begin(), g_scannedSums->Begin());

    nutty::Compact(rayDstBegin, raySrcBegin, raySrcBegin + rayCount, maskBegin, g_scannedRayMask->Begin(), 0U);
    auto it = g_scannedRayMask->Begin() + rayCount - 1;
    return *(g_scannedRayMask->Begin() + rayCount - 1) + *(maskBegin + rayCount - 1);
}

void traceRays(float4* colors, 
               int recDepth, 
               uint lastRayCount,
               uint width, uint height)
{
    nutty::ZeroMem(*g_scannedRayMask);
    nutty::ZeroMem(*g_sums);

    byte toggle = 0;
    std::queue<RayRange> q[2];

    RayRange ir;
    ir.begin = 0;
    ir.end = lastRayCount;
    q[0].push(ir);
    g_info.str("");
    for(int i = 0; i < recDepth; ++i)
    {
        uint src = toggle % 2;
        uint dst = (toggle+1) % 2;
        toggle ^= 1;

        //print("\nDepth=%d\n", i);
        g_info << "\nDepth=" << i << "\n";
        uint offset = (1 << (max(0, (recDepth - i - 2)))) * width * height;

        while(!q[src].empty())
        {
            RayRange range = q[src].front();
            q[src].pop();

            lastRayCount = compactRays(g_rayMask->Begin() + range.begin, g_rays[dst]->Begin() + range.begin, g_rays[src]->Begin() + range.begin, range.Length());
            
            g_lastRays += lastRayCount;
            //print("Range: from '%d' -> '%d' (L=%d) got '%d' Rays\n", range.begin, range.end, range.Length(), lastRayCount);
            g_info << "Range: '" << range.begin << "' -> '"<< range.end << " got '" << lastRayCount << "' Rays\n";
            if(lastRayCount > 0)
            {
                uint blockSize = 256;
                dim3 g = nutty::cuda::GetCudaGrid(lastRayCount, blockSize);

                RayPair newShadowRays;
                newShadowRays.mask = g_shadowRayMask->Begin()();
                newShadowRays.rays = g_shadowRays[0]->Begin()();

                RayPair newRefractionRaysRays;
                newRefractionRaysRays.mask = g_refractionRayMask->Begin()();
                newRefractionRaysRays.rays = g_refractionRays[0]->Begin()();

                RayPair rayPairDst;
                rayPairDst.mask = g_rayMask->Begin()() + range.begin;
                rayPairDst.rays = g_rays[dst]->Begin()() + range.begin;

                nutty::ZeroMem<uint>(g_rayMask->Begin() + range.begin, g_rayMask->Begin() + range.end);
                nutty::ZeroMem(*g_refractionRayMask);
                nutty::ZeroMem(*g_shadowRayMask);

                if(i+1 == recDepth)
                {
                    _traceRays<0><<<g, blockSize>>>(colors, rayPairDst.rays, rayPairDst, newRefractionRaysRays, newShadowRays, width, height, lastRayCount);
                }
                else
                {
                    _traceRays<1><<<g, blockSize>>>(colors, rayPairDst.rays, rayPairDst, newRefractionRaysRays, newShadowRays, width, height, lastRayCount);

                    RayRange refRange;
                    refRange.begin = range.begin;
                    refRange.end = range.begin + lastRayCount;
                    q[dst].push(refRange);
                }

                //print("%d\n", cudaDeviceSynchronize());

                uint shadowRaysCount = compactRays(g_shadowRayMask->Begin(), g_shadowRays[1]->Begin(), g_shadowRays[0]->Begin(), lastRayCount);

                g_lastRays += shadowRaysCount;

                if(shadowRaysCount > 0)
                {
                    g = nutty::cuda::GetCudaGrid(shadowRaysCount, blockSize);
                    _traceShadowRays<<<g, blockSize>>>(colors, g_shadowRays[1]->Begin()(), width, height, shadowRaysCount);
                }

#if defined COMPUTE_REFRACTION
                if(i+1 < recDepth)
                {
                    uint refractionRaysCount = compactRays(g_refractionRayMask->Begin(), g_refractionRays[1]->Begin(), g_refractionRays[0]->Begin(), lastRayCount);
                    g_lastRays += refractionRaysCount;

                    if(refractionRaysCount > 0)
                    {
                        g = nutty::cuda::GetCudaGrid(refractionRaysCount, blockSize);
                        RayPair rayPair;
                        rayPair.mask = offset + g_rayMask->Begin()() + range.begin;
                        rayPair.rays = offset + g_rays[dst]->Begin()() + range.begin;
                        _traceRefractionRays<<<g, blockSize>>>(colors, g_refractionRays[1]->Begin()(), rayPair, width, height, refractionRaysCount);

                        RayRange refRange;
                        refRange.begin = offset + range.begin;
                        refRange.end = offset + range.begin + refractionRaysCount;
                        q[dst].push(refRange);
                    }
                }
#endif
            }
        }
    }
}

extern "C" void RT_Trace(float4* colors, const float3* view, float3 eye, BBox& bbox)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_view, view, 3 * sizeof(float3), 0, cudaMemcpyHostToDevice));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_bbox, &bbox, sizeof(BBox), 0, cudaMemcpyHostToDevice));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_bbox, &bbox, sizeof(BBox), 0, cudaMemcpyHostToDevice));

    nutty::ZeroMem(*g_rayMask);

    computeInitialRays<<<g_grid, g_grp>>>(colors, g_rays[0]->Begin()(), g_rayMask->Begin()(), eye, g_width, g_height, g_width * g_height);

    g_lastRays = 0;

    traceRays(colors, g_recDepth, g_width * g_height, g_width, g_height);
}
 
extern "C" void RT_BindTree(TreeNodes& tree)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_tree, &tree, sizeof(TreeNodes), 0, cudaMemcpyHostToDevice));
}

extern "C" void RT_SetLightPos(const float3& pos)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_sunPos, &pos, sizeof(float3), 0, cudaMemcpyHostToDevice));
}

extern "C" void RT_BindGeometry(Triangles& tries)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_geometry, &tries, sizeof(Triangles), 0, cudaMemcpyHostToDevice));
}

extern "C" int RT_GetRecDepth(void)
{
    return g_recDepth;
}

extern "C" void RT_SetRecDepth(int d)
{
    g_recDepth =  d < 1 ? 1 : (d > MAX_RECURSION ? MAX_RECURSION : d);
}

extern "C" void RT_IncDepth(void)
{
    RT_SetRecDepth(g_recDepth+1);
}

extern "C" void RT_DecDepth(void)
{
    RT_SetRecDepth(g_recDepth-1);
}

extern "C" void RT_BindTextureAtlas(const cudaArray_t array)
{
//     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint2>();
//     g_atlas.normalized = 1;
//     g_atlas.filterMode = cudaFilterModePoint;
//     g_atlas.addressMode[0] = cudaAddressModeClamp;
//     g_atlas.addressMode[1] = cudaAddressModeClamp;
//     g_atlas.addressMode[2] = cudaAddressModeClamp;
// 
//     CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_atlas, array, channelDesc));
}

void __device__ Ray::clampToBBox(void)
{
    setMin(0);
    setMax(1000.0);
    Real3 o = {origin_min.x, origin_min.y, origin_min.z};
    Real3 d = {dir_max.x, dir_max.y, dir_max.z};

    intersectP(o, d, BBOX_MIN, BBOX_MAX, getMin(), getMax());
}
