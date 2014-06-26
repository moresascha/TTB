#include <Nutty.h>
#include <DeviceBuffer.h>
#include <cuda/cuda_helper.h>
#include "ct.h"
#include "geometry.h"
#include <cuda/Globals.cuh>

template<>
struct ShrdMemory<_AABB>
{
    __device__ _AABB* Ptr(void) 
    { 
        extern __device__ __shared__ _AABB s_AABB[];
        return s_AABB;
    }
};

#include <Reduce.h>

__device__ __forceinline CTreal3 min3(const CTreal3& a, const CTreal3& b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline CTreal3 max3(const CTreal3& a, const CTreal3& b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__global__ void _createAABB(CTreal3* tris, _AABB* aabbs, CTuint N)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= N)
    {
        return;
    }

    CTreal3 A = tris[3 * idx + 0];
    CTreal3 B = tris[3 * idx + 1];
    CTreal3 C = tris[3 * idx + 2];

    _AABB bb;
    bb.Reset();

    bb.AddVertex(A);
    bb.AddVertex(B);
    bb.AddVertex(C);

    aabbs[idx] = bb;
}

extern "C" void cudaCreateTriangleAABBs(CTreal3* tris, _AABB* aabbs, CTuint N)
{
    dim3 grps(256);
    dim3 grid;

    grid.x = nutty::cuda::GetCudaGrid(N, grps.x);

    _createAABB<<<grid, grps>>>(tris, aabbs, N);
}

struct AABB_OP
{
    __device__ __host__ AABB_OP(void)
    {

    }

    __device__ _AABB operator()(_AABB& t0, _AABB& t1)
    {
        _AABB res;
        res.m_min = min3(t0.m_min, t1.m_min);
        res.m_max = max3(t0.m_max, t1.m_max);
        return res;
    }
};

extern "C" void cudaGetSceneBBox(nutty::DeviceBuffer<_AABB>& aabbs, CTuint N, _AABB& aabb)
{
    nutty::Reduce(aabbs.Begin(), aabbs.End(), AABB_OP(), _AABB());
    aabb = aabbs[0];
}