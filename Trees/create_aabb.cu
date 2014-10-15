#include <Nutty.h>
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

__global__ void _createAABBNaiv(CTreal3* tris, _AABB* aabbs, CTuint N)
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

__global__ void _createAABB(CTreal* tris, CTreal* aabbs, CTuint N)
{
    const uint blockSize = 256;

    __shared__ CTreal s_data[9 * blockSize];

    uint offset = blockIdx.x * blockDim.x * 9;
#pragma unroll
    for(int i = 0; i < 9; ++i)
    {
       s_data[threadIdx.x + i * blockSize] = offset + threadIdx.x + i * blockSize < 9 * N ? 
		   tris[offset + threadIdx.x + i * blockSize] : 0;
    }

    __syncthreads();

//     CTreal3 A = ((CTreal3*)s_data)[3 * threadIdx.x + 0];
//     CTreal3 B = ((CTreal3*)s_data)[3 * threadIdx.x + 1];
//     CTreal3 C = ((CTreal3*)s_data)[3 * threadIdx.x + 2];

    CTreal3 A;
    CTreal3 B;
    CTreal3 C;

    A.x = s_data[9 * threadIdx.x + 0];
    A.y = s_data[9 * threadIdx.x + 1];
    A.z = s_data[9 * threadIdx.x + 2];

    B.x = s_data[9 * threadIdx.x + 3];
    B.y = s_data[9 * threadIdx.x + 4];
    B.z = s_data[9 * threadIdx.x + 5];

    C.x = s_data[9 * threadIdx.x + 6];
    C.y = s_data[9 * threadIdx.x + 7];
    C.z = s_data[9 * threadIdx.x + 8];

    _AABB bb;
    bb.Reset();

    bb.AddVertex(A);
    bb.AddVertex(B);
    bb.AddVertex(C);

    __syncthreads();

    ((_AABB*)s_data)[threadIdx.x] = bb;

    __syncthreads();

#pragma unroll
    for(int i = 0; i < 6; ++i)
    {
        uint add = blockIdx.x * blockDim.x * 6 + threadIdx.x + i * blockSize;
        if(add < 6 * N)
        {
            aabbs[add] =  s_data[threadIdx.x + i * blockSize];
        }
    }
}

__global__ void test(int* timings)
{
    __shared__ float s_data[3 * 32];
    unsigned long long start = clock();
    ((float3*)s_data)[threadIdx.x] = make_float3(1,1,1);
    unsigned long long end = clock();
    *timings = (int)(end - start);
}

extern "C" void cudaCreateTriangleAABBs(CTreal3* tris, _AABB* aabbs, CTuint N, cudaStream_t pStream)
{
    const CTuint grps = 256;
    CTuint grid;

    grid = nutty::cuda::GetCudaGrid(N, grps);

    _createAABB<<<grid, grps, 0, pStream>>>((CTreal*)tris, (CTreal*)aabbs, N);
    //_createAABBNaiv<<<grid, grps, 0, pStream>>>(tris, aabbs, N);
//     int htiming;
//     int* dtiming = 0;
//     cudaMalloc(&dtiming, sizeof(int));
//     test<<<1, 32>>>(dtiming);
//     cudaError_t err = cudaDeviceSynchronize();
//     err = cudaMemcpy(&htiming, dtiming, sizeof(int), cudaMemcpyDeviceToHost);
//     __ct_printf("%d %d\n", (htiming - 14)/32, err);
//     cudaDeviceReset();
//     exit(0);
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
    //throw "Error";
    //nutty::Reduce(aabbs.Begin(), aabbs.End(), AABB_OP(), _AABB());
    //aabb = aabbs[0];
    OutputDebugStringA("cudaGetSceneBBox fails...\n"); exit(-1);
}