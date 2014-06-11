
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#include "ct.h"
#include <Nutty.h>
#include <DeviceBuffer.h>
#include <cuda/cuda_helper.h>
#include "shared_kernel.h"
#include <cutil_math.h>

__device__ __forceinline CTreal3 transform4f(float4* m3x3l, const CTreal4* vector)
{
    CTreal3 res = make_real3(dot(m3x3l[0], *vector), dot(m3x3l[1], *vector), dot(m3x3l[2], *vector));
    res.x += m3x3l[3].x;
    res.y += m3x3l[3].y;
    res.z += m3x3l[3].z;
    return res;
}

__constant__ CTreal4 g__matrix[4]; 
__global__ void __transform3f(const CTreal3* in, CTreal3* out, uint N)
{
    uint id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= N)
    {
        return;
    }

    CTreal3 _n = in[id];

    CTreal4 v = make_real4(_n.x, _n.y, _n.z, 1);

    out[id] = transform4f(g__matrix, &v);
}

extern "C" void cudaTransformVector(nutty::DeviceBuffer<CTreal3>::iterator& v_in, nutty::DeviceBuffer<CTreal3>::iterator& v_out, const CTreal4* matrix, CTuint N)
{
    dim3 grid; dim3 group(64);

    grid.x = nutty::cuda::GetCudaGrid(N, group.x);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g__matrix, matrix, 4 * sizeof(CTreal4), 0, cudaMemcpyHostToDevice));

    __transform3f<<<grid, group>>>(v_in(), v_out(), N);
}