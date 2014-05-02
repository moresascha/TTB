#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#pragma comment(lib, "cuda.lib")

__device__  float2 f0(float2 v)
{
    v.x = v.x / 2.0;
    v.y = v.y / 2.0;
    return v;
}

__device__  float2 f1(float2 v)
{
    v.x = (v.x+1) / 2.0;
    v.y = v.y/2.0;
    return v;
}

__device__ float2 f2(float2 v)
{
    v.x = v.x / 2.0;
    v.y = (v.y+1) / 2.0;
    return v;
}

__device__ float2 linear(float2 p, float r, float theta)
{
    return p;
}
__device__ float2 spherical(float2 p, float r, float theta)
{
    float d = p.x*p.x + p.y*p.y;
    return make_float2(p.x/d, p.y/d);
}

__device__ float2 swirl(float2 p, float r, float theta)
{
    return make_float2(r*cos(theta+r), r*sin(theta+r));
}

typedef float2 (*func[])(float2);

typedef float2 (*func2[])(float2 p, float r, float theta);

__global__ void computeIFS(float4* data, float time, int* rands, int randsN, unsigned int width, unsigned int height, unsigned int N)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(N <= idx)
    {
        return;
    }

    float4 points = data[idx];

    float rc = (rands[(idx + (int)(time * 1000)) % randsN] % 1000) / 100000.0;

    int rand = rands[(idx + (int)(time * 1000)) % randsN] % 3;

    //func f = {f0, f1, f2};
    func2 f = {linear, spherical, swirl};

    float2 tp = make_float2(points.x, points.y);
    
    float a = 1.567;
    float2 p = tp;

    p.x = cos(a) * tp.x - sin(a) * tp.y;
    p.y = sin(a) * tp.x + cos(a) * tp.y;

    float r = length(p);

    float theta = atan2(p.x, p.y);

    p = f[rand](p, r, theta);

//     float c = time;
//     c = c - int(time);

    p.x = clamp(p.x, -1.0, 1.0);
    p.y = clamp(p.y, -1.0, 1.0);

    if(points.z > 1)
    {
        points.z = 0;
    }

    data[idx] = make_float4(p.x, p.y, points.z + rc, 0);
}

__global__ void fillGrid(float4* data, float4* color, unsigned int width, unsigned int height, cudaTextureObject_t gradientTexture, unsigned int N)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(N <= idx)
    {
        return;
    }

    float4 p = data[idx];
    float c = p.z;
    p = 0.5 + 0.5 * p;

    float x = clamp(p.x, 0.0, 1.0);
    float y = clamp(p.y, 0.0, 1.0);

    int ix = max(0, (int)(width * x) - 1);
    int iy = max(0, (int)(height * y) - 1);

    color[width * iy + ix] = tex1D<float4>(gradientTexture, c);
}

__global__ void init(float4* data, float4* colors, unsigned int width, unsigned int height, unsigned int N)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(N <= idx)
    {
        return;
    }

    float x = -1 + 2 * (idx % width) / (float)width;
    float y = -1 + 2 * idx / (float)(height * width);

    colors[idx] = make_float4(0, 0, 0, 0);
    data[idx] = make_float4(x, y, 0, 0);
}

extern "C" void initData(float4* data, float4* colors, unsigned int width, unsigned int height)
{
    unsigned int groupSize = 256;
    unsigned int grid;
    unsigned int N = width * height;

    if(N % groupSize == 0)
    {
        grid = N / groupSize;
    }
    else
    {
        grid = (N / groupSize + 1);
    }

    init<<<grid, 256>>>(data, colors, width, height, N);
}

extern "C" void generateColors(float4* data, float4* color, cudaTextureObject_t gradientTexture, float dt, int* rands, int randN, unsigned int width, unsigned int height)
{
    unsigned int groupSize = 256;
    unsigned int grid;
    unsigned int N = width * height;

    if(N % groupSize == 0)
    {
        grid = N / groupSize;
    }
    else
    {
        grid = (N / groupSize + 1);
    }

    computeIFS<<<grid, 256>>>(data, dt, rands, randN, width, height, N);
    fillGrid<<<grid, 256>>>(data, color, width, height, gradientTexture, N);
}