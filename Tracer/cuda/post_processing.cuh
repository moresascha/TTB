#pragma once

#include "globals.cuh"
#include <cutil_inline.h>

__global__ void postProcess(float4* colors, uint N)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;

    if(id >= N) return;

    colors[id] = make_float4(1,1,1,1) - colors[id];
}

template<int factor>
__global__ void scaleDown(float4* dst, cudaSurfaceObject_t src, uint width, uint height)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;
    if(idx >= width || idy >= height)
    {
        return;
    }

    uint tidx = idx * factor;
    uint tidy = idy * factor;
    
    float4 color = make_float4(0,0,0,0);
    for(int i = 0; i < factor; ++i)
    {
        for(int j = 0; j < factor; ++j)
        {
            float4 cc;
            surf2Dread(&cc, src, sizeof(float4) * (tidx + i), tidy + j);
            color += cc;
        }
    }
    color /= (factor * factor);
    dst[id] = color;
    //surf2Dwrite(color, dst, sizeof(float4)*idx, idy);
}

__global__ void getHDValues(float4* colors, float4* hd, cudaSurfaceObject_t surface, float* logLuminance, uint N)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;

    if(id >= N) return;

    float4 color = colors[id];

    float L = 0.0000001 + 0.27 * color.x + 0.67 * color.y + 0.06 * color.z;

    logLuminance[id] = log(L);

    if(!(color.x > 1 || color.y > 1 || color.z > 1))
    {
        color = make_float4(0,0,0,0);
    }

    //hd[id] = color;

    surf2Dwrite(color, surface, sizeof(float4)*idx, idy);
}

__global__ void addHDValus(float4* finalColor, cudaTextureObject_t hdBlur, float4* rawColors, uint N, uint blurLine, uint width, float* logAverageLuminace)
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint id = idx + idy * blockDim.x * gridDim.x;

    if(id >= N) return;

    uint bidx = (blockDim.x * blockIdx.x + threadIdx.x) / blurLine;
    uint bidy = (blockDim.y * blockIdx.y + threadIdx.y) / blurLine;

    float4 color = rawColors[id];
    
    float lumi = 0.27*color.x + 0.67*color.y + 0.06*color.z;
    float Lw = exp(1.0f / logAverageLuminace[0]) / (float)N;
    
    float key = 0.1;
    lumi = key * lumi / Lw;

    float lumiScaled = lumi / (1 + lumi);

    float4 blur = tex2D<float4>(hdBlur, idx / (float)width, idy / (float)width);

    color += blur;
    color = color * lumiScaled;

    finalColor[id] = color;
}

#define KERNEL_RADIUS 3
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_Kernel[KERNEL_LENGTH];

static __inline void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 8
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 4
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(
    float4 *d_Dst,
    float4 *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float4 s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : make_float4(0,0,0,0);
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : make_float4(0,0,0,0);
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float4 sum = make_float4(0,0,0,0);

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        //int linearAdd = baseY * pitch + baseX;
        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
//         int y = (linearAdd + i * ROWS_BLOCKDIM_X) / imageW;
//         int x = (linearAdd + i * ROWS_BLOCKDIM_X) % imageW;
//         surf2Dwrite(sum, surface, sizeof(float4) * x, y);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 8
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(
    float4 *d_Dst,
    float4 *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float4 s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : make_float4(0,0,0,0);
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : make_float4(0,0,0,0);
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float4 sum = make_float4(0,0,0,0);
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}