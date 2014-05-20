#include "texturing.cuh"
#include "globals.cuh"
#include <Nutty.h>
#include "../texture_array.h"

__constant__ cudaTextureObject_t g_textures[8];

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex0;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex1;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex2;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex3;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex4;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex5;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex6;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex7;

extern "C" __device__ float4 readTexture(uint slot, const Real2& tc) 
{
#ifdef KEPLER
    cudaTextureObject_t tex = g_textures[slot];
    return tex2D<float4>(tex, tc.x, tc.y);
#else
    if(slot == 0) return tex2D(g_tex0, tc.x, tc.y);
    if(slot == 1) return tex2D(g_tex1, tc.x, tc.y);
    if(slot == 2) return tex2D(g_tex2, tc.x, tc.y);
    if(slot == 3) return tex2D(g_tex3, tc.x, tc.y);
    if(slot == 4) return tex2D(g_tex4, tc.x, tc.y);
    if(slot == 5) return tex2D(g_tex5, tc.x, tc.y);
    if(slot == 6) return tex2D(g_tex6, tc.x, tc.y);
    if(slot == 7) return tex2D(g_tex7, tc.x, tc.y);
#endif
    return make_float4(1,0,0,0);
}

extern "C" void RT_BindTextures(const cuTextureObj* textures, uint size)
{
#ifdef KEPLER
    cudaTextureObject_t tex[8];
    memset(tex, 0, sizeof(cudaTextureObject_t) * 8);
    for(int i = 0; i < size; ++i)
    {
        tex[i] = textures[i].tex;
    }
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_textures, tex, size * sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
#else

     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

     for(int i = 0; i < size; ++i)
     {
         switch(i)
         {
         case 0:
             {
                 g_tex0.normalized = true;
                 g_tex0.filterMode = cudaFilterModeLinear;
                 g_tex0.addressMode[0] = cudaAddressModeWrap;
                 g_tex0.addressMode[1] = cudaAddressModeWrap;
                 g_tex0.addressMode[2] = cudaAddressModeWrap;
                 CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex0, textures[0].array, channelDesc));
             } break;
         case 1:
             {
                 g_tex1.normalized = true;
                 g_tex1.filterMode = cudaFilterModeLinear;
                 g_tex1.addressMode[0] = cudaAddressModeWrap;
                 g_tex1.addressMode[1] = cudaAddressModeWrap;
                 g_tex1.addressMode[2] = cudaAddressModeWrap;
                 CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex1, textures[1].array, channelDesc));
             } break;
         case 2:
             {
                 g_tex2.normalized = true;
                 g_tex2.filterMode = cudaFilterModeLinear;
                 g_tex2.addressMode[0] = cudaAddressModeWrap;
                 g_tex2.addressMode[1] = cudaAddressModeWrap;
                 g_tex2.addressMode[2] = cudaAddressModeWrap;
                 CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex2, textures[2].array, channelDesc));
             } break;
         case 3:
             {
                 g_tex3.normalized = true;
                 g_tex3.filterMode = cudaFilterModeLinear;
                 g_tex3.addressMode[0] = cudaAddressModeWrap;
                 g_tex3.addressMode[1] = cudaAddressModeWrap;
                 g_tex3.addressMode[2] = cudaAddressModeWrap;
                 CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex3, textures[3].array, channelDesc));
             } break;
         case 4:
             {
                 g_tex4.normalized = true;
                 g_tex4.filterMode = cudaFilterModeLinear;
                 g_tex4.addressMode[0] = cudaAddressModeWrap;
                 g_tex4.addressMode[1] = cudaAddressModeWrap;
                 g_tex4.addressMode[2] = cudaAddressModeWrap;
                 CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex4, textures[4].array, channelDesc));
             } break;
         case 5:
             {
                 g_tex5.normalized = true;
                 g_tex5.filterMode = cudaFilterModeLinear;
                 g_tex5.addressMode[0] = cudaAddressModeWrap;
                 g_tex5.addressMode[1] = cudaAddressModeWrap;
                 g_tex5.addressMode[2] = cudaAddressModeWrap;
                 CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex5, textures[5].array, channelDesc));
             } break;
         }
     }
#endif
}