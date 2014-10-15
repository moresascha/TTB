#include "texturing.cuh"
#include "globals.cuh"
#include <Nutty.h>
#include "../texture_array.h"

__constant__ cudaTextureObject_t g_textures[8];

#define ADD_TEXTURE_SLOT_CASE(slot) \
 case slot: \
{ \
    g_tex##slot##.normalized = true; \
    g_tex##slot##.filterMode = cudaFilterModeLinear; \
    g_tex##slot##.addressMode[0] = cudaAddressModeWrap; \
    g_tex##slot##.addressMode[1] = cudaAddressModeWrap; \
    g_tex##slot##.addressMode[2] = cudaAddressModeWrap; \
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaBindTextureToArray(g_tex##slot##, textures[slot].array, channelDesc)); \
} break;

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex0;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex1;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex2;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex3;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex4;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex5;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex6;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex7;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex8;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex9;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex10;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex11;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex12;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex13;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex14;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex15;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex16;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex17;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex18;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex19;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex20;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex21;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex22;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex23;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> g_tex24;

extern "C" __device__ float4 readTexture(uint slot, const Real2& tc) 
{
#ifdef KEPLER
    cudaTextureObject_t tex = g_textures[0];
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
    if(slot == 8) return tex2D(g_tex8, tc.x, tc.y);
    if(slot == 9) return tex2D(g_tex9, tc.x, tc.y);
    if(slot == 10) return tex2D(g_tex10, tc.x, tc.y);
    if(slot == 11) return tex2D(g_tex11, tc.x, tc.y);
    if(slot == 12) return tex2D(g_tex12, tc.x, tc.y);
    if(slot == 13) return tex2D(g_tex13, tc.x, tc.y);
    if(slot == 14) return tex2D(g_tex14, tc.x, tc.y);
    if(slot == 15) return tex2D(g_tex15, tc.x, tc.y);
    if(slot == 16) return tex2D(g_tex16, tc.x, tc.y);
    if(slot == 17) return tex2D(g_tex17, tc.x, tc.y);
    if(slot == 18) return tex2D(g_tex18, tc.x, tc.y);
    if(slot == 19) return tex2D(g_tex19, tc.x, tc.y);
    if(slot == 20) return tex2D(g_tex20, tc.x, tc.y);
    if(slot == 21) return tex2D(g_tex21, tc.x, tc.y);
    if(slot == 22) return tex2D(g_tex22, tc.x, tc.y);
    if(slot == 23) return tex2D(g_tex23, tc.x, tc.y);
    if(slot == 24) return tex2D(g_tex24, tc.x, tc.y);
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
     std::stringstream ss;
     ss << size;
     OutputDebugStringA("TextureCount=");
     OutputDebugStringA(ss.str().c_str());
     OutputDebugStringA("\n");
     for(int i = 0; i < size; ++i)
     {
         switch(i)
         {
             ADD_TEXTURE_SLOT_CASE(0);
             ADD_TEXTURE_SLOT_CASE(1);
             ADD_TEXTURE_SLOT_CASE(2);
             ADD_TEXTURE_SLOT_CASE(3);
             ADD_TEXTURE_SLOT_CASE(4);
             ADD_TEXTURE_SLOT_CASE(5);
             ADD_TEXTURE_SLOT_CASE(6);
             ADD_TEXTURE_SLOT_CASE(7);
             ADD_TEXTURE_SLOT_CASE(8);
             ADD_TEXTURE_SLOT_CASE(9);
             ADD_TEXTURE_SLOT_CASE(10);
             ADD_TEXTURE_SLOT_CASE(11);
             ADD_TEXTURE_SLOT_CASE(12);
             ADD_TEXTURE_SLOT_CASE(13);
             ADD_TEXTURE_SLOT_CASE(14);
             ADD_TEXTURE_SLOT_CASE(15);
             ADD_TEXTURE_SLOT_CASE(16);
             ADD_TEXTURE_SLOT_CASE(17);
             ADD_TEXTURE_SLOT_CASE(18);
             ADD_TEXTURE_SLOT_CASE(19);
             ADD_TEXTURE_SLOT_CASE(20);
             ADD_TEXTURE_SLOT_CASE(21);
             ADD_TEXTURE_SLOT_CASE(22);
             ADD_TEXTURE_SLOT_CASE(23);
             ADD_TEXTURE_SLOT_CASE(24);
         }
     }
#endif
}