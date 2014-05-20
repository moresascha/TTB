#include "texture_array.h"
#include "io.h"
#include <Nutty.h>

cuTextureAtlas::cuTextureAtlas(void) : m_entries(0), m_size(16)
{

}

void cuTextureAtlas::Resize(void)
{
    m_size *= 2;
    //todo
}

void cuTextureAtlas::Init(void)
{
    m_detailTextures = new cuTextureObj[m_size];
}

int cuTextureAtlas::AddTexture(uchar4* data, unsigned int width, unsigned int height)
{
    int slot = m_entries;
    m_entries++;

    int elementSize = sizeof(uchar4);

    cuTextureObj t;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMallocArray(&t.array, &desc, width, height));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToArray(t.array, 0, 0, data, elementSize * (width * height), cudaMemcpyHostToDevice));

#ifdef KEPLER
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = t.array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[3] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = 1;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaCreateTextureObject(&t.tex, &resDesc, &texDesc, NULL));
#endif
    m_detailTextures[slot] = t;

    return slot;
}

int cuTextureAtlas::AddTexture(const char* file)
{
    TextureData texData = GetTextureJPGData(file);

    int slot = AddTexture(texData.data, texData.width, texData.height);

    delete[] texData.data;

    return slot;
}

const cuTextureObj* cuTextureAtlas::GetTextures(uint* count)
{
    *count = m_entries;
    return m_detailTextures;
}

cuTextureAtlas::~cuTextureAtlas(void)
{
    for(int i = 0; i < m_entries; ++i)
    {
        if(m_detailTextures[i].array)
        {
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaFreeArray(m_detailTextures[i].array));
            if(m_detailTextures[i].tex)
            {
                CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDestroyTextureObject(m_detailTextures[i].tex));
            }
        }
    }
    delete[] m_detailTextures;
}