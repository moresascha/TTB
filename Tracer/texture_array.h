#pragma once
#include <cutil_inline.h>
#include "cuda/globals.cuh"
struct cuTextureObj
{
    cudaArray_t array;
    cudaTextureObject_t tex;

    cuTextureObj(void) : array(NULL), tex(NULL)
    {

    }
};

class cuTextureAtlas
{
private:
    cuTextureObj* m_detailTextures;
    int m_entries;
    int m_size;

    void Resize(void);

public:
    cuTextureAtlas(void);

    void Init(void);

    int AddTexture(const char* file);

    int AddTexture(uchar4* data, unsigned int width, unsigned int height);

    const cuTextureObj* GetTextures(unsigned int* count);

    ~cuTextureAtlas(void);
};