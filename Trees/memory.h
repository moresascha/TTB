#pragma once
#include <memory>
#include "ct_base.h"

#define RawAlloc(size) malloc(size)
#define RawFree(ptr) free(ptr)

#define CT_SAFE_FREE(obj) { if(obj) { CTMemFreeObject(obj); obj = NULL; } }

class ICTAllocator
{
public:
    virtual void* Alloc(size_t size) = 0;

    virtual void Free(void* ptr) = 0;

    virtual ~ICTAllocator(void) { }
};

class ICTMemoryPool
{
public:
    virtual ICTInterface* Alloc(size_t size) = 0;

    virtual bool Free(ICTInterface* ptr) = 0;

    virtual ~ICTMemoryPool(void) { }
};

ICTAllocator* CTMemAllocator(void);

ICTMemoryPool* CTMemPool(void);

void CTMemInit(void);

void CTMemRelease(void);

template <
    typename T
>
T* CTMemAllocObject(void)
{
    void* raw = CTMemPool()->Alloc(sizeof(T));
    T* ptr = new(raw) T();
    return ptr;
}

template <
    typename T
>
void CTMemFreeObject(T* ptr)
{
    if(!CTMemPool()->Free((ICTInterface*)ptr))
    {
        RawFree((void*)ptr);
    }
}