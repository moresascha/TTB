#include "memory.h"
#include <map>

#define SAFE_DELETE(a) if( (a) != NULL ) delete (a); (a) = NULL;
#define SAFE_ARRAY_DELETE(a) if( (a) != NULL ) delete[] (a); (a) = NULL;

class DefaultAllocator : public ICTAllocator
{
public:
    void* Alloc(size_t size)
    {
        return malloc(size);
    }

    void Free(void* ptr)
    {
        free(ptr);
    }

    ~DefaultAllocator(void) { }
};

class DefaultMemoryPool : public ICTMemoryPool
{
    std::map<ICTInterface*, ICTInterface*> m_ptrs;
public:
    ICTInterface* Alloc(size_t size)
    {
        ICTInterface* ptr = (ICTInterface*)CTMemAllocator()->Alloc(size);
        m_ptrs.insert(std::pair<ICTInterface*, ICTInterface*>(ptr, ptr));
        return ptr;
    }

    bool Free(ICTInterface* ptr)
    {
        if(m_ptrs.find(ptr) != m_ptrs.end())
        {
            m_ptrs.erase(ptr);
            ptr->~ICTInterface();
            CTMemAllocator()->Free((void*)ptr);
        }
        return true;
    }

    ~DefaultMemoryPool(void) 
    {
        auto& it = m_ptrs.begin();
        while(it != m_ptrs.end())
        {
            it->second->~ICTInterface();
            CTMemAllocator()->Free(it->second);
            m_ptrs.erase(it++);
        }
        m_ptrs.clear();
    }
};

DefaultAllocator* g_pDefaultAllocator = NULL;
DefaultMemoryPool* g_pDefaultMemoryPool = NULL;

void CTMemInit(void)
{
    g_pDefaultAllocator = new DefaultAllocator();
    g_pDefaultMemoryPool = new DefaultMemoryPool();
}

void CTMemRelease(void)
{
    SAFE_DELETE(g_pDefaultMemoryPool);
    SAFE_DELETE(g_pDefaultAllocator);
}

ICTMemoryPool* CTMemPool(void)
{
    return g_pDefaultMemoryPool;
}

ICTAllocator* CTMemAllocator(void)
{
    return g_pDefaultAllocator;
}