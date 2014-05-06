#include "geometry.h"
#include "ct.h"
#include "cuKDTree.h"
#include "cpuKDTree.h"
#include "memory.h"

#ifdef _DEBUG
#pragma comment(lib, "Nuttyx64Debug.lib")
#else
#pragma comment(lib, "Nuttyx64Release.lib")
#endif

bool g_bInitialized = false;
uint g_flags = 0;

bool APIENTRY DllMain(HINSTANCE hInst, DWORD reason, LPVOID lpReserved)
{
    return true;
}

CT_RESULT CT_API CTInit(uint flags)
{
    if(g_bInitialized || (flags & CT_ENABLE_CUDA_ACCEL) == 0)
    {
        return CT_INVALID_VALUE;
    }

    CTMemInit();

    nutty::Init(NULL);

    g_bInitialized = true;
    g_flags = flags;

    return CT_SUCCESS;
}

CT_RESULT CT_API CTRelease(void)
{
    if(g_bInitialized)
    {
        nutty::Release();
        CTMemRelease();
        g_bInitialized = false;
    }
    return CT_SUCCESS;
}

CT_RESULT CT_API CTCreateTree(ICTTree** tree, CT_TREE_DESC* desc)
{
    switch(desc->type)
    {
    case eCT_TreeTypeKD :
        {
            switch(desc->strategy)
            {
            case eCT_SplitStrategySAH :
                {
                    ICTTree* _tree;
                    if(desc->flags & CT_CREATE_TREE_CPU)
                    {
                       _tree = CTMemAllocObject<cpuKDTree>();
                    }
                    else
                    {
                        _tree = CTMemAllocObject<cuKDTree>();
                    }
                    *tree = _tree;
                    if(_tree == NULL)
                    {
                        return CT_MEMORY_ALLOC_FAILURE;
                    }
                } break;
            default : return CT_INVALID_ENUM;
            }
        } break;
    default : return CT_INVALID_ENUM;
    }
    return (*tree)->Init(desc->flags | (g_flags & CT_TREE_ENABLE_DEBUG_LAYER));
}

CT_RESULT CT_API CTCreateSAHKDTree(ICTTree** tree, uint flags)
{
    CT_TREE_DESC _desc;
    ZeroMemory(&_desc, sizeof(CT_TREE_DESC));
    _desc.strategy = eCT_SplitStrategySAH;
    _desc.type = eCT_TreeTypeKD;
    _desc.flags = flags;
    return CTCreateTree(tree, &_desc);
}

CT_RESULT CT_API CTCreateGeometry(ICTTree* tree, ICTGeometry** geo)
{
    if(tree->GetDeviceType() == eCT_CPU)
    {
        CT_RESULT res = CTAllocObjectRoutine<Geometry>((void**)geo);
        if(res == CT_SUCCESS)
        {
            ((Geometry*)(*geo))->SetTree((cpuKDTree*)tree);
        }
        return res;
    }
    else if(tree->GetDeviceType() == eCT_GPU)
    {
        return CTAllocObjectRoutine<GPUGeometry>((void**)geo);
    }
    else
    {
        return CT_INVALID_ENUM;
    }
}

CT_RESULT CT_API CTCreatePrimitive(ICTTree* tree, ICTPrimitive** prim)
{
    if(tree->GetDeviceType() == eCT_CPU && tree->GetTopology() == CT_TRIANGLES)
    {
        return CTAllocObjectRoutine<Triangle>((void**)prim);
    }
    else if(tree->GetDeviceType() == eCT_GPU && tree->GetTopology() == CT_TRIANGLES)
    {
        return CTAllocObjectRoutine<GPUTriangle>((void**)prim);
    }
    else
    {
        return CT_INVALID_ENUM;
    }
}

CT_RESULT CT_API CTCreateVertex(const ICTTree* tree, ICTVertex** v)
{
    if(tree->GetDeviceType() == eCT_CPU)
    {
        return CTAllocObjectRoutine<Vertex>((void**)v);
    }
    else if(tree->GetDeviceType() == eCT_GPU)
    {
        return CTAllocObjectRoutine<GPUVertex>((void**)v);
    }
    else
    {
        return CT_INVALID_ENUM;
    }
}

CT_RESULT CT_API CTReleaseObject(ICTInterface* obj)
{
    if(!g_bInitialized)
    {
        return CT_INVALID_VALUE;
    }
    CTMemFreeObject(obj);
    return CT_SUCCESS;
}

CT_RESULT CT_API CTReleaseTree(ICTTree* tree)
{
    CTMemFreeObject(tree);
    return CT_SUCCESS;
}