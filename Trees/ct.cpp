#include "geometry.h"
#include "ct.h"
//#include "cuKDTree.h"
#include "traverse.h"
#include <Nutty.h>
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

CT_RESULT CT_API CTInit(CTuint flags)
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

CT_RESULT CT_API CTGetPrimitiveCount(const ICTTree* tree, CTuint* count)
{
    *count = tree->GetPrimitiveCount();
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
            case eCT_SAH :
                {
                    ICTTree* _tree;
                    if(desc->flags & CT_CREATE_TREE_CPU)
                    {
                       _tree = CTMemAllocObject<cpuKDTree>();
                    }
                    else
                    {
                        _tree = NULL;//CTMemAllocObject<cuKDTree>();
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

CT_RESULT CT_API CTCreateSAHKDTree(ICTTree** tree, CTuint flags)
{
    CT_TREE_DESC _desc;
    ZeroMemory(&_desc, sizeof(CT_TREE_DESC));
    _desc.strategy = eCT_SAH;
    _desc.type = eCT_TreeTypeKD;
    _desc.flags = flags;
    return CTCreateTree(tree, &_desc);
}

CT_RESULT CT_API CTCreateGeometry(ICTGeometry** geo)
{
    return CTAllocObjectRoutine<Geometry>((void**)geo);
}

CT_RESULT CT_API CTAddGeometry(ICTTree* tree, ICTGeometry* geo, CTGeometryHandle* handle)
{
    return tree->AddGeometry(geo, handle);
}

CT_RESULT CT_API CTAddPrimitive(ICTGeometry* geo, ICTPrimitive* prim)
{
    return geo->AddPrimitive(prim);
}

CT_RESULT CT_API CTUpdate(ICTTree* tree)
{
    return tree->Update();
}

CT_RESULT CT_API CTGetAxisAlignedBB(const ICTTree* tree, const ICTAABB ** aabb)
{
    *aabb = tree->GetAxisAlignedBB();
    return CT_SUCCESS;
}

CT_RESULT CT_API CTGetLinearMemory(const ICTTree* tree, CTuint* cnt, const void** memory, CT_LINEAR_MEMORY_TYPE type)
{
    *memory = tree->GetLinearMemory(type, cnt);
    if(*memory == NULL)
    {
        *memory = NULL;
        cnt = 0;
        return CT_INVALID_OPERATION;
    }
    return CT_SUCCESS;
}

CT_RESULT CT_API CTGetRawLinearMemory(const ICTTree* tree, CTuint* cnt, const void** memory)
{
    const CTreal* d = tree->GetRawPrimitives(cnt);
    if(d == NULL)
    {
        *memory = NULL;
        cnt = 0;
        return CT_INVALID_OPERATION;
    }
    *memory = d;
    return CT_SUCCESS;
}

CT_RESULT CT_API CTGetRootNode(const ICTTree* tree, ICTTreeNode** node)
{
    *node = tree->GetRoot();
    return *node != NULL ? CT_SUCCESS : CT_INVALID_OPERATION;
}

CT_RESULT CT_API CTGetDepth(const ICTTree* tree, CTuint* depth)
{
    *depth = tree->GetDepth();
    return CT_SUCCESS;
}

CT_RESULT CT_API CTGetLeafNodeCount(const ICTTree* tree, CTuint* count)
{
    *count = tree->GetLeafNodesCount();
    return CT_SUCCESS;
}

CT_RESULT CT_API CTGetInteriorNodeCount(const ICTTree* tree, CTuint* count)
{
    *count = tree->GetInteriorNodesCount();
    return CT_SUCCESS;
}

CT_RESULT CT_API CTTraverse(ICTTree* tree, CT_TREE_TRAVERSAL order, OnNodeTraverse callBack, void* userData)
{ 
    return tree->Traverse(order, callBack, userData);
}

CT_RESULT CT_API CTTransformGeometryHandle(ICTTree* tree, CTGeometryHandle handle, const CTreal4* matrix4x4)
{
    tree->TransformGeometry(handle, matrix4x4);
    return CT_SUCCESS;
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
    return CTReleaseObject(tree);
}

CT_RESULT CT_API CTReleaseGeometry(ICTGeometry* geo)
{
    return CTReleaseObject(geo);
}