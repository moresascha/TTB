#include "cpuKDTree.h"

#include "ct_runtime.h"

#include <memory>
#include <sstream>
#include "ct_output.h"

#include <iostream>

static struct LeakDetecter
{
    LeakDetecter(void)
    {
        //_CrtSetBreakAlloc(152);
    }
} detecter;

extern "C" void bucketTest(void);

int main(void)
{

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    CT_SAFE_CALL(CTInit(CT_ENABLE_CUDA_ACCEL | CT_TREE_ENABLE_DEBUG_LAYER));

    ICTTree* tree;
    CT_SAFE_CALL(CTCreateSAHKDTree(&tree, CT_CREATE_TREE_GPU));
    
    ICTGeometry* geo;
    CT_SAFE_CALL(CTCreateGeometry(tree, &geo));

    for(int gc = 0; gc < 16; ++gc)
    {
        ICTPrimitive* p;
        CT_SAFE_CALL(CTCreatePrimitive(tree, &p));
        geo->AddPrimitive(p);
        for(int i = 0; i < 4; ++i)
        {
            ICTVertex* v;
            CT_SAFE_CALL(CTCreateVertex(tree, &v));
            ctfloat3 pos;
            pos.x = -1 + 2 * rand() / (float)RAND_MAX;
            pos.y = -1 + 2 * rand() / (float)RAND_MAX;
            pos.z = -1 + 2 * rand() / (float)RAND_MAX;
            v->SetPosition(pos);
            p->AddVertex(v);
        }
    }

    CT_SAFE_CALL(tree->AddGeometry(geo));

    ICTMemoryView* inter;
    CT_SAFE_CALL(tree->QueryInterface<ICTMemoryView>(&inter));

    ct_printf("%p\n", inter->GetMemory());

    ICTTreeNode* node = tree->GetRoot();

    CT_SAFE_CALL(tree->Update());

    CT_SAFE_CALL(CTRelease());

    return 0;
}