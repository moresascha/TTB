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
    CT_SAFE_CALL(CTCreateSAHKDTree(&tree, CT_CREATE_TREE_CPU));

    for(int gc = 0; gc < 16; ++gc)
    {
        ICTGeometry* geo;
        CT_SAFE_CALL(CTCreateGeometry(&geo));
        for(int i = 0; i < 4; ++i)
        {
            ICTVertex* v;
            CT_SAFE_CALL(CTCreateVertex(&v));
            ctfloat3 pos;
            pos.x = -1 + 2 * rand() / (float)RAND_MAX;
            pos.y = -1 + 2 * rand() / (float)RAND_MAX;
            pos.z = -1 + 2 * rand() / (float)RAND_MAX;
            v->SetPosition(pos);
            geo->AddVertex(v);
        }
        tree->AddGeometry(geo);
    }

    /*ICTMemoryView* inter;
    CT_SAFE_CALL(tree->QueryInterface(__uuidof(ICTMemoryView), (void**)&inter));

    ct_printf("%x\n", inter->GetMemory());*/

    ICTTreeNode* node = tree->GetNodesEntryPtr();

    CT_SAFE_CALL(tree->Update());

    //tree->DebugDraw(dbLayer);

    CT_SAFE_CALL(CTRelease());

    //system("pause");

    return 0;
}