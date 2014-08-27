#include "ct_runtime.h"

#include <memory>
#include <sstream>
#include <iostream>

static struct LeakDetecter
{
    LeakDetecter(void)
    {
        //_CrtSetBreakAlloc(152);
    }
} detecter;

void CompactionTest(void);

int main(void)
{

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_CHECK_CRT_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_EVERY_16_DF);
#endif

    CompactionTest();

//     CT_SAFE_CALL(CTInit(CT_ENABLE_CUDA_ACCEL | CT_TREE_ENABLE_DEBUG_LAYER));
// 
//     ICTTree* tree;
//     CT_SAFE_CALL(CTCreateSAHKDTree(&tree, CT_CREATE_TREE_GPU));
//     
//     ICTGeometry* geo;
//     CT_SAFE_CALL(CTCreateGeometry(&geo));
// 
//     for(int gc = 0; gc < 128; ++gc)
//     {
//         CTTriangle triangle;
//         for(int i = 0; i < 3; ++i)
//         {
//             CTreal3 pos;
//             pos.x = -1 + 2 * rand() / (float)RAND_MAX;
//             pos.y = -1 + 2 * rand() / (float)RAND_MAX;
//             pos.z = -1 + 2 * rand() / (float)RAND_MAX;
//             triangle.SetValue(i, pos);
//         }
//         CT_SAFE_CALL(CTAddPrimitive(geo, &triangle));
//     }
// 
//     CTGeometryHandle handle;
//     CT_SAFE_CALL(CTAddGeometry(tree, geo, &handle));
//     CTTransformGeometryHandle(tree, handle, NULL);
//     CT_SAFE_CALL(CTReleaseGeometry(geo));
// 
//     CT_SAFE_CALL(CTUpdate(tree));
// 
//     CT_SAFE_CALL(CTRelease());
    
    return 0;
}