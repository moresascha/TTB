#pragma once
#include <windows.h>
#include "ct_def.h"
#include "ct_base.h"

//forward declaration
class ICTTree;
class ICTGeometry;
class ICTVertex;
class ICTAABB;
class ICTTreeDebugLayer;

/*struct ctfloat3
{
    float x, y, z;
};*/
struct float3;

typedef float3 ctfloat3;

typedef float ctReal;

enum CT_GEOMETRY_TOPOLOGY
{
    CT_TRIANGLES = 0,
    CT_POINTS = 1
};

enum CT_TREE_TYPE
{
    eCT_TreeTypeKD = 0
};

enum CT_SPLIT_STRATEGY
{
    eCT_SplitStrategySAH = 0
};

struct CT_TREE_DESC
{
    CT_TREE_TYPE type;
    CT_SPLIT_STRATEGY strategy;
    uint flags;
};

extern "C"
{
    CT_EXPORT CT_RESULT CT_API CTInit
        (
        uint flags
        );

    CT_EXPORT CT_RESULT CT_API CTRelease
        (
        void
        );
    
    CT_EXPORT CT_RESULT CT_API CTCreateSAHKDTree
        (
        ICTTree** tree,
        uint flags = 0
        );

    CT_EXPORT CT_RESULT CT_API CTCreateTree
        (
        ICTTree** tree,
        CT_TREE_DESC* desc
        );

    CT_EXPORT CT_RESULT CT_API CTCreateGeometry
        (
        ICTGeometry** geo
        );
    
    CT_EXPORT CT_RESULT CT_API CTCreateVertex
        (
        ICTVertex** vertex
        );

    CT_EXPORT CT_RESULT CT_API CTReleaseObject
        (
        ICTInterface* obj
        );

    //-- APIENTRY
    bool APIENTRY DllMain(HINSTANCE hInst, DWORD reason, LPVOID lpReserved);
};