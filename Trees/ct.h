#pragma once
#include <windows.h>
#include "ct_def.h"
#include "ct_base.h"
#include "ct_error.h"

//forward declaration
class ICTTree;
class ICTGeometry;
class ICTVertex;
class ICTAABB;
class ICTTreeDebugLayer;
class ICTPrimitive;
class CTTriangle;
struct float4;
struct ICTTreeNode;

struct float3;
struct float2;
typedef unsigned char CTbyte;
typedef float3 CTreal3;
typedef float4 CTreal4;
typedef float2 CTreal2;
typedef float CTreal;
typedef bool CTbool;
typedef int CTint;
typedef unsigned int CTuint;
typedef long long CTlong;
typedef unsigned long long CTulong;

typedef CTlong CTGeometryHandle;

typedef void(*OnNodeTraverse)(ICTTreeNode* node, void* userData);

enum CT_GEOMETRY_TOPOLOGY
{
    CT_TRIANGLES = 0
};

enum CT_PRIMITIVE_TYPE
{
    CT_TRIANGLE = 0
};

enum CT_TREE_DEVICE
{
    eCT_CPU,
    eCT_GPU
};

enum CT_TREE_TYPE
{
    eCT_TreeTypeKD = 0
};

enum CT_SPLIT_STRATEGY
{
    eCT_SAH = 0
};

struct CT_TREE_DESC
{
    CT_TREE_TYPE type;
    CT_SPLIT_STRATEGY strategy;
    CTuint flags;
};

enum CT_SPLIT_AXIS
{
    eCT_X = 0,
    eCT_Y = 1,
    eCT_Z = 2
};

enum CT_TREE_TRAVERSAL
{
    eCT_DEPTH_FIRST,
    eCT_BREADTH_FIRST,
    eCT_VEB
};

class ICTAABB : public ICTInterface
{
public:
    virtual void AddVertex(const CTreal3& v) = 0;

    virtual const CTreal3& GetMin(void) const = 0;

    virtual const CTreal3& GetMax(void) const = 0;

    virtual void ShrinkMax(CTbyte axis, CTreal v) = 0;

    virtual void ShrinkMin(CTbyte axis, CTreal v) = 0;

    virtual void Reset(void) = 0;

    virtual ~ICTAABB(void) {}
};

struct ICTTreeNode : public ICTInterface
{
    virtual CTbool IsLeaf(void) = 0;

    virtual CTuint LeftIndex(void) = 0;

    virtual CTuint RightIndex(void) = 0;

    virtual ICTTreeNode* LeftNode(void) = 0;

    virtual ICTTreeNode* RightNode(void) = 0;

    virtual CTuint GetPrimitiveCount(void) = 0;
 
    virtual CTuint GetPrimitive(CTuint index) = 0;

    virtual CT_SPLIT_AXIS GetSplitAxis(void) = 0;

    virtual CTreal GetSplit(void) = 0;

    static CTuuid uuid(void) { return "ICTTreeNode"; }

    ~ICTTreeNode(void) {}
};

CT_EXPORT CT_RESULT CT_API CTInit
    (
    CTuint flags
    );

CT_EXPORT CT_RESULT CT_API CTRelease
    (
    void
    );

CT_EXPORT CT_RESULT CT_API CTCreateSAHKDTree
    (
    ICTTree** tree,
    CTuint flags = 0
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

CT_EXPORT CT_RESULT CT_API CTAddGeometry
    (
    ICTTree* tree,
    ICTGeometry* geo,
    CTGeometryHandle* handle
    );

CT_EXPORT CT_RESULT CT_API CTTransformGeometryHandle
    (
    ICTTree* tree,
    CTGeometryHandle handle,
    const CTreal4* matrix4x4
    );

CT_EXPORT CT_RESULT CT_API CTTransformGeometry
    (
    ICTGeometry* geo,
    CTreal matrix[4][4]
    );

CT_EXPORT CT_RESULT CT_API CTAddPrimitive
    (
    ICTGeometry* geo,
    ICTPrimitive* prim
    );

CT_EXPORT CT_RESULT CT_API CTUpdate
    (
    ICTTree* tree
    );

CT_EXPORT CT_RESULT CT_API CTGetDepth
    (
    const ICTTree* tree,
    CTuint* depth
    );

CT_EXPORT CT_RESULT CT_API CTGetRawLinearMemory
    (
    const ICTTree* tree,
    CTuint* byteCnt,
    const void** memory
    );

CT_EXPORT CT_RESULT CTGetAxisAlignedBB
    (
    const ICTTree* tree, 
    const ICTAABB** aabb
    );

CT_EXPORT CT_RESULT CT_API CTGetLeafNodeCount
    (
    const ICTTree* tree,
    CTuint* count
    );

CT_EXPORT CT_RESULT CT_API CTGetInteriorNodeCount
    (
    const ICTTree* tree,
    CTuint* count
    );

CT_EXPORT CT_RESULT CT_API CTGetRootNode
    (
    const ICTTree* tree,
    ICTTreeNode** node
    );

CT_EXPORT CT_RESULT CT_API CTTraverse
    (
    ICTTree* tree,
    CT_TREE_TRAVERSAL order,
    OnNodeTraverse callBack,
    void* userData = NULL
    );

CT_EXPORT CT_RESULT CT_API CTReleaseObject
    (
    ICTInterface* obj
    );

CT_EXPORT CT_RESULT CT_API CTReleaseGeometry
    (
    ICTGeometry* geo
    );

CT_EXPORT CT_RESULT CT_API CTReleasePrimitive
    (
    ICTPrimitive* prim
    );

CT_EXPORT CT_RESULT CT_API CTReleaseTree
    (
    ICTTree* tree
    );